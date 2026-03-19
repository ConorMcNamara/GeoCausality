from datetime import date as date_cls
from math import ceil
from typing import Any

import narwhals as nw
import numpy as np
import plotly.graph_objects as go
import polars as pl
from narwhals.typing import IntoDataFrame
from plotly.subplots import make_subplots
from scipy.optimize import Bounds, LinearConstraint, minimize
from tabulate import tabulate  # type: ignore

from GeoCausality._base import EconometricEstimator


class PenalizedSyntheticControl(EconometricEstimator):
    def __init__(
        self,
        data: IntoDataFrame,
        geo_variable: str = "geo",
        test_geos: list[str] | None = None,
        control_geos: list[str] | None = None,
        treatment_variable: str | None = "is_treatment",
        date_variable: str = "date",
        pre_period: str = "2021-01-01",
        post_period: str = "2021-01-02",
        y_variable: str = "y",
        alpha: float = 0.1,
        msrp: float = 0.0,
        spend: float = 0.0,
        lambda_: float = 0.1,
    ) -> None:
        """A class to run Penalized Synthetic Control for our geo-test.

        Parameters
        ----------
        data : pandas or polars data frame
            Our geo-based time-series data
        geo_variable : str
            The name of the variable representing our geo-data
        test_geos : list, optional
            The geos that were assigned treatment. If not provided, rely on treatment variable
        control_geos : list, optional
            The geos that were withheld from treatment. If not provided, rely on treatment variable
        treatment_variable : str, optional
            If test and control geos are not provided, the column denoting which is test and control. Assumes that
            1 is coded as "treatment" and 0 is coded as "control"
        date_variable : str
            The name of the variable representing our dates
        pre_period : str
            The time period used to train our models. Starts from the first date in our data to pre_period.
        post_period : str
            The time period used to evaluate our performance. Starts from post_period to the last date in our data
        y_variable : str
            The name of the variable representing the results of our data
        alpha : float, default=0.1
            The alpha level for our experiment
        msrp : float, default=0.0
            The average MSRP of our sale. Used to calculate incremental revenue.
        spend : float, default=0.0
            The amount we spent on our treatment. Used to calculate ROAS (return on ad spend)
             or cost-per-acquisition.
        lambda_ : float, default=0.1
            Ridge parameter to use

        Notes
        -----
        Based on Abadie & L'Hour :cite:`penalized2021`
        """
        super().__init__(
            data,
            geo_variable,
            test_geos,
            control_geos,
            treatment_variable,
            date_variable,
            pre_period,
            post_period,
            y_variable,
            alpha,
            msrp,
            spend,
        )
        self.V: np.ndarray | None = None
        self.dates: list[Any] | None = None
        self.prediction_pre: nw.DataFrame | None = None
        self.prediction_post: nw.DataFrame | None = None
        self.actual_pre: nw.DataFrame | None = None
        self.actual_post: nw.DataFrame | None = None
        self.lambda_ = lambda_

    def pre_process(self) -> "PenalizedSyntheticControl":
        super().pre_process()
        self.dates = sorted(self.data[self.date_variable].unique().to_list())
        assert self.treatment_variable is not None
        x_sum = (
            self.data.filter((nw.col(self.treatment_variable) == 0) & (nw.col("treatment_period") == 0))
            .select([self.y_variable, self.geo_variable, self.date_variable])
            .group_by(self.geo_variable)
            .agg(nw.col(self.y_variable).mean())
            .sort(self.geo_variable)
        )
        # Build groupby_x as a numpy row-vector: shape (1, n_geos)
        # columns are geo names, single row is the mean y value
        groupby_x_arr = x_sum[self.y_variable].to_numpy().reshape(1, -1)
        groupby_x_cols = x_sum[self.geo_variable].to_list()

        y_sum = (
            self.data.filter((nw.col(self.treatment_variable) == 1) & (nw.col("treatment_period") == 0))
            .select([self.y_variable, self.date_variable])
            .group_by(self.date_variable)
            .agg(nw.col(self.y_variable).sum())
            .sort(self.date_variable)
        )
        groupby_y_scalar = float(y_sum[self.y_variable].to_numpy().mean())
        self._create_v(groupby_x_arr, groupby_x_cols, groupby_y_scalar)
        return self

    def generate(self) -> "PenalizedSyntheticControl":
        assert self.treatment_variable is not None
        self.actual_pre = (
            self.data.filter((nw.col(self.treatment_variable) == 1) & (nw.col("treatment_period") == 0))
            .select([self.y_variable, self.date_variable])
            .group_by(self.date_variable)
            .agg(nw.col(self.y_variable).sum())
            .sort(self.date_variable)
        )
        self.actual_post = (
            self.data.filter((nw.col(self.treatment_variable) == 1) & (nw.col("treatment_period") == 1))
            .select([self.y_variable, self.date_variable])
            .group_by(self.date_variable)
            .agg(nw.col(self.y_variable).sum())
            .sort(self.date_variable)
        )
        control_pre = (
            self.data.filter((nw.col(self.treatment_variable) == 0) & (nw.col("treatment_period") == 0))
            .select([self.y_variable, self.date_variable, self.geo_variable])
            .group_by([self.date_variable, self.geo_variable])
            .agg(nw.col(self.y_variable).sum())
            .sort([self.date_variable, self.geo_variable])
        )
        control_post = (
            self.data.filter((nw.col(self.treatment_variable) == 0) & (nw.col("treatment_period") == 1))
            .select([self.y_variable, self.date_variable, self.geo_variable])
            .group_by([self.date_variable, self.geo_variable])
            .agg(nw.col(self.y_variable).sum())
            .sort([self.date_variable, self.geo_variable])
        )
        # Pivot: rows=dates, cols=geos — equivalent to the transposed pandas pivot
        control_pre_pivot = nw.from_native(
            control_pre.to_native().pivot(on=self.geo_variable, index=self.date_variable, values=self.y_variable),
            eager_only=True,
        )
        control_post_pivot = nw.from_native(
            control_post.to_native().pivot(on=self.geo_variable, index=self.date_variable, values=self.y_variable),
            eager_only=True,
        )
        # Drop the index column before matrix multiply; keep only geo columns
        control_pre_mat = control_pre_pivot.drop(self.date_variable).to_numpy()
        control_post_mat = control_post_pivot.drop(self.date_variable).to_numpy()

        prediction_pre_arr = control_pre_mat @ self.model
        prediction_post_arr = control_post_mat @ self.model

        self.prediction_pre = nw.from_native(
            pl.DataFrame(
                {
                    self.date_variable: control_pre_pivot[self.date_variable].to_native(),
                    self.y_variable: prediction_pre_arr.flatten(),
                }
            ),
            eager_only=True,
        )
        self.prediction_post = nw.from_native(
            pl.DataFrame(
                {
                    self.date_variable: control_post_pivot[self.date_variable].to_native(),
                    self.y_variable: prediction_post_arr.flatten(),
                }
            ),
            eager_only=True,
        )
        self.results = {
            "test": self.actual_post,
            "counterfactual": self.prediction_post,
            "lift": self.actual_post[self.y_variable].to_numpy() - self.prediction_post[self.y_variable].to_numpy(),
        }
        self.results["incrementality"] = float(np.sum(self.results["lift"]))
        return self

    def summarize(self, lift: str) -> None:
        if self.results is None:
            raise ValueError("results must not be None")
        lift = lift.casefold()
        if lift not in [
            "absolute",
            "relative",
            "incremental",
            "cost-per",
            "revenue",
            "roas",
        ]:
            raise ValueError(
                f"Cannot measure {lift}. Choose one of `absolute`, `relative`,  `incremental`, `cost-per`, `revenue` "
                f"or `roas`"
            )
        # ci_alpha = self._get_ci_print()
        if lift in ["incremental", "absolute"]:
            table_dict = {
                "Variant": [self.results["test"][self.y_variable].sum()],
                "Baseline": [self.results["counterfactual"][self.y_variable].sum()],
                "Metric": [self.y_variable],
                "Lift Type ": ["Incremental"],
                "Lift": [f"""{ceil(self.results["incrementality"]):,}"""],
            }
        elif lift == "relative":
            table_dict = {
                "Variant": [self.results["test"][self.y_variable].sum()],
                "Baseline": [self.results["counterfactual"][self.y_variable].sum()],
                "Metric": [self.y_variable],
                "Lift Type": ["Relative"],
                "Lift": [
                    f"""{
                        round(
                            float(self.results["incrementality"])
                            * 100
                            / (self.results["counterfactual"][self.y_variable].sum()),
                            2,
                        )
                    }%"""
                ],
            }
        elif lift == "revenue":
            table_dict = {
                "Variant": [f"""${round(self.results["test"][self.y_variable].sum() * self.msrp, 2):,}"""],
                "Baseline": [f"""${round((self.results["counterfactual"][self.y_variable].sum()) * self.msrp, 2):,}"""],
                "Metric": ["Revenue"],
                "Lift Type ": ["Incremental"],
                "Lift": [f"""${round(self.results["incrementality"] * self.msrp, 2):,}"""],
            }
        else:
            roas_lift, _, _ = self._get_roas()
            table_dict = {
                "Variant": [f"""${round(self.spend / self.results["test"][self.y_variable].sum(), 2)}"""],
                "Baseline": [f"""${round(self.spend / (self.results["counterfactual"][self.y_variable].sum()), 2)}"""],
                "Metric": ["ROAS"],
                "Lift Type": ["Incremental"],
                "Lift": [f"${round(roas_lift, 2)}"],
            }
        print(tabulate(table_dict, headers="keys", tablefmt="grid"))

    def _get_roas(self) -> tuple[float, float, float]:
        if self.results is None:
            raise ValueError("results must not be None")
        lift = ceil(self.results["incrementality"])
        roas_lift = self.spend / lift if lift > 0 else np.inf
        return roas_lift, 1, 2

    def _create_v(
        self,
        groupby_x_arr: np.ndarray,
        groupby_x_cols: list[str],
        groupby_y_scalar: float,
    ) -> "PenalizedSyntheticControl":
        """Creates the V matrix used for calculating the weights of our model

        Parameters
        ----------
        groupby_x_arr : numpy array
            Contains the average y-variable of our control geos (shape: 1 x n_geos)
        groupby_x_cols : list of str
            Column names (geo names) for groupby_x
        groupby_y_scalar : float
            Contains the average cumulative y-variable of our test geos

        Returns
        -------
        Itself, but with the weights model created
        """
        # Stack control averages (1 x n_geos) and test mean (scalar) into one matrix
        # for scaling: shape (1, n_geos+1)
        X_full = np.hstack([groupby_x_arr, [[groupby_y_scalar]]])  # (1, n_geos+1)
        std_vals = X_full.std(axis=1, keepdims=True)
        std_vals = np.where(std_vals == 0, 1.0, std_vals)
        X_scaled = X_full / std_vals

        groupby_x_scaled = X_scaled[:, :-1]  # (1, n_geos)
        groupby_y_scaled = X_scaled[:, -1:]  # (1, 1)

        self.V = np.identity(groupby_x_scaled.shape[0])
        self.model = self._create_model(groupby_x_scaled, groupby_y_scaled.flatten())
        return self

    def _create_model(self, groupby_x: np.ndarray, groupby_y: np.ndarray) -> np.ndarray:
        """Creates the weights model used to establish our counterfactual

        Parameters
        ----------
        groupby_x : pandas DataFrame
            Contains the average y-variable of our control geos
        groupby_y : pandas Series
            Contains the average cumulative y-variable of our test geos

        Returns
        -------
        An array containing the weights of our model
        """
        if self.V is None:
            raise ValueError("V must not be None")
        if self.dates is None:
            raise ValueError("dates must not be None")
        n_r, n_c = groupby_x.shape
        diff = np.subtract(groupby_x, groupby_y.reshape(-1, 1))
        r = np.diag(diff.T @ self.V @ diff)
        p = groupby_x.T @ self.V @ groupby_x
        q = -1.0 * groupby_y.T @ self.V @ groupby_x + (self.lambda_ / 2.0) * r.T
        bounds = Bounds(lb=np.full(n_c, 0.0), ub=np.full(n_c, 1.0))
        x0 = np.full(n_c, 1 / n_c)
        if len(self.dates) < n_c:
            constraints = LinearConstraint(A=np.full(n_c, 1.0), lb=1.0, ub=1.0)
            res = minimize(
                fun=lambda x: self._loss_w(x, p, q),
                x0=x0,
                bounds=bounds,
                constraints=constraints,
                method="SLSQP",
            )
        else:
            res = minimize(
                fun=lambda x: self._loss_w(x, p, q),
                x0=x0,
                bounds=bounds,
                method="SLSQP",
            )
        weights = res["x"]
        return weights

    @staticmethod
    def _loss_w(x: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Calculates the loss function for our model weights matrix

        Parameters
        ----------
        x : numpy array
            Our predictors
        p : numpy array
            x.T * v * x
        q : numpy array
            -1.0 * y.T * v * x + (penalty / 2) * diagonal_matrix of our differences

        Returns
        -------
        The loss function for our model weights matrix
        """
        return q.T @ x + 0.5 * x.T @ p @ x

    def plot(self) -> None:
        """Plots our actual results, our counterfactual, the pointwise difference and cumulative difference

        Returns
        -------
        Our three plots determining the results
        """
        if self.actual_pre is None:
            raise ValueError("actual_pre must not be None")
        if self.actual_post is None:
            raise ValueError("actual_post must not be None")
        if self.prediction_pre is None:
            raise ValueError("prediction_pre must not be None")
        if self.prediction_post is None:
            raise ValueError("prediction_post must not be None")
        if self.dates is None:
            raise ValueError("dates must not be None")
        total_fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(
                "Expected vs Counterfactual",
                "Pointwise Difference",
                "Cumulative Difference",
            ),
        )
        top_fig = go.Figure(
            [
                go.Scatter(
                    x=self.dates,
                    y=np.concatenate(
                        [
                            self.actual_pre[self.y_variable].to_numpy(),
                            self.actual_post[self.y_variable].to_numpy(),
                        ]
                    ),
                    marker={"color": "blue"},
                    mode="lines",
                    name="Actual",
                ),
                go.Scatter(
                    x=self.dates,
                    y=np.concatenate(
                        [
                            self.prediction_pre[self.y_variable].to_numpy(),
                            self.prediction_post[self.y_variable].to_numpy(),
                        ]
                    ),
                    marker={"color": "red"},
                    mode="lines",
                    name="Counterfactual",
                ),
            ]
        )
        residuals = np.concatenate(
            [self.actual_pre[self.y_variable].to_numpy(), self.actual_post[self.y_variable].to_numpy()]
        ) - np.concatenate(
            [
                self.prediction_pre[self.y_variable].to_numpy(),
                self.prediction_post[self.y_variable].to_numpy(),
            ]
        )
        middle_fig = go.Figure(
            [
                go.Scatter(
                    x=self.dates,
                    y=residuals,
                    marker={"color": "purple"},
                    mode="lines",
                    name="Residuals",
                )
            ]
        )
        cum_resids = self.actual_post[self.y_variable].to_numpy() - self.prediction_post[self.y_variable].to_numpy()
        post_period_date = date_cls.fromisoformat(self.post_period)
        marketing_start = [d for d in self.dates if d >= post_period_date]
        bottom_fig = go.Figure(
            [
                go.Scatter(
                    x=marketing_start,
                    y=cum_resids.cumsum(),
                    marker={"color": "orange"},
                    mode="lines",
                    name="Cumulative Incrementality",
                )
            ]
        )
        figures = [top_fig, middle_fig, bottom_fig]
        for i, figure in enumerate(figures):
            for trace_data in figure.data:
                total_fig.add_trace(trace_data, row=i + 1, col=1)
                total_fig.add_vline(
                    x=self.post_period,
                    line_width=1,
                    line_dash="dash",
                    line_color="black",
                    row=i + 1,
                    col=1,
                )
        total_fig.show()
