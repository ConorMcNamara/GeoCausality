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


class AugmentedSyntheticControl(EconometricEstimator):
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
        lambda_: float | None = None,
    ) -> None:
        """A class to run Augmented Synthetic Control for our geo-test.

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
            Ridge parameter to use. If not specified, then it is calculated through cross-validation

        Notes
        -----
        Based on Ben-Michael, Feller & Rothstein :cite:`augsynth2021` as well as https://github.com/sdfordham/pysyncon/blob/main/pysyncon/augsynth.py
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
        self.groupby_x: np.ndarray | None = None
        self.groupby_x_cols: list[str] | None = None
        self.groupby_y: float | None = None
        self.daily_x: nw.DataFrame | None = None
        self.daily_y: np.ndarray | None = None
        self.dates: list[Any] | None = None
        self.lambda_ = lambda_
        self.prediction_pre: nw.DataFrame | None = None
        self.prediction_post: nw.DataFrame | None = None
        self.actual_pre: nw.DataFrame | None = None
        self.actual_post: nw.DataFrame | None = None

    def pre_process(self) -> "AugmentedSyntheticControl":
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
        # groupby_x: shape (1, n_geos) — one row (the mean y per geo)
        self.groupby_x = x_sum[self.y_variable].to_numpy().reshape(1, -1)
        self.groupby_x_cols = x_sum[self.geo_variable].to_list()

        y_sum = (
            self.data.filter((nw.col(self.treatment_variable) == 1) & (nw.col("treatment_period") == 0))
            .select([self.y_variable, self.date_variable])
            .group_by(self.date_variable)
            .agg(nw.col(self.y_variable).sum())
            .sort(self.date_variable)
        )
        self.groupby_y = float(y_sum[self.y_variable].to_numpy().mean())

        day_x = self.data.filter((nw.col(self.treatment_variable) == 0) & (nw.col("treatment_period") == 0)).select(
            [self.y_variable, self.geo_variable, self.date_variable]
        )
        # Pivot: rows=dates, cols=geos
        self.daily_x = nw.from_native(
            day_x.to_native().pivot(on=self.geo_variable, index=self.date_variable, values=self.y_variable),
            eager_only=True,
        )
        daily_y_df = (
            self.data.filter((nw.col(self.treatment_variable) == 1) & (nw.col("treatment_period") == 0))
            .select([self.y_variable, self.date_variable])
            .group_by(self.date_variable)
            .agg(nw.col(self.y_variable).sum())
            .sort(self.date_variable)
        )
        self.daily_y = daily_y_df[self.y_variable].to_numpy()
        return self

    def generate(self) -> "AugmentedSyntheticControl":
        self.model = self._create_model()
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
        # Pivot: rows=dates, cols=geos
        control_pre_pivot = nw.from_native(
            control_pre.to_native().pivot(on=self.geo_variable, index=self.date_variable, values=self.y_variable),
            eager_only=True,
        )
        control_post_pivot = nw.from_native(
            control_post.to_native().pivot(on=self.geo_variable, index=self.date_variable, values=self.y_variable),
            eager_only=True,
        )
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

    def _create_model(self) -> np.ndarray:
        if self.daily_x is None:
            raise ValueError("daily_x must not be None")
        if self.daily_y is None:
            raise ValueError("daily_y must not be None")
        daily_x_arr, daily_y_arr, groupby_x_arr, groupby_y_arr = self._normalize()
        x_stacked = np.vstack([daily_x_arr, groupby_x_arr])
        y_stacked = np.concatenate([daily_y_arr, groupby_y_arr])
        daily_x_mat = self.daily_x.drop(self.date_variable).to_numpy()
        if self.lambda_ is None:
            lambdas = self._generate_lambdas(daily_x_mat)
            lambdas, errors_means, errors_se = self._cross_validate(daily_x_mat, self.daily_y, lambdas)
            self.lambda_ = lambdas[errors_means.argmin()].item()
        n_r, _ = daily_x_mat.shape
        V_mat = np.diag(np.full(n_r, 1 / n_r))
        W = self._get_weights(
            V_matrix=V_mat,
            x=daily_x_mat,
            y=self.daily_y,
        )
        W_ridge = self._get_ridge_weights(y_stacked, x_stacked, W, self.lambda_)
        return W + W_ridge

    def _get_weights(self, V_matrix: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Creates our synthetic control using v, x and y

        Parameters
        ----------
        V_matrix : numpy array
            Our V matrix
        x : numpy array
            Our predictors
        y : numpy array
            What we're trying to predict

        Returns
        -------
        The weights for our model
        """
        _, n_c = x.shape

        P = x.T @ V_matrix @ x
        q = y.T @ V_matrix @ x

        bounds = Bounds(lb=np.full(n_c, 0.0), ub=np.full(n_c, 1.0))
        constraints = LinearConstraint(A=np.full(n_c, 1.0), lb=1.0, ub=1.0)

        x0 = np.full(n_c, 1 / n_c)
        res = minimize(
            fun=lambda x: self._loss_function(x, P, q),
            x0=x0,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
        )
        weights = res["x"]
        return weights

    @staticmethod
    def _get_ridge_weights(a: np.ndarray, b: np.ndarray, w: np.ndarray, lambda_: float | np.ndarray) -> np.ndarray:
        """Calculate the ridge adjustment to the weights.

        Parameters
        ----------
        a : numpy array
            Our y values
        b : numpy array
            Our x values
        w : numpy array
            Weights matrix
        lambda_ : numpy array
            Our penalty term

        Returns
        -------
        The weights for our ridge penalty
        """
        m = a - b @ w
        n = np.linalg.inv(b @ b.T + lambda_ * np.identity(b.shape[0]))
        return m @ n @ b

    def _normalize(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Normalise the data before the weight calculation."""
        if self.groupby_x is None:
            raise ValueError("groupby_x must not be None")
        if self.groupby_y is None:
            raise ValueError("groupby_y must not be None")
        if self.daily_x is None:
            raise ValueError("daily_x must not be None")
        if self.daily_y is None:
            raise ValueError("daily_y must not be None")

        daily_x_mat = self.daily_x.drop(self.date_variable).to_numpy()  # (n_dates, n_geos)
        daily_y_arr = self.daily_y  # (n_dates,)

        # groupby_x is (1, n_geos); demean along axis=1 (the geos axis)
        groupby_x_demean = self.groupby_x - self.groupby_x.mean(axis=1, keepdims=True)
        groupby_y_demean = self.groupby_y - self.groupby_y  # scalar - scalar = 0.0, shape: scalar

        # daily_x: demean each row (date) across geos
        daily_x_demean = daily_x_mat - daily_x_mat.mean(axis=1, keepdims=True)
        daily_y_demean = daily_y_arr - daily_y_arr.mean()

        groupby_x_std = groupby_x_demean.std(axis=1, keepdims=True, ddof=1)  # (1, 1)
        groupby_x_std = np.where(groupby_x_std == 0, 1.0, groupby_x_std)
        daily_x_std = daily_x_demean.std(ddof=1)

        groupby_x_normal = (groupby_x_demean / groupby_x_std) * daily_x_std
        groupby_y_normal = (groupby_y_demean / groupby_x_std.flatten()[0]) * daily_x_std

        return daily_x_demean, daily_y_demean, groupby_x_normal, np.array([groupby_y_normal])

    def _cross_validate(
        self, X: np.ndarray, Y: np.ndarray, lambdas: np.ndarray, holdout_len: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Method that calculates the mean error and standard error to the mean
        error using a cross-validation procedure for the given ridge parameter
        values.
        """
        n_rows = X.shape[0]
        V = np.identity(n_rows - holdout_len)
        res = list()
        for start in range(n_rows - holdout_len + 1):
            holdout = slice(start, start + holdout_len)
            train_mask = np.ones(n_rows, dtype=bool)
            train_mask[holdout] = False

            X_t = X[train_mask]
            X_v = X[holdout]
            Y_t = Y[train_mask]
            Y_v = Y[holdout]

            w = self._get_weights(V_matrix=V, x=X_t, y=Y_t)
            this_res = list()
            for lambda_ in lambdas:
                ridge_weights = self._get_ridge_weights(a=Y_t, b=X_t, w=w, lambda_=lambda_)
                W_aug = w + ridge_weights
                err = np.sum((Y_v - X_v @ W_aug) ** 2)
                this_res.append(err.item())
            res.append(this_res)
        means = np.array(res).mean(axis=0)
        ses = np.array(res).std(axis=0) / np.sqrt(len(lambdas))
        return lambdas, means, ses

    @staticmethod
    def _generate_lambdas(X: np.ndarray, lambda_min_ratio: float = 1e-08, n_lambda: int = 20) -> np.ndarray:
        """Generate a suitable set of lambdas to run the cross-validation procedure on.

        Parameters
        ----------
        X : pandas DataFrame
            A dataframe containing the values of our training data
        lambda_min_ratio : float, default=1e-08
            The scaling factor
        n_lambda : int, default=20
            The number of lambdas we wish to return

        Returns
        -------
        An array containing the lambdas needed to run Augmented Synthetic Control
        """
        _, s, _ = np.linalg.svd(X.T)
        lambda_max = np.power(s[0].item(), 2)
        scaler = np.power(lambda_min_ratio, (1 / n_lambda))
        return lambda_max * (np.power(scaler, np.ndarray(range(n_lambda))))

    @staticmethod
    def _loss_function(x: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        return 0.5 * x.T @ p @ x - q.T @ x

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
