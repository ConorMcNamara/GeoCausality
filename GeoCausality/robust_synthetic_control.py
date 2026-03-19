from datetime import date as date_cls
from math import ceil
from typing import Any

import narwhals as nw
import numpy as np
import plotly.graph_objects as go
import polars as pl
from narwhals.typing import IntoDataFrame
from plotly.subplots import make_subplots
from tabulate import tabulate  # type: ignore

from GeoCausality._base import EconometricEstimator


class RobustSyntheticControl(EconometricEstimator):
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
        threshold: float | None = None,
        sv_count: int | None = None,
    ) -> None:
        """A class to run Robust Synthetic Control for our geo-test.

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
        threshold : float, optional
            Remove singular values that are less than this threshold.
        sv_count : int, optional
            Keep this many of the largest singular values when
            reducing the outcome matrix

        Notes
        -----
        Based on Amjad, Shah & Shen :cite:`robust2018` and https://github.com/sdfordham/pysyncon/blob/main/pysyncon/robust.py
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
        self.dates: list[Any] | None = None
        self.prediction_pre: nw.DataFrame | None = None
        self.prediction_post: nw.DataFrame | None = None
        self.actual_pre: nw.DataFrame | None = None
        self.actual_post: nw.DataFrame | None = None
        self.lambda_ = lambda_
        if (threshold is None) and (sv_count is None):
            raise ValueError("At least one of `threshold` or `sv_count` cannot be None")
        self.threshold = threshold
        self.sv_count = sv_count
        self.daily_x: nw.DataFrame | None = None
        self.daily_y: nw.DataFrame | None = None

    def pre_process(self) -> "RobustSyntheticControl":
        super().pre_process()
        self.dates = sorted(self.data[self.date_variable].unique().to_list())
        assert self.treatment_variable is not None
        day_x = self.data.filter(nw.col(self.treatment_variable) == 0).select(
            [self.y_variable, self.geo_variable, self.date_variable]
        )
        # Pivot: rows=dates, cols=geos
        self.daily_x = nw.from_native(
            day_x.to_native().pivot(on=self.geo_variable, index=self.date_variable, values=self.y_variable),
            eager_only=True,
        )
        daily_y = (
            self.data.filter(nw.col(self.treatment_variable) == 1)
            .select([self.y_variable, self.date_variable])
            .group_by(self.date_variable)
            .agg(nw.col(self.y_variable).sum())
            .sort(self.date_variable)
        )
        self.daily_y = daily_y
        return self

    def generate(self) -> "RobustSyntheticControl":
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
        self.results["incrementality"] = float(np.sum(self.results["lift"].to_numpy()))
        return self

    def _create_model(self) -> np.ndarray:
        """Generates the weights used to predict our counterfactual

        Returns
        -------
        The weights matrix used to create our model
        """
        if self.daily_x is None:
            raise ValueError("daily_x must not be None")
        if self.daily_y is None:
            raise ValueError("daily_y must not be None")
        # daily_x pivot has date column first; drop it to get geo columns only
        daily_x_mat = self.daily_x.drop(self.date_variable).to_numpy()
        daily_x_transposed = daily_x_mat.T
        M_hat = self._svd(daily_x_transposed).T
        post_period_date = date_cls.fromisoformat(self.post_period)
        date_list = self.daily_x[self.date_variable].to_list()
        end_idx = date_list.index(post_period_date)
        M_hat_neg = M_hat[:end_idx, :]
        Y1_neg = self.daily_y[self.y_variable].to_numpy()[:end_idx]

        W = np.matmul(
            np.linalg.inv(M_hat_neg.T @ M_hat_neg + self.lambda_ * np.identity(M_hat_neg.shape[1])),
            M_hat_neg.T @ Y1_neg,
        )
        return W

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

    def _svd(self, groupby_x_transposed: np.ndarray) -> np.ndarray:
        """Performs singular value decomposition of our groupby_x_transposed matrix

        Parameters
        ----------
        groupby_x_transposed : numpy array
            The transpose of our groupby_data. Formatted such that for each geo, we list the average
            y_metric specified in our class initiation

        Returns
        -------
        M_hat, a matrix based on our SVD.
        """
        u, s, v = np.linalg.svd(groupby_x_transposed)
        s_shape = s.shape[0] - 1
        if self.threshold:
            idx = 0
            while s[idx] > self.threshold and idx < s_shape:
                idx += 1
        else:
            if self.sv_count is None:
                raise ValueError("sv_count must not be None")
            idx = self.sv_count
        s_res = np.zeros_like(groupby_x_transposed)
        s_res[:idx, :idx] = np.diag(s[:idx])
        r, c = groupby_x_transposed.shape
        p_hat = max(np.count_nonzero(groupby_x_transposed) / (r * c), 1 / (r * c))
        M_hat = (1 / p_hat) * (u @ s_res @ v)
        return M_hat

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
