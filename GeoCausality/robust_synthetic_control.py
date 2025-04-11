from math import ceil
from typing import Union, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tabulate import tabulate

from GeoCausality._base import EconometricEstimator


class RobustSyntheticControl(EconometricEstimator):

    def __init__(
        self,
        data: Union[pd.DataFrame],
        geo_variable: str = None,
        test_geos: Optional[list[str]] = None,
        control_geos: Optional[list[str]] = None,
        treatment_variable: Optional[str] = None,
        date_variable: str = "date",
        pre_period: str = None,
        post_period: str = None,
        y_variable: str = None,
        alpha: float = 0.1,
        msrp: float = 0.0,
        spend: float = 0.0,
        lambda_: float = 0.1,
        threshold: Optional[float] = None,
        sv_count: Optional[int] = None,
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
        self.dates = None
        self.prediction_pre = None
        self.prediction_post = None
        self.actual_pre = None
        self.actual_post = None
        self.lambda_ = lambda_
        if (threshold is None) and (sv_count is None):
            raise ValueError("At least one of `threshold` or `sv_count` cannot be None")
        self.threshold = threshold
        self.sv_count = sv_count
        self.daily_x = None
        self.daily_y = None

    def pre_process(self) -> "RobustSyntheticControl":
        super().pre_process()
        self.dates = sorted(self.data[self.date_variable].unique())
        day_x = self.data.loc[
            (self.data[self.treatment_variable] == 0),
            [self.y_variable, self.geo_variable, self.date_variable],
        ]
        self.daily_x = day_x.pivot(index=self.date_variable, columns=self.geo_variable)[
            [self.y_variable]
        ]
        daily_y = (
            self.data.loc[
                (self.data[self.treatment_variable] == 1),
                [self.y_variable, self.date_variable],
            ]
            .groupby([self.date_variable])[self.y_variable]
            .sum()
            .reset_index()
        )
        self.daily_y = daily_y.set_index(self.date_variable)
        return self

    def generate(self) -> "RobustSyntheticControl":
        self.model = self._create_model()
        self.actual_pre = (
            self.data.loc[
                (self.data[self.treatment_variable] == 1)
                & (self.data["treatment_period"] == 0),
                [self.y_variable, self.date_variable],
            ]
            .groupby([self.date_variable])[self.y_variable]
            .sum()
            .reset_index()
        )
        self.actual_post = (
            self.data.loc[
                (self.data[self.treatment_variable] == 1)
                & (self.data["treatment_period"] == 1),
                [self.y_variable, self.date_variable],
            ]
            .groupby([self.date_variable])[self.y_variable]
            .sum()
            .reset_index()
        )
        control_pre = (
            self.data.loc[
                (self.data[self.treatment_variable] == 0)
                & (self.data["treatment_period"] == 0),
                [self.y_variable, self.date_variable, self.geo_variable],
            ]
            .groupby([self.date_variable, self.geo_variable])[self.y_variable]
            .sum()
            .reset_index()
        )
        control_post = (
            self.data.loc[
                (self.data[self.treatment_variable] == 0)
                & (self.data["treatment_period"] == 1),
                [self.y_variable, self.date_variable, self.geo_variable],
            ]
            .groupby([self.date_variable, self.geo_variable])[self.y_variable]
            .sum()
            .reset_index()
        )
        control_pre_pivot = control_pre.pivot(
            index=self.geo_variable, columns=self.date_variable
        )[[self.y_variable]].T
        control_post_pivot = control_post.pivot(
            index=self.geo_variable, columns=self.date_variable
        )[[self.y_variable]].T
        prediction_pre = control_pre_pivot @ self.model
        prediction_post = control_post_pivot @ self.model
        self.prediction_post = (
            prediction_post.reset_index()
            .drop(["level_0"], axis=1)
            .rename({0: self.y_variable}, axis=1)
        )
        self.prediction_pre = (
            prediction_pre.reset_index()
            .drop(["level_0"], axis=1)
            .rename({0: self.y_variable}, axis=1)
        )
        self.results = {
            "test": self.actual_post,
            "counterfactual": self.prediction_post,
            "lift": self.actual_post[self.y_variable]
            - self.prediction_post[self.y_variable],
        }
        self.results["incrementality"] = float(np.sum(self.results["lift"]))
        return self

    def _create_model(self) -> np.array:
        """Generates the weights used to predict our counterfactual

        Returns
        -------
        The weights matrix used to create our model
        """
        daily_x_transposed = self.daily_x.T.values
        M_hat = self._svd(daily_x_transposed).T
        time_end = pd.to_datetime(self.post_period)
        end_idx = self.daily_x.index.to_list().index(time_end)
        M_hat_neg = M_hat[:end_idx, :]
        Y1_neg = self.daily_y.to_numpy()[:end_idx]

        W = np.matmul(
            np.linalg.inv(
                M_hat_neg.T @ M_hat_neg + self.lambda_ * np.identity(M_hat_neg.shape[1])
            ),
            M_hat_neg.T @ Y1_neg,
        )
        return W

    def summarize(self, lift: str) -> None:
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
        ci_alpha = self._get_ci_print()
        if lift in ["incremental", "absolute"]:
            table_dict = {
                "Variant": [np.sum(self.results["test"][self.y_variable])],
                "Baseline": [np.sum(self.results["counterfactual"][self.y_variable])],
                "Metric": [self.y_variable],
                "Lift Type ": ["Incremental"],
                "Lift": [f"""{ceil(self.results["incrementality"]):,}"""],
            }
        elif lift == "relative":
            table_dict = {
                "Variant": [np.sum(self.results["test"][self.y_variable])],
                "Baseline": [np.sum(self.results["counterfactual"][self.y_variable])],
                "Metric": [self.y_variable],
                "Lift Type": ["Relative"],
                "Lift": [
                    f"""{round(
                    float(self.results["incrementality"]) * 100
                    / (np.sum(self.results["counterfactual"][self.y_variable])), 2)}%"""
                ],
            }
        elif lift == "revenue":
            table_dict = {
                "Variant": [
                    f"""${round(np.sum(self.results["test"][self.y_variable]) * self.msrp, 2):,}"""
                ],
                "Baseline": [
                    f"""${
                    round(
                       (np.sum(self.results["counterfactual"][self.y_variable])
                    )
                    * self.msrp, 2):,}"""
                ],
                "Metric": ["Revenue"],
                "Lift Type ": ["Incremental"],
                "Lift": [
                    f"""${round(self.results["incrementality"] * self.msrp, 2):,}"""
                ],
            }
        else:
            roas_lift, _, _ = self._get_roas()
            table_dict = {
                "Variant": [
                    f"""${round(self.spend / np.sum(self.results["test"][self.y_variable]), 2)}"""
                ],
                "Baseline": [
                    f"""${
                    round(self.spend
                    / (
                        np.sum(self.results["counterfactual"][self.y_variable])
                    ), 2)}"""
                ],
                "Metric": ["ROAS"],
                "Lift Type": ["Incremental"],
                "Lift": [f"${round(roas_lift, 2)}"],
            }
        print(tabulate(table_dict, headers="keys", tablefmt="grid"))

    def _get_roas(self) -> tuple:
        lift = ceil(self.results["incrementality"])
        roas_lift = self.spend / lift if lift > 0 else np.inf
        return roas_lift, 1, 2

    def _svd(self, groupby_x_transposed: np.array) -> np.array:
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
                            self.actual_pre[self.y_variable],
                            self.actual_post[self.y_variable],
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
                            self.prediction_pre[self.y_variable],
                            self.prediction_post[self.y_variable],
                        ]
                    ),
                    marker={"color": "red"},
                    mode="lines",
                    name="Counterfactual",
                ),
            ]
        )
        residuals = np.concatenate(
            [self.actual_pre[self.y_variable], self.actual_post[self.y_variable]]
        ) - np.concatenate(
            [
                self.prediction_pre[self.y_variable],
                self.prediction_post[self.y_variable],
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
        cum_resids = np.array(self.actual_post[self.y_variable]) - (
            np.array(self.prediction_post[self.y_variable])
        )
        marketing_start = [
            date for date in self.dates if date >= pd.to_datetime(self.post_period)
        ]
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
            for trace in range(len(figure["data"])):
                total_fig.add_trace(figure["data"][trace], row=i + 1, col=1)
                total_fig.add_vline(
                    x=self.post_period,
                    line_width=1,
                    line_dash="dash",
                    line_color="black",
                    row=i + 1,
                    col=1,
                )
        total_fig.show()
