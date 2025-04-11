from math import ceil
from typing import Union, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from plotly.subplots import make_subplots
from scipy.stats import t as t_dist
from tabulate import tabulate

from GeoCausality._base import MLEstimator


class GeoX(MLEstimator):
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
    ) -> None:
        """A class to run Geoexperiments for our geo-test.

        Parameters
        ----------
        data: pandas or polars data frame
            Our geo-based time-series data
        geo_variable: str
            The name of the variable representing our geo-data
        test_geos: list, optional
            The geos that were assigned treatment. If not provided, rely on treatment variable
        control_geos: list, optional
            The geos that were withheld from treatment. If not provided, rely on treatment variable
        treatment_variable: str, optional
            If test and control geos are not provided, the column denoting which is test and control. Assumes that
            1 is coded as "treatment" and 0 is coded as "control"
        date_variable: str
            The name of the variable representing our dates
        pre_period: str
            The time period used to train our models. Starts from the first date in our data to pre_period.
        post_period: str
            The time period used to evaluate our performance. Starts from post_period to the last date in our data
        y_variable: str
            The name of the variable representing the results of our data
        alpha: float, default=0.1
            The alpha level for our experiment
        msrp: float, default=0.0
            The average MSRP of our sale. Used to calculate incremental revenue.
        spend: float, default=0.0
            The amount we spent on our treatment. Used to calculate ROAS (return on ad spend)
             or cost-per-acquisition.

        Notes
        -----
        Based on https://github.com/google/matched_markets?tab=readme-ov-file
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
        self.intercept_test = None
        self.prediction_pre = None
        self.prediction_post = None
        self.dates = None

    def generate(self, rescale: float = 1.0) -> "GeoX":
        intercept_train = sm.add_constant(self.pre_control)
        self.model = sm.OLS(self.pre_test.values, intercept_train.values).fit()
        self.intercept_test = sm.add_constant(self.post_control.values)
        model_summary = self.model.get_prediction(self.intercept_test).summary_frame(
            alpha=self.alpha
        )
        self.post_test["counterfactual"] = model_summary["mean"]
        self.results = {
            "date": self.test_dates,
            "test": self.post_test["orders"],
            "control": self.post_control["orders"],
            "counterfactual": self.post_test["counterfactual"],
            "counterfactual_ci_lower": model_summary["obs_ci_lower"],
            "counterfactual_ci_upper": model_summary["obs_ci_upper"],
            "incrementality": self.post_test["orders"]
            - self.post_test["counterfactual"],
            "incrementality_ci_lower": self.post_test["orders"]
            - model_summary["obs_ci_upper"],
            "incrementality_ci_upper": self.post_test["orders"]
            - model_summary["obs_ci_lower"],
        }
        self.results["cumulative_incrementality"] = self.results[
            "incrementality"
        ].cumsum()
        ci_dict = self._get_cumulative_cis(rescale)
        self.results["cumulative_incrementality_ci_lower"] = ci_dict[
            "cumulative_ci_lower"
        ]
        self.results["cumulative_incrementality_ci_upper"] = ci_dict[
            "cumulative_ci_upper"
        ]
        self.results["p_value"] = ci_dict["p_value"]
        return self

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
        table_dict = {
            "Variant": [np.sum(self.results["test"])],
            "Baseline": [np.sum(self.results["counterfactual"])],
        }
        ci_alpha = self._get_ci_print()
        if lift in ["incremental", "absolute"]:
            table_dict["Metric"] = [self.y_variable]
            table_dict["Lift Type "] = ["Incremental"]
            table_dict["Lift"] = [
                f"""{ceil(self.results["cumulative_incrementality"].iloc[-1]):,}"""
            ]
            table_dict[f"{ci_alpha} Lower CI"] = [
                f"""{ceil(self.results["cumulative_incrementality_ci_lower"][-1]):,}"""
            ]
            table_dict[f"{ci_alpha} Upper CI"] = [
                f"""{ceil(self.results["cumulative_incrementality_ci_upper"][-1]):,}"""
            ]
        elif lift == "relative":
            table_dict["Metric"] = [self.y_variable]
            table_dict["Lift Type"] = ["Relative"]
            table_dict["Lift"] = [
                f"""{round(
                    float(self.results["cumulative_incrementality"].iloc[-1]) * 100
                    / np.sum(self.results["counterfactual"]), 2)}%"""
            ]
            table_dict[f"{ci_alpha} Lower CI"] = [
                f"""{round(
                     self.results["cumulative_incrementality_ci_lower"][-1] * 100
                    / np.sum(self.results["counterfactual"]), 2)}%"""
            ]
            table_dict[f"{ci_alpha} Upper CI"] = [
                f"""{round(
                    self.results["cumulative_incrementality_ci_upper"][-1] * 100
                    / np.sum(self.results["counterfactual"]), 2)}%"""
            ]
        elif lift == "revenue":
            table_dict["Metric"] = ["Revenue"]
            table_dict["Lift Type "] = ["Incremental"]
            table_dict["Lift"] = [
                f"""${round(self.results["cumulative_incrementality"].iloc[-1] * self.msrp, 2):,}"""
            ]
            table_dict[f"{ci_alpha} Lower CI"] = [
                f"""${round(self.results["cumulative_incrementality_ci_lower"][-1] * self.msrp, 2):,}"""
            ]
            table_dict[f"{ci_alpha} Upper CI"] = [
                f"""${round(self.results["cumulative_incrementality_ci_upper"][-1] * self.msrp, 2):,}"""
            ]
        else:
            table_dict["Metric"] = ["ROAS"]
            table_dict["Lift Type "] = ["Incremental"]
            roas_lift, roas_ci_lower, roas_ci_upper = self._get_roas()
            table_dict["Lift"] = [f"${round(roas_lift, 2)}"]
            table_dict[f"{ci_alpha} Lower CI"] = [f"${round(roas_ci_lower, 2)}"]
            table_dict[f"{ci_alpha} Upper CI"] = [f"${round(roas_ci_upper, 2)}"]
        table_dict["p_value"] = [self.results["p_value"][-1]]
        print(tabulate(table_dict, headers="keys", tablefmt="grid"))

    def _get_roas(self) -> tuple:
        lift = ceil(self.results["cumulative_incrementality"].iloc[-1])
        roas_lift = self.spend / lift if lift > 0 else np.inf
        ci_upper = ceil(self.results["cumulative_incrementality_ci_upper"][-1])
        roas_ci_lower = self.spend / ci_upper if ci_upper > 0 else np.inf
        ci_lower = ceil(self.results["cumulative_incrementality_ci_lower"][-1])
        roas_ci_upper = self.spend / ci_lower if ci_lower > 0 else np.inf
        return roas_lift, roas_ci_lower, roas_ci_upper

    def _cumulative_distribution(self, rescale: float = 1.0) -> "t_dist":
        """Calculates the shifted distribution of our cumulative data

        Parameters
        ----------
        rescale: float, default=1.0
            An additional scaling factor for our delta

        Returns
        -------
        Our shifted t-distribution, as explained in Section 9.1 of https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45950.pdf

        Notes
        -----
        Taken from https://github.com/google/matched_markets/blob/master/matched_markets/methodology/tbr.py
        """
        test_len = len(self.post_control)
        one_to_t = np.arange(1, test_len + 1)
        one_to_t.shape = (test_len, 1)
        control_matrix = sm.add_constant(self.results["control"])
        cumulative_control_t = control_matrix.cumsum() / one_to_t
        param_covariance = np.array(self.model.cov_params())
        var_params = []
        for t in range(test_len):
            # Sum of parameter variance terms from eqn 5 of Kerman 2017.
            var_t = (
                cumulative_control_t.iloc[t,].values
                @ param_covariance
                @ cumulative_control_t.iloc[t,].values.T
            )
            var_params.append(var_t)
        var_params = np.array(var_params).reshape(test_len, 1)
        var_from_params = var_params * pow(one_to_t, 2)
        sigma_square = self.model.scale
        var_from_observations = one_to_t * sigma_square
        delta_mean = (
            rescale * np.array(self.results["cumulative_incrementality"]).flatten()
        )
        delta_var = var_from_params + var_from_observations
        delta_scale = rescale * np.sqrt(delta_var).flatten()
        delta_df = self.model.df_resid
        return t_dist(delta_df, loc=delta_mean, scale=delta_scale)

    def _get_cumulative_cis(self, rescale: float = 1.0) -> dict:
        """From our shifted T-distribution, calculates the confidence intervals and p-values

        Parameters
        ----------
        rescale: float, default=1.0
            An additional scaling factor for our delta

        Returns
        -------
        ci_dict: dict
            A dictionary containing our confidence intervals as well as p-values.
        """
        delta = self._cumulative_distribution(rescale=rescale)
        test_len = len(self.post_control)
        ci_lower = delta.ppf(self.alpha / 2).reshape(test_len)
        ci_upper = delta.ppf(1 - self.alpha / 2).reshape(test_len)
        p_value = delta.cdf(0.0).reshape(test_len)
        ci_dict = {
            "cumulative_ci_lower": ci_lower,
            "cumulative_ci_upper": ci_upper,
            "p_value": p_value,
        }
        return ci_dict

    def plot(self) -> None:
        """Plots our actual results, our counterfactual, the pointwise difference and cumulative difference

        Returns
        -------
        Our three plots determining the results
        """
        self.dates = sorted(self.data[self.date_variable].unique())
        marketing_start = [
            date for date in self.dates if date >= pd.to_datetime(self.post_period)
        ]
        control_data = pd.concat([self.pre_control, self.post_control])
        counterfactual = self.model.predict(sm.add_constant(control_data))
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
                        [self.pre_test["orders"], self.post_test["orders"]]
                    ),
                    marker={"color": "blue"},
                    mode="lines",
                    name="Actual",
                ),
                go.Scatter(
                    x=self.dates,
                    y=counterfactual,
                    marker={"color": "red"},
                    mode="lines",
                    name="Counterfactual",
                ),
                go.Scatter(
                    name="Counterfactual Upper Bound",
                    x=marketing_start,
                    y=self.results["counterfactual_ci_upper"],
                    mode="lines",
                    marker=dict(color="red"),
                    line=dict(width=0),
                    showlegend=False,
                ),
                go.Scatter(
                    name="Counterfactual Lower Bound",
                    x=marketing_start,
                    y=self.results["counterfactual_ci_lower"],
                    marker=dict(color="red"),
                    line=dict(width=0),
                    mode="lines",
                    fillcolor="rgba(68, 68, 68, 0.3)",
                    fill="tonexty",
                    showlegend=False,
                ),
            ]
        )
        residuals = (
            np.concatenate([self.pre_test["orders"], self.post_test["orders"]])
            - counterfactual
        )
        middle_fig = go.Figure(
            [
                go.Scatter(
                    x=self.dates,
                    y=residuals,
                    marker={"color": "purple"},
                    mode="lines",
                    name="Residuals",
                ),
                go.Scatter(
                    name="Pointwise Difference Upper Bound",
                    x=marketing_start,
                    y=self.results["incrementality_ci_upper"],
                    mode="lines",
                    marker=dict(color="purple"),
                    line=dict(width=0),
                    showlegend=False,
                ),
                go.Scatter(
                    name="Pointwise Difference Lower Bound",
                    x=marketing_start,
                    y=self.results["incrementality_ci_lower"],
                    marker=dict(color="purple"),
                    line=dict(width=0),
                    mode="lines",
                    fillcolor="rgba(68, 68, 68, 0.3)",
                    fill="tonexty",
                    showlegend=False,
                ),
            ]
        )
        cum_resids = self.results["cumulative_incrementality"]
        bottom_fig = go.Figure(
            [
                go.Scatter(
                    x=marketing_start,
                    y=cum_resids,
                    marker={"color": "orange"},
                    mode="lines",
                    name="Cumulative Incrementality",
                ),
                go.Scatter(
                    name="Cumulative Difference Upper Bound",
                    x=marketing_start,
                    y=self.results["cumulative_incrementality_ci_upper"],
                    mode="lines",
                    marker=dict(color="orange"),
                    line=dict(width=0),
                    showlegend=False,
                ),
                go.Scatter(
                    name="Cumulative Difference Lower Bound",
                    x=marketing_start,
                    y=self.results["cumulative_incrementality_ci_lower"],
                    marker=dict(color="orange"),
                    line=dict(width=0),
                    mode="lines",
                    fillcolor="rgba(68, 68, 68, 0.3)",
                    fill="tonexty",
                    showlegend=False,
                ),
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
