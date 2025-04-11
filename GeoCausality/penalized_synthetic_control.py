from math import ceil
from typing import Union, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize, Bounds, LinearConstraint
from tabulate import tabulate

from GeoCausality._base import EconometricEstimator


class PenalizedSyntheticControl(EconometricEstimator):

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
        self.V = None
        self.dates = None
        self.prediction_pre = None
        self.prediction_post = None
        self.actual_pre = None
        self.actual_post = None
        self.lambda_ = lambda_

    def pre_process(self) -> "PenalizedSyntheticControl":
        super().pre_process()
        self.dates = sorted(self.data[self.date_variable].unique())
        x_sum = (
            self.data.loc[
                (self.data[self.treatment_variable] == 0)
                & (self.data["treatment_period"] == 0),
                [self.y_variable, self.geo_variable, self.date_variable],
            ]
            .groupby([self.geo_variable])[self.y_variable]
            .mean()
            .reset_index()
        )
        groupby_x = x_sum.T
        groupby_x.columns = x_sum[self.geo_variable]
        groupby_x = groupby_x.drop([self.geo_variable], axis=0)
        y_sum = (
            self.data.loc[
                (self.data[self.treatment_variable] == 1)
                & (self.data["treatment_period"] == 0),
                [self.y_variable, self.date_variable],
            ]
            .groupby([self.date_variable])[self.y_variable]
            .sum()
            .reset_index()
        )
        groupby_y = pd.Series(
            y_sum[self.y_variable].mean(), name=-1, index=[self.y_variable]
        )
        self._create_v(groupby_x, groupby_y)
        return self

    def generate(self) -> "PenalizedSyntheticControl":
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

    def _create_v(
        self, groupby_x: pd.DataFrame, groupby_y: pd.Series
    ) -> "PenalizedSyntheticControl":
        """Creates the V matrix used for calculating the weights of our model

        Parameters
        ----------
        groupby_x : pandas DataFrame
            Contains the average y-variable of our control geos
        groupby_y : pandas Series
            Contains the average cumulative y-variable of our test geos

        Returns
        -------
        Itself, but with the weights model created
        """
        X = pd.concat([groupby_x, groupby_y], axis=1)
        X_scaled = X.divide(X.std(axis=1), axis=0)
        groupby_x_scaled, groupby_y_scaled = (
            X_scaled.drop(columns=groupby_y.name),
            X_scaled[groupby_y.name],
        )
        groupby_x_scaled, groupby_y_scaled = (
            groupby_x_scaled.to_numpy(),
            groupby_y_scaled.to_numpy(),
        )
        self.V = np.identity(groupby_x_scaled.shape[0])
        self.model = self._create_model(groupby_x_scaled, groupby_y_scaled)
        return self

    def _create_model(self, groupby_x: np.array, groupby_y: np.array) -> np.array:
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
    def _loss_w(x: np.array, p: np.array, q: np.array) -> np.array:
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
