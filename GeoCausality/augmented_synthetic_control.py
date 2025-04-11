from math import ceil
from typing import Union, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize, Bounds, LinearConstraint
from tabulate import tabulate

from GeoCausality._base import EconometricEstimator
from GeoCausality.utils import HoldoutSplitter


class AugmentedSyntheticControl(EconometricEstimator):
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
        self.groupby_x = None
        self.groupby_y = None
        self.daily_x = None
        self.daily_y = None
        self.dates = None
        self.lambda_ = lambda_

    def pre_process(self) -> "AugmentedSyntheticControl":
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
        self.groupby_x = groupby_x.drop([self.geo_variable], axis=0)
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
        self.groupby_y = pd.Series(
            y_sum[self.y_variable].mean(), name=-1, index=[self.y_variable]
        )
        day_x = self.data.loc[
            (self.data[self.treatment_variable] == 0)
            & (self.data["treatment_period"] == 0),
            [self.y_variable, self.geo_variable, self.date_variable],
        ]
        self.daily_x = day_x.pivot(index=self.date_variable, columns=self.geo_variable)[
            [self.y_variable]
        ]
        daily_y = (
            self.data.loc[
                (self.data[self.treatment_variable] == 1)
                & (self.data["treatment_period"] == 0),
                [self.y_variable, self.date_variable],
            ]
            .groupby([self.date_variable])[self.y_variable]
            .sum()
            .reset_index()
        )
        daily_y = daily_y.set_index(self.date_variable)
        self.daily_y = pd.Series(daily_y.values.flatten())
        return self

    def generate(self) -> "AugmentedSyntheticControl":
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

    def _create_model(self) -> np.array:
        daily_x_demean, daily_y_demean, groupby_x_normal, groupby_y_normal = (
            self._normalize()
        )
        daily_x_demean.columns = groupby_x_normal.columns
        x_stacked = pd.concat([daily_x_demean, groupby_x_normal], axis=0)
        y_stacked = pd.concat([daily_y_demean, groupby_y_normal], axis=0)
        if self.lambda_ is None:
            lambdas = self._generate_lambdas(self.daily_x)
            lambdas, errors_means, errors_se = self._cross_validate(
                self.daily_x, self.daily_y, lambdas
            )
            self.lambda_ = lambdas[errors_means.argmin()].item()
        n_r, _ = self.daily_x.shape
        V_mat = np.diag(np.full(n_r, 1 / n_r))
        W = self._get_weights(
            V_matrix=V_mat,
            x=self.daily_x.to_numpy(),
            y=self.daily_y.to_numpy(),
        )
        W_ridge = self._get_ridge_weights(
            y_stacked.to_numpy(), x_stacked.to_numpy(), W, self.lambda_
        )
        return W + W_ridge

    def _get_weights(self, V_matrix: np.array, x: np.array, y: np.array) -> np.array:
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
    def _get_ridge_weights(
        a: np.array, b: np.array, w: np.array, lambda_: np.array
    ) -> np.array:
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
    ) -> tuple:
        """Normalise the data before the weight calculation."""
        groupby_x_demean = self.groupby_x.subtract(self.groupby_x.mean(axis=1), axis=0)
        groupby_y_demean = self.groupby_y.subtract(self.groupby_y.mean(), axis=0)

        daily_x_demean = self.daily_x.subtract(self.daily_x.mean(axis=1), axis=0)
        daily_y_demean = self.daily_y.subtract(self.daily_y.mean(), axis=0)
        daily_y_demean.index = daily_x_demean.index

        groupby_x_std = groupby_x_demean.std(axis=1)
        daily_x_std = daily_x_demean.to_numpy().std(ddof=1).item()

        groupby_x_normal = groupby_x_demean.divide(groupby_x_std, axis=0) * daily_x_std
        groupby_y_normal = groupby_y_demean.divide(groupby_x_std, axis=0) * daily_x_std
        return daily_x_demean, daily_y_demean, groupby_x_normal, groupby_y_normal

    def _cross_validate(
        self, X: np.array, Y: np.array, lambdas: np.ndarray, holdout_len: int = 1
    ) -> tuple:
        """Method that calculates the mean error and standard error to the mean
        error using a cross-validation procedure for the given ridge parameter
        values.
        """
        V = np.identity(X.shape[0] - holdout_len)
        res = list()
        for X_t, X_v, Y_t, Y_v in HoldoutSplitter(X, Y, holdout_len=holdout_len):
            Y_t.index = X_t.index
            w = self._get_weights(V_matrix=V, x=X_t.to_numpy(), y=Y_t.to_numpy())
            this_res = list()
            for l in lambdas:
                ridge_weights = self._get_ridge_weights(a=Y_t, b=X_t, w=w, lambda_=l)
                W_aug = w + ridge_weights
                err = (Y_v - X_v @ W_aug).pow(2).sum()
                this_res.append(err.item())
            res.append(this_res)
        means = np.array(res).mean(axis=0)
        ses = np.array(res).std(axis=0) / np.sqrt(len(lambdas))
        return lambdas, means, ses

    @staticmethod
    def _generate_lambdas(
        X: pd.DataFrame, lambda_min_ratio: float = 1e-08, n_lambda: int = 20
    ) -> np.array:
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
        return lambda_max * (np.power(scaler, np.array(range(n_lambda))))

    @staticmethod
    def _loss_function(x: np.array, p: np.array, q: np.array):
        return 0.5 * x.T @ p @ x - q.T @ x

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
