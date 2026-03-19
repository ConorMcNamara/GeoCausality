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


class SyntheticControl(EconometricEstimator):
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
    ) -> None:
        """A class to run Synthetic Control for our geo-test.

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

        Notes
        -----
        Based on https://matheusfacure.github.io/python-causality-handbook/15-Synthetic-Control.html
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
        self.synthetic_test_df: nw.DataFrame | None = None
        self.synthetic_control_df: nw.DataFrame | None = None
        self.actual_pre: np.ndarray | None = None
        self.actual_post: np.ndarray | None = None
        self.prediction_pre: np.ndarray | None = None
        self.prediction_post: np.ndarray | None = None
        self.dates: list[Any] | None = None

    def pre_process(self) -> "SyntheticControl":
        super().pre_process()
        assert self.treatment_variable is not None
        self.dates = sorted(self.data[self.date_variable].unique().to_list())
        test_pre = (
            self.data.filter((nw.col(self.treatment_variable) == 1) & (nw.col("treatment_period") == 0))
            .group_by(self.date_variable)
            .agg(nw.col(self.y_variable).sum())
            .sort(self.date_variable)
        )
        test_post = (
            self.data.filter((nw.col(self.treatment_variable) == 1) & (nw.col("treatment_period") == 1))
            .group_by(self.date_variable)
            .agg(nw.col(self.y_variable).sum())
            .sort(self.date_variable)
        )
        control_pre = (
            self.data.filter((nw.col(self.treatment_variable) == 0) & (nw.col("treatment_period") == 0))
            .group_by([self.date_variable, self.geo_variable])
            .agg(nw.col(self.y_variable).sum())
            .sort([self.date_variable, self.geo_variable])
        )
        control_post = (
            self.data.filter((nw.col(self.treatment_variable) == 0) & (nw.col("treatment_period") == 1))
            .group_by([self.date_variable, self.geo_variable])
            .agg(nw.col(self.y_variable).sum())
            .sort([self.date_variable, self.geo_variable])
        )
        control_pre_pivot = nw.from_native(
            control_pre.to_native().pivot(on=self.geo_variable, index=self.date_variable, values=self.y_variable),
            eager_only=True,
        )
        control_post_pivot = nw.from_native(
            control_post.to_native().pivot(on=self.geo_variable, index=self.date_variable, values=self.y_variable),
            eager_only=True,
        )
        self.synthetic_control_df = test_pre.join(control_pre_pivot, on=self.date_variable, how="left")
        self.synthetic_test_df = test_post.join(control_post_pivot, on=self.date_variable, how="left")
        return self

    def generate(self) -> "SyntheticControl":
        if self.synthetic_control_df is None:
            raise ValueError("synthetic_control_df must not be None")
        if self.synthetic_test_df is None:
            raise ValueError("synthetic_test_df must not be None")
        train_x = self.synthetic_control_df.drop([self.date_variable, self.y_variable])
        self.actual_pre = self.synthetic_control_df[self.y_variable].to_numpy()
        test_x = self.synthetic_test_df.drop([self.date_variable, self.y_variable])
        self.actual_post = self.synthetic_test_df[self.y_variable].to_numpy()
        self.model = self._create_model(self.actual_pre, train_x.to_numpy())
        self.prediction_pre = train_x.to_numpy() @ self.model
        self.prediction_post = test_x.to_numpy() @ self.model
        self.results = {
            "test": self.actual_post,
            "counterfactual": self.prediction_post,
            "lift": self.actual_post - self.prediction_post,
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
        table_dict = {
            "Variant": [np.sum(self.results["test"])],
            "Baseline": [np.sum(self.results["counterfactual"])],
        }
        # ci_alpha = self._get_ci_print()
        if lift in ["incremental", "absolute"]:
            table_dict["Metric"] = [self.y_variable]
            table_dict["Lift Type "] = ["Incremental"]
            table_dict["Lift"] = [f"""{ceil(self.results["incrementality"]):,}"""]
        elif lift == "relative":
            table_dict["Metric"] = [self.y_variable]
            table_dict["Lift Type"] = ["Relative"]
            table_dict["Lift"] = [
                f"""{round(float(self.results["incrementality"]) * 100 / np.sum(self.results["counterfactual"]), 2)}%"""
            ]
        elif lift == "revenue":
            table_dict["Metric"] = ["Revenue"]
            table_dict["Lift Type "] = ["Incremental"]
            table_dict["Lift"] = [f"""${round(self.results["incrementality"] * self.msrp, 2):,}"""]
        else:
            table_dict["Metric"] = ["ROAS"]
            table_dict["Lift Type "] = ["Incremental"]
            roas_lift, _, _ = self._get_roas()
            table_dict["Lift"] = [f"${round(roas_lift, 2)}"]
        print(tabulate(table_dict, headers="keys", tablefmt="grid"))

    def _get_roas(self) -> tuple[float, float, float]:
        if self.results is None:
            raise ValueError("results must not be None")
        lift = ceil(self.results["incrementality"])
        roas_lift = self.spend / lift if lift > 0 else np.inf
        return roas_lift, 1, 2

    @staticmethod
    def loss_square(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Loss function being the sum of squared distances

        Parameters
        ----------
        w : numpy array
            An array containing the weights applied to our X variables
        x : numpy array
            A multidimensional array containing the geos in our control group
        y : numpy array
            An array containing the values we are trying to predict

        Returns
        -------
        An array minimizing the squared distance between x and y with our weights applied
        """
        return (y - x @ w).T @ (y - x @ w)

    @staticmethod
    def loss_root(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Loss function being the root of the sum of squared distances

        Parameters
        ----------
        w : numpy array
            An array containing the weights applied to our X variables
        x : numpy array
            A multidimensional array containing the geos in our control group
        y : numpy array
            An array containing the values we are trying to predict

        Returns
        -------
        An array minimizing the root of the squared distance between x and y with our weights applied
        """
        return np.sqrt((y - x @ w).T @ (y - x @ w))

    def _create_model(self, y: Any, x: Any) -> np.ndarray:
        """Creates our OLS model for synthetic control, with the constraint that our weights must
        add up to 1.

        Parameters
        ----------
        y : numpy array
            An array containing the values we are trying to predict
        x : numpy array
            A multidimensional array containing the geos in our control group

        Returns
        -------
        An array containing the weights to be applied to each control geo
        """
        n_r, n_c = x.shape
        bounds = Bounds(lb=np.full(n_c, 0.0), ub=np.full(n_c, 1.0))
        constraints = LinearConstraint(A=np.full(n_c, 1.0), lb=1.0, ub=1.0)
        x0 = np.full(n_c, 1 / n_c)
        if n_r < n_c:
            res = minimize(
                fun=lambda w: self.loss_root(w, x, y),
                x0=x0,
                bounds=bounds,
                constraints=constraints,
                method="SLSQP",
            )
        else:
            res = minimize(
                fun=lambda w: self.loss_square(w, x, y),
                x0=x0,
                bounds=bounds,
                constraints=constraints,
                method="SLSQP",
            )
        weights = res["x"]
        return weights

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
                    y=np.concatenate([self.actual_pre, self.actual_post]),
                    marker={"color": "blue"},
                    mode="lines",
                    name="Actual",
                ),
                go.Scatter(
                    x=self.dates,
                    y=np.concatenate([self.prediction_pre, self.prediction_post]),
                    marker={"color": "red"},
                    mode="lines",
                    name="Counterfactual",
                ),
            ]
        )
        residuals = np.concatenate([self.actual_pre, self.actual_post]) - np.concatenate(
            [self.prediction_pre, self.prediction_post]
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
        cum_resids = self.actual_post - self.prediction_post
        post_period_date = date_cls.fromisoformat(self.post_period)
        marketing_start = [d for d in self.dates if d >= post_period_date]
        bottom_fig = go.Figure(
            [
                go.Scatter(
                    x=marketing_start,
                    y=np.cumsum(cum_resids),
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


class SyntheticControlV(EconometricEstimator):
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
    ) -> None:
        """A class to run Synthetic Control, specifically accounting for the V matrix, for our geo-test.

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

        Notes
        -----
        Based on Abadie & Gardeazabal :cite:`basque2003` and https://github.com/sdfordham/pysyncon/blob/main/pysyncon/synth.py
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

    def pre_process(self) -> "SyntheticControlV":
        super().pre_process()
        assert self.treatment_variable is not None
        self.dates = sorted(self.data[self.date_variable].unique().to_list())
        x_sum = (
            self.data.filter((nw.col(self.treatment_variable) == 0) & (nw.col("treatment_period") == 0))
            .group_by(self.geo_variable)
            .agg(nw.col(self.y_variable).mean())
            .sort(self.geo_variable)
        )
        # groupby_x: 1-row numpy array with one value per control geo (mean y per geo)
        groupby_x_arr: np.ndarray = x_sum[self.y_variable].to_numpy().reshape(1, -1)

        y_sum = (
            self.data.filter((nw.col(self.treatment_variable) == 1) & (nw.col("treatment_period") == 0))
            .group_by(self.date_variable)
            .agg(nw.col(self.y_variable).sum())
            .sort(self.date_variable)
        )
        # groupby_y: scalar mean of test y, shaped as (1,) array
        groupby_y_arr: np.ndarray = np.array([y_sum[self.y_variable].to_numpy().mean()])

        day_x = self.data.filter((nw.col(self.treatment_variable) == 0) & (nw.col("treatment_period") == 0)).select(
            [self.y_variable, self.geo_variable, self.date_variable]
        )
        # daily_x: rows=dates, columns=geos, values=y
        daily_x_pivot = nw.from_native(
            day_x.to_native().pivot(on=self.geo_variable, index=self.date_variable, values=self.y_variable),
            eager_only=True,
        )
        daily_x_arr: np.ndarray = daily_x_pivot.drop(self.date_variable).to_numpy()

        daily_y = (
            self.data.filter((nw.col(self.treatment_variable) == 1) & (nw.col("treatment_period") == 0))
            .group_by(self.date_variable)
            .agg(nw.col(self.y_variable).sum())
            .sort(self.date_variable)
        )
        daily_y_arr: np.ndarray = daily_y[self.y_variable].to_numpy()

        self._create_v(groupby_x_arr, groupby_y_arr, daily_x_arr, daily_y_arr)
        return self

    def generate(self) -> "SyntheticControlV":
        assert self.treatment_variable is not None
        self.actual_pre = (
            self.data.filter((nw.col(self.treatment_variable) == 1) & (nw.col("treatment_period") == 0))
            .group_by(self.date_variable)
            .agg(nw.col(self.y_variable).sum())
            .sort(self.date_variable)
        )
        self.actual_post = (
            self.data.filter((nw.col(self.treatment_variable) == 1) & (nw.col("treatment_period") == 1))
            .group_by(self.date_variable)
            .agg(nw.col(self.y_variable).sum())
            .sort(self.date_variable)
        )
        control_pre = (
            self.data.filter((nw.col(self.treatment_variable) == 0) & (nw.col("treatment_period") == 0))
            .group_by([self.date_variable, self.geo_variable])
            .agg(nw.col(self.y_variable).sum())
            .sort([self.date_variable, self.geo_variable])
        )
        control_post = (
            self.data.filter((nw.col(self.treatment_variable) == 0) & (nw.col("treatment_period") == 1))
            .group_by([self.date_variable, self.geo_variable])
            .agg(nw.col(self.y_variable).sum())
            .sort([self.date_variable, self.geo_variable])
        )
        control_pre_pivot = nw.from_native(
            control_pre.to_native().pivot(on=self.geo_variable, index=self.date_variable, values=self.y_variable),
            eager_only=True,
        )
        control_post_pivot = nw.from_native(
            control_post.to_native().pivot(on=self.geo_variable, index=self.date_variable, values=self.y_variable),
            eager_only=True,
        )
        control_pre_arr = control_pre_pivot.drop(self.date_variable).to_numpy()
        control_post_arr = control_post_pivot.drop(self.date_variable).to_numpy()
        prediction_pre_arr = control_pre_arr @ self.model
        prediction_post_arr = control_post_arr @ self.model
        self.prediction_post = nw.from_native(
            pl.DataFrame(
                {
                    self.date_variable: control_post_pivot[self.date_variable].to_native(),
                    self.y_variable: prediction_post_arr,
                }
            ),
            eager_only=True,
        )
        self.prediction_pre = nw.from_native(
            pl.DataFrame(
                {
                    self.date_variable: control_pre_pivot[self.date_variable].to_native(),
                    self.y_variable: prediction_pre_arr,
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

    def _create_model(self, v: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Creates our synthetic control using v, x and y

        Parameters
        ----------
        v : numpy array
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
        bounds = Bounds(lb=np.full(n_c, 0.0), ub=np.full(n_c, 1.0))
        x0 = np.full(n_c, 1 / n_c)
        p = x.T @ v @ x
        q = y.T @ v @ x
        # constraints = LinearConstraint(A=np.full(n_c, 1.0), lb=1.0, ub=1.0)
        res = minimize(
            fun=lambda a: self._loss_w(a, p, q),
            x0=x0,
            bounds=bounds,
            # constraints=constraints,
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
            y.T * v * x

        Returns
        -------
        The loss function for our model weights matrix
        """
        return 0.5 * x.T @ p @ x - q.T @ x

    def _create_v(
        self,
        groupby_x: np.ndarray,
        groupby_y: np.ndarray,
        daily_x: np.ndarray,
        daily_y: np.ndarray,
    ) -> "SyntheticControlV":
        """Finds the V matrix so that we can create our model

        Parameters
        ----------
        groupby_x : numpy array
            Contains the average y-variable of our control geos
        groupby_y : numpy array
            Contains the average cumulative y-variable of our test geos
        daily_x : numpy array
            Contains the daily y-variable of our control geos
        daily_y : numpy array
            Contains the daily cumulative y-variable of our test geos

        Returns
        -------
        Itself, to be chained with other methods
        """
        # X is shape (1, n_geos+1): concat control means and test mean as last column
        X = np.hstack([groupby_x, groupby_y.reshape(1, -1)])  # (1, n_geos+1)
        # Scale each row by its std (here only 1 row, so divide by scalar std)
        row_std = np.std(X, axis=1, keepdims=True)
        row_std = np.where(row_std == 0, 1.0, row_std)
        X_scaled = X / row_std
        groupby_x_arr: np.ndarray = X_scaled[:, :-1]  # (1, n_geos)
        groupby_y_arr: np.ndarray = X_scaled[:, -1]  # (1,)
        daily_x_arr: np.ndarray = daily_x
        daily_y_arr: np.ndarray = daily_y

        n_r, _ = groupby_x_arr.shape
        groupby_arr = np.hstack([groupby_x_arr, groupby_y_arr.reshape(-1, 1)])
        groupby_arr = np.hstack([np.full((groupby_arr.shape[1], 1), 1), groupby_arr.T])

        daily_arr = np.hstack([daily_x_arr, daily_y_arr.reshape(-1, 1)])

        try:
            beta = (np.linalg.inv(groupby_arr.T @ groupby_arr) @ groupby_arr.T @ daily_arr.T)[1:,]
        except np.linalg.LinAlgError:
            raise ValueError("Could not invert X^T.X. There is most likely collinearity in your data.")
        x0 = np.diag(beta @ beta.T)
        res = minimize(
            fun=lambda x: self._loss_v(x, groupby_x_arr, groupby_y_arr, daily_x_arr, daily_y_arr.reshape(-1)),
            x0=x0,
            method="SLSQP",
        )
        self.V = np.diag(np.abs(res["x"])) / np.sum(np.abs(res["x"]))
        self.model = self._create_model(v=self.V, y=groupby_y_arr, x=groupby_x_arr)
        return self

    def _loss_v(
        self,
        x: np.ndarray,
        groupby_x: np.ndarray,
        groupby_y: np.ndarray,
        daily_x: np.ndarray,
        daily_y: np.ndarray,
    ) -> np.ndarray:
        """Generates the weights and loss of our V matrix

        Parameters
        ----------
        x : numpy array
            Our initial guess for the V matrix
        groupby_x : numpy array
            Contains the average y-variable of our control geos
        groupby_y : numpy array
            Contains the average y-variable of our test geos
        daily_x : numpy array
            Contains the daily y-variable of our control geos
        daily_y : numpy array
            Contains the daily cumulative y-variable of our test geos

        Returns
        -------
        The loss weights used to calculate V
        """
        v = np.diag(np.abs(x)) / np.sum(np.abs(x))
        W = self._create_model(v, groupby_x, groupby_y)
        loss_V = self.calc_loss_v(W=W, x=daily_x, y=daily_y)
        return loss_V

    @staticmethod
    def calc_loss_v(W: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculates the V loss.

        Parameters
        ----------
        W : numpy array
            Vector of the control weights
        x : numpy array
            Matrix of the time series of the outcome variable with each
            column corresponding to a control unit and the rows are the time
            steps.
        y : numpy array
            Column vector giving the outcome variable values over time for the
            treated unit

        Returns
        -------
        V loss.
        """
        loss_V = (y - x @ W).T @ (y - x @ W) / len(x)
        return loss_V

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
                    f"""{
                        round(
                            float(self.results["incrementality"])
                            * 100
                            / (np.sum(self.results["counterfactual"][self.y_variable])),
                            2,
                        )
                    }%"""
                ],
            }
        elif lift == "revenue":
            table_dict = {
                "Variant": [f"""${round(np.sum(self.results["test"][self.y_variable]) * self.msrp, 2):,}"""],
                "Baseline": [
                    f"""${round((np.sum(self.results["counterfactual"][self.y_variable])) * self.msrp, 2):,}"""
                ],
                "Metric": ["Revenue"],
                "Lift Type ": ["Incremental"],
                "Lift": [f"""${round(self.results["incrementality"] * self.msrp, 2):,}"""],
            }
        else:
            roas_lift, _, _ = self._get_roas()
            table_dict = {
                "Variant": [f"""${round(self.spend / np.sum(self.results["test"][self.y_variable]), 2)}"""],
                "Baseline": [
                    f"""${round(self.spend / (np.sum(self.results["counterfactual"][self.y_variable])), 2)}"""
                ],
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
                    y=np.cumsum(cum_resids),
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
