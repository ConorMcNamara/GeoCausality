"""Augmented Synthetic Control method for geo-experiment causal inference."""

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
    """Run augmented synthetic control for our geo-test."""

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
        conformal_q: float = 1.0,
    ) -> None:
        """Initialize the augmented synthetic control estimator.

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
        conformal_q : float, default=1.0
            The exponent of the moving-block test statistic used for conformal
            inference (p-values and confidence intervals).

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
        self.daily_x: nw.DataFrame | None = None
        self.daily_y: np.ndarray | None = None
        # Per-donor and treated pre-period means, used to anchor the level of the
        # intercept-augmented (de-meaned) counterfactual.
        self._x_bar: np.ndarray | None = None
        self._y_bar: float | None = None
        self.dates: list[Any] | None = None
        self.lambda_ = lambda_
        self.prediction_pre: nw.DataFrame | None = None
        self.prediction_post: nw.DataFrame | None = None
        self.actual_pre: nw.DataFrame | None = None
        self.actual_post: nw.DataFrame | None = None
        self.conformal_q = conformal_q

    def pre_process(self) -> "AugmentedSyntheticControl":
        """Aggregate the pre-period control and test data into the matrices used to fit weights.

        Returns
        -------
        AugmentedSyntheticControl
            Itself, so it can be chained with generate().
        """
        super().pre_process()
        self.dates = sorted(self.data[self.date_variable].unique().to_list())
        assert self.treatment_variable is not None
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
        """Build the counterfactual from the fitted weights and compute lift and inference.

        Returns
        -------
        AugmentedSyntheticControl
            Itself, so it can be chained with summarize().
        """
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

        # Intercept-augmented (de-meaned) counterfactual: anchor the level at the
        # treated unit's own pre-period mean and track donor deviations from their
        # pre-period means, using the same means the weights were fit against.
        assert self._x_bar is not None and self._y_bar is not None
        donor_pre_mean = self._x_bar.flatten()
        prediction_pre_arr = self._y_bar + (control_pre_mat - donor_pre_mean) @ self.model
        prediction_post_arr = self._y_bar + (control_post_mat - donor_pre_mean) @ self.model

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
        self.results.update(
            self._conformal_inference(
                self.actual_pre[self.y_variable].to_numpy(),
                self.prediction_pre[self.y_variable].to_numpy(),
                self.actual_post[self.y_variable].to_numpy(),
                self.prediction_post[self.y_variable].to_numpy(),
                q=self.conformal_q,
            )
        )
        return self

    def summarize(self, lift: str) -> None:
        """Print a tabulated summary of the augmented synthetic control results.

        Parameters
        ----------
        lift : str
            The kind of lift to report. One of ``"absolute"``, ``"relative"``,
            ``"incremental"``, ``"cost-per"``, ``"revenue"`` or ``"roas"``.
        """
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
        ci_alpha = self._get_ci_print()
        baseline = self.results["counterfactual"][self.y_variable].sum()
        if lift in ["incremental", "absolute"]:
            table_dict = {
                "Variant": [self.results["test"][self.y_variable].sum()],
                "Baseline": [baseline],
                "Metric": [self.y_variable],
                "Lift Type ": ["Incremental"],
                "Lift": [f"""{ceil(self.results["incrementality"]):,}"""],
                f"{ci_alpha} Lower CI": [f"""{ceil(self.results["incrementality_ci_lower"]):,}"""],
                f"{ci_alpha} Upper CI": [f"""{ceil(self.results["incrementality_ci_upper"]):,}"""],
            }
        elif lift == "relative":
            table_dict = {
                "Variant": [self.results["test"][self.y_variable].sum()],
                "Baseline": [baseline],
                "Metric": [self.y_variable],
                "Lift Type": ["Relative"],
                "Lift": [f"""{round(float(self.results["incrementality"]) * 100 / baseline, 2)}%"""],
                f"{ci_alpha} Lower CI": [f"""{round(self.results["incrementality_ci_lower"] * 100 / baseline, 2)}%"""],
                f"{ci_alpha} Upper CI": [f"""{round(self.results["incrementality_ci_upper"] * 100 / baseline, 2)}%"""],
            }
        elif lift == "revenue":
            table_dict = {
                "Variant": [f"""${round(self.results["test"][self.y_variable].sum() * self.msrp, 2):,}"""],
                "Baseline": [f"""${round(baseline * self.msrp, 2):,}"""],
                "Metric": ["Revenue"],
                "Lift Type ": ["Incremental"],
                "Lift": [f"""${round(self.results["incrementality"] * self.msrp, 2):,}"""],
                f"{ci_alpha} Lower CI": [f"""${round(self.results["incrementality_ci_lower"] * self.msrp, 2):,}"""],
                f"{ci_alpha} Upper CI": [f"""${round(self.results["incrementality_ci_upper"] * self.msrp, 2):,}"""],
            }
        else:
            roas_lift, roas_ci_lower, roas_ci_upper = self._get_roas()
            table_dict = {
                "Variant": [f"""${round(self.spend / self.results["test"][self.y_variable].sum(), 2)}"""],
                "Baseline": [f"""${round(self.spend / baseline, 2)}"""],
                "Metric": ["ROAS"],
                "Lift Type": ["Incremental"],
                "Lift": [f"${round(roas_lift, 2)}"],
                f"{ci_alpha} Lower CI": [f"${round(roas_ci_lower, 2)}"],
                f"{ci_alpha} Upper CI": [f"${round(roas_ci_upper, 2)}"],
            }
        table_dict["p_value"] = [self.results["p_value"]]
        print(tabulate(table_dict, headers="keys", tablefmt="grid"))

    def _get_roas(self) -> tuple[float, float, float]:
        if self.results is None:
            raise ValueError("results must not be None")
        lift = ceil(self.results["incrementality"])
        roas_lift = self.spend / lift if lift > 0 else np.inf
        ci_upper = ceil(self.results["incrementality_ci_upper"])
        roas_ci_lower = self.spend / ci_upper if ci_upper > 0 else np.inf
        ci_lower = ceil(self.results["incrementality_ci_lower"])
        roas_ci_upper = self.spend / ci_lower if ci_lower > 0 else np.inf
        return roas_lift, roas_ci_lower, roas_ci_upper

    def _create_model(self) -> np.ndarray:
        if self.daily_x is None:
            raise ValueError("daily_x must not be None")
        if self.daily_y is None:
            raise ValueError("daily_y must not be None")
        # Fit in the time-de-meaned (intercept-augmented) space: subtract each
        # donor's and the treated unit's pre-period mean, so the weights model
        # deviations and the level is carried by the stored means. Without this
        # the simplex weights produce a raw weighted average of donor levels,
        # which cannot match an aggregated treated unit outside the donor convex
        # hull and biases the counterfactual low.
        daily_x_mat = self.daily_x.drop(self.date_variable).to_numpy()
        self._x_bar = daily_x_mat.mean(axis=0, keepdims=True)
        self._y_bar = float(self.daily_y.mean())
        daily_x_dm = daily_x_mat - self._x_bar
        daily_y_dm = self.daily_y - self._y_bar
        if self.lambda_ is None:
            lambdas = self._generate_lambdas(daily_x_dm)
            lambdas, errors_means, errors_se = self._cross_validate(daily_x_dm, daily_y_dm, lambdas)
            self.lambda_ = lambdas[errors_means.argmin()].item()
        n_r, _ = daily_x_dm.shape
        V_mat = np.diag(np.full(n_r, 1 / n_r))
        W = self._get_weights(V_matrix=V_mat, x=daily_x_dm, y=daily_y_dm)
        W_ridge = self._get_ridge_weights(daily_y_dm, daily_x_dm, W, self.lambda_)
        return W + W_ridge

    def _get_weights(self, V_matrix: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Create our synthetic control using v, x and y.

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

    def _cross_validate(
        self, X: np.ndarray, Y: np.ndarray, lambdas: np.ndarray, holdout_len: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the mean error and standard error of the mean error via cross-validation.

        Uses a cross-validation procedure across the given ridge parameter values.
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
        return lambda_max * (np.power(scaler, np.arange(n_lambda)))

    @staticmethod
    def _loss_function(x: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        return 0.5 * x.T @ p @ x - q.T @ x

    def plot(self) -> None:
        """Plot our actual results, our counterfactual, the pointwise difference and cumulative difference.

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
