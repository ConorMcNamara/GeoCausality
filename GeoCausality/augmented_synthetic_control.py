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
        if self.treatment_variable is None:
            raise ValueError("treatment_variable must not be None")
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
        if self.treatment_variable is None:
            raise ValueError("treatment_variable must not be None")
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
        # Cache the donor matrices for the shared faithful jackknife+ loop.
        self._jk_x_pre = control_pre_mat
        self._jk_x_post = control_post_mat
        self._jk_y_pre = self.actual_pre[self.y_variable].to_numpy()

        # Intercept-augmented (de-meaned) counterfactual: anchor the level at the
        # treated unit's own pre-period mean and track donor deviations from their
        # pre-period means, using the same means the weights were fit against.
        if self._x_bar is None or self._y_bar is None:
            raise ValueError("_x_bar and _y_bar must not be None")
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
        # Intercept-augmented ridge ASCM (Ben-Michael et al.): fit ridge-penalised
        # donor weights on the time-de-meaned pre-period (subtracting each donor's
        # and the treated unit's pre-period mean). The level is carried by the
        # stored means and the weights are NOT constrained to the simplex, so they
        # can match an aggregated treated unit outside the donor convex hull. Plain
        # simplex weights produce a raw weighted average of donor levels and bias
        # the counterfactual low. lambda is chosen by the one-standard-error rule,
        # which favours the more-regularised fit within one SE of the CV minimum --
        # the bare minimum under-regularises and inflates post-period extrapolation.
        daily_x_mat = self.daily_x.drop(self.date_variable).to_numpy()
        self._x_bar = daily_x_mat.mean(axis=0, keepdims=True)
        self._y_bar = float(self.daily_y.mean())
        daily_x_dm = daily_x_mat - self._x_bar
        daily_y_dm = self.daily_y - self._y_bar
        if self.lambda_ is None:
            lambdas = self._generate_lambdas(daily_x_dm)
            self.lambda_ = self._select_lambda(daily_x_dm, daily_y_dm, lambdas)
        return self._ridge_weights(daily_x_dm, daily_y_dm, self.lambda_)

    @staticmethod
    def _ridge_weights(x: np.ndarray, y: np.ndarray, lambda_: float) -> np.ndarray:
        """Ridge-regression donor weights on the de-meaned pre-period.

        Solves ``min_w ||y - x w||^2 + lambda ||w||^2`` in closed form. The weights
        are unconstrained (off-simplex), matching the augmented synthetic control
        formulation of Ben-Michael et al.

        Parameters
        ----------
        x : numpy array, shape (n_periods, n_donors)
            De-meaned donor matrix.
        y : numpy array, shape (n_periods,)
            De-meaned treated series.
        lambda_ : float
            Ridge penalty.

        Returns
        -------
        The donor weight vector, shape (n_donors,).
        """
        n_c = x.shape[1]
        return np.linalg.solve(x.T @ x + lambda_ * np.identity(n_c), x.T @ y)

    def _fit_predict_weights(self, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray | None:
        """Refit the intercept-augmented ridge weights on a subset and predict.

        The penalty ``self.lambda_`` is held at its full-fit value (not re-selected
        per fold) so the leave-one-out counterfactuals reflect the same model.

        Parameters
        ----------
        x_train : numpy array, shape (n_train, n_donors)
            Pre-period donor rows used to refit.
        y_train : numpy array, shape (n_train,)
            Treated pre-period series on the same rows.
        x_eval : numpy array, shape (n_eval, n_donors)
            Donor rows to predict.

        Returns
        -------
        The counterfactual for each ``x_eval`` row.
        """
        x_bar = x_train.mean(axis=0, keepdims=True)
        y_bar = float(y_train.mean())
        if self.lambda_ is None:
            raise ValueError("lambda_ must not be None")
        w = self._ridge_weights(x_train - x_bar, y_train - y_bar, self.lambda_)
        return y_bar + (x_eval - x_bar.flatten()) @ w

    def _select_lambda(self, x: np.ndarray, y: np.ndarray, lambdas: np.ndarray) -> float:
        """Select the ridge penalty by the one-standard-error rule.

        Leave-one-out cross-validation over the pre-period gives a mean squared
        error and its standard error per candidate lambda; the chosen lambda is the
        largest whose mean error is within one standard error of the minimum. That
        favours more regularisation (less extrapolation) than the bare CV minimum,
        which under-regularises and inflates the counterfactual's post-period
        movement.

        Parameters
        ----------
        x : numpy array, shape (n_periods, n_donors)
            De-meaned donor matrix.
        y : numpy array, shape (n_periods,)
            De-meaned treated series.
        lambdas : numpy array
            Candidate ridge penalties.

        Returns
        -------
        The selected ridge penalty.
        """
        lambdas = np.sort(lambdas)  # ascending, so the last within-threshold index is the largest lambda
        n = x.shape[0]
        fold_err = np.empty((n, lambdas.shape[0]))
        for i in range(n):
            mask = np.arange(n) != i
            gram = x[mask].T @ x[mask]
            rhs = x[mask].T @ y[mask]
            identity = np.identity(gram.shape[0])
            for j, lambda_ in enumerate(lambdas):
                w = np.linalg.solve(gram + lambda_ * identity, rhs)
                fold_err[i, j] = (y[i] - x[i] @ w) ** 2
        mean = fold_err.mean(axis=0)
        se = fold_err.std(axis=0, ddof=1) / np.sqrt(n)
        j_min = int(np.argmin(mean))
        within = np.flatnonzero(mean <= mean[j_min] + se[j_min])
        return float(lambdas[within[-1]])

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
