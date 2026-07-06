"""Synthetic Difference-in-Differences (Arkhangelsky et al. 2021) for geo-experiment causal inference."""

from math import ceil
from typing import Any

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrame
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.stats import norm
from tabulate import tabulate  # type: ignore

from GeoCausality._base import EconometricEstimator


class SyntheticDiffInDiff(EconometricEstimator):
    """Run synthetic difference-in-differences for our geo-test.

    Synthetic difference-in-differences (SDID) sits between plain
    difference-in-differences and the synthetic-control method. Like synthetic
    control it fits non-negative unit weights on the donor pool, but it fits them
    against the treated *trend* (an intercept absorbs any level gap, so the donors
    only need to move parallel to the treated unit, not match its level) with an
    L2 penalty. On top of that it fits non-negative *time* weights that focus the
    pre-period comparison on the periods most predictive of the post-period. The
    estimand is the scalar average treatment effect on the treated -- the doubly
    weighted difference-in-differences -- and inference is the placebo variance of
    Arkhangelsky et al. (2021).
    """

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
        zeta: float | None = None,
    ) -> None:
        """Initialize the synthetic difference-in-differences estimator.

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
        zeta : float, optional
            The L2 regularization strength on the unit weights. When ``None`` (the
            default) it is set to the Arkhangelsky et al. (2021) rule
            ``(n_post_periods) ** (1 / 4) * sd(first-differences of the donor
            outcomes over the pre-period)``.

        Notes
        -----
        Based on Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021),
        "Synthetic Difference-in-Differences", American Economic Review.
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
        self.zeta = zeta
        self.unit_weights: np.ndarray | None = None
        self.unit_intercept: float | None = None
        self.time_weights: np.ndarray | None = None
        self.att: float | None = None

    def pre_process(self) -> "SyntheticDiffInDiff":
        """Aggregate the control and test data into the matrices used to fit weights.

        Returns
        -------
        SyntheticDiffInDiff
            Itself, so it can be chained with generate().
        """
        super().pre_process()
        if self.treatment_variable is None:
            raise ValueError("treatment_variable must not be None")
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

    def generate(self) -> "SyntheticDiffInDiff":
        """Fit the unit and time weights, build the counterfactual, and compute the ATT and inference.

        Returns
        -------
        SyntheticDiffInDiff
            Itself, so it can be chained with summarize().
        """
        if self.synthetic_control_df is None:
            raise ValueError("synthetic_control_df must not be None")
        if self.synthetic_test_df is None:
            raise ValueError("synthetic_test_df must not be None")
        x_pre = self.synthetic_control_df.drop([self.date_variable, self.y_variable]).to_numpy()
        self.actual_pre = self.synthetic_control_df[self.y_variable].to_numpy()
        x_post = self.synthetic_test_df.drop([self.date_variable, self.y_variable]).to_numpy()
        self.actual_post = self.synthetic_test_df[self.y_variable].to_numpy()

        zeta = self.zeta if self.zeta is not None else self._default_zeta(x_pre)
        self.unit_intercept, self.unit_weights = self._solve_weights(x_pre, self.actual_pre, zeta)
        self.time_weights = self._solve_time_weights(x_pre, x_post)
        # Cache the donor matrices for the shared faithful jackknife+ loop.
        self._jk_x_pre = x_pre
        self._jk_x_post = x_post
        self._jk_y_pre = self.actual_pre

        # The SDID counterfactual is the donor-weighted trajectory shifted by the
        # unit fixed effect: alpha is the time-weighted pre-period gap between the
        # treated series and its synthetic control. The per-period gap of this
        # trajectory averages to the SDID ATT (the doubly-weighted DID), so the
        # summed lift matches the reported incrementality.
        alpha = float(self.time_weights @ (self.actual_pre - x_pre @ self.unit_weights))
        self.prediction_pre = x_pre @ self.unit_weights + alpha
        self.prediction_post = x_post @ self.unit_weights + alpha

        self.att = float(np.mean(self.actual_post - self.prediction_post))
        t1 = self.actual_post.shape[0]
        self.results = {
            "test": self.actual_post,
            "counterfactual": self.prediction_post,
            "lift": self.actual_post - self.prediction_post,
            "att": self.att,
            "incrementality": self.att * t1,
        }
        self.results.update(self._placebo_inference(x_pre, x_post, t1))
        return self

    def _default_zeta(self, x_pre: np.ndarray) -> float:
        """Arkhangelsky et al. (2021) default for the unit-weight L2 penalty.

        ``zeta = (n_post_periods) ** (1 / 4) * sd(first-differences of the donor
        outcomes over the pre-period)``. With a single (aggregated) treated unit
        the treated-unit factor drops out of the rule.

        Parameters
        ----------
        x_pre : numpy array, shape (n_pre, n_donors)
            Donor pre-period outcome matrix (rows = dates, cols = donor geos).

        Returns
        -------
        The regularization strength.
        """
        n_post = self.actual_post.shape[0] if self.actual_post is not None else 1
        if x_pre.shape[0] < 2:
            return 1.0
        diffs = np.diff(x_pre, axis=0)
        sigma = float(np.std(diffs))
        if sigma == 0.0:
            sigma = 1.0
        return float(n_post**0.25) * sigma

    def _solve_weights(self, x: np.ndarray, y: np.ndarray, zeta: float) -> tuple[float, np.ndarray]:
        """Fit a free intercept and simplex-constrained, L2-penalized donor weights.

        Solves ``min_{w0, w} ||y - (w0 + x @ w)||^2 + zeta^2 * n_rows * ||w||^2``
        subject to ``w >= 0`` and ``sum(w) = 1``. The free intercept ``w0`` is the
        unit fixed effect that lets the donors match the treated *trend* rather
        than its level, and is what distinguishes SDID's weights from plain
        synthetic control.

        Parameters
        ----------
        x : numpy array, shape (n_rows, n_cols)
            Predictor matrix (donor outcomes for the unit weights).
        y : numpy array, shape (n_rows,)
            Target series (the treated outcomes for the unit weights).
        zeta : float
            The L2 penalty strength on the weights.

        Returns
        -------
        A tuple of (intercept, weights).
        """
        n_r, n_c = x.shape
        penalty = zeta * zeta * n_r
        # Profile out the free intercept: for any weights the optimal intercept is
        # ``mean(y) - mean(x @ w)``, so centering both sides removes it from the
        # optimisation. Solving the well-conditioned centered problem (rather than
        # carrying a large-magnitude unbounded intercept variable) keeps SLSQP
        # stable at outcome scales such as GDP.
        x_bar = x.mean(axis=0)
        y_bar = float(y.mean())
        xc = x - x_bar
        yc = y - y_bar
        # Divide the objective by the target variance so its value and gradients
        # are O(1) regardless of the outcome scale. The minimiser is unchanged (a
        # positive constant does not move the argmin), but this keeps SLSQP's
        # relative convergence test from stopping early at large outcome scales
        # such as GDP -- otherwise the fitted weights, and hence the estimate,
        # drift with the platform's BLAS/LAPACK build.
        scale = max(float(np.var(yc)), 1e-12)
        bounds = Bounds(lb=np.full(n_c, 0.0), ub=np.full(n_c, 1.0))
        constraints = LinearConstraint(A=np.full(n_c, 1.0), lb=1.0, ub=1.0)
        w0 = np.full(n_c, 1.0 / n_c)

        def loss(w: np.ndarray) -> float:
            resid = yc - xc @ w
            return float((resid @ resid + penalty * (w @ w)) / scale)

        res = minimize(fun=loss, x0=w0, bounds=bounds, constraints=constraints, method="SLSQP")
        weights = res["x"]
        intercept = y_bar - float(x_bar @ weights)
        return intercept, weights

    def _solve_time_weights(self, x_pre: np.ndarray, x_post: np.ndarray) -> np.ndarray:
        """Fit non-negative time weights matching the donors' post-period average.

        Symmetric to the unit weights but over periods: each donor's post-period
        mean is regressed on its pre-period trajectory, so the fitted weights pick
        out the pre-periods most predictive of the post-period. A negligible ridge
        keeps the fit well posed when the pre-period is long relative to the donor
        pool.

        Parameters
        ----------
        x_pre : numpy array, shape (n_pre, n_donors)
            Donor pre-period outcome matrix.
        x_post : numpy array, shape (n_post, n_donors)
            Donor post-period outcome matrix.

        Returns
        -------
        The pre-period time weights, shape (n_pre,).
        """
        # Rows are donor units, columns are pre-periods; target is each donor's
        # post-period mean outcome.
        donor_pre = x_pre.T
        donor_post_mean = np.mean(x_post, axis=0)
        sigma = float(np.std(np.diff(x_pre, axis=0))) if x_pre.shape[0] > 1 else 1.0
        zeta_time = 1e-6 * (sigma if sigma > 0 else 1.0)
        _, weights = self._solve_weights(donor_pre, donor_post_mean, zeta_time)
        return weights

    def _sdid_att(
        self, x_pre: np.ndarray, x_post: np.ndarray, treated_pre: np.ndarray, treated_post: np.ndarray
    ) -> float:
        """Compute the doubly-weighted difference-in-differences from scratch.

        Fits fresh unit and time weights on the supplied donor pool and returns the
        SDID ATT for the supplied treated series. Used both for the point estimate's
        placebo replicates and could serve any refit-based routine.

        Parameters
        ----------
        x_pre, x_post : numpy array
            Donor pre- and post-period outcome matrices.
        treated_pre, treated_post : numpy array
            Treated pre- and post-period series.

        Returns
        -------
        The SDID average treatment effect on the treated.
        """
        zeta = self._default_zeta_for(x_pre, treated_post.shape[0])
        _, unit_w = self._solve_weights(x_pre, treated_pre, zeta)
        time_w = self._solve_time_weights(x_pre, x_post)
        treated_did = float(np.mean(treated_post) - time_w @ treated_pre)
        control_did = float(unit_w @ (np.mean(x_post, axis=0) - time_w @ x_pre))
        return treated_did - control_did

    @staticmethod
    def _default_zeta_for(x_pre: np.ndarray, n_post: int) -> float:
        """Compute the Arkhangelsky et al. zeta rule for an arbitrary donor pool.

        Parameters
        ----------
        x_pre : numpy array, shape (n_pre, n_donors)
            Donor pre-period outcome matrix.
        n_post : int
            Number of post-periods.

        Returns
        -------
        The regularization strength.
        """
        if x_pre.shape[0] < 2:
            return 1.0
        sigma = float(np.std(np.diff(x_pre, axis=0)))
        if sigma == 0.0:
            sigma = 1.0
        return float(max(n_post, 1) ** 0.25) * sigma

    def _placebo_inference(self, x_pre: np.ndarray, x_post: np.ndarray, t1: int) -> dict[str, Any]:
        """Placebo variance of the SDID ATT (Arkhangelsky et al. 2021, section 5).

        With a single (aggregated) treated unit the leave-one-out jackknife is
        undefined, so we use the placebo estimator: each donor is treated as a
        pseudo-treated unit against the remaining donors, and the variance of the
        resulting placebo ATTs estimates the sampling variance of the real ATT.
        Every placebo refits its own unit and time weights, so the inference is
        time-weighted exactly as the point estimate is.

        Parameters
        ----------
        x_pre, x_post : numpy array
            Donor pre- and post-period outcome matrices.
        t1 : int
            Number of post-periods (to scale the per-period ATT to incrementality).

        Returns
        -------
        A dict of p-value, per-period ``lift`` CIs, total ``incrementality`` CIs,
        the placebo band half-width, and the ``method``.
        """
        att = float(self.att) if self.att is not None else 0.0
        n_donors = x_pre.shape[1]
        placebos: list[float] = []
        for j in range(n_donors):
            mask = np.arange(n_donors) != j
            if not mask.any():
                continue
            placebos.append(self._sdid_att(x_pre[:, mask], x_post[:, mask], x_pre[:, j], x_post[:, j]))
        z = float(norm.ppf(1 - self.alpha / 2))
        placebo_arr = np.asarray(placebos, dtype=float)
        if placebo_arr.shape[0] >= 2:
            se = float(np.sqrt(np.mean((placebo_arr - placebo_arr.mean()) ** 2)))
        else:
            se = float("nan")
        if not np.isfinite(se) or se == 0.0:
            # Degenerate donor pool: no usable spread, so we cannot reject.
            lower, upper, p_value = att, att, 1.0
        else:
            lower, upper = att - z * se, att + z * se
            p_value = float(2.0 * (1.0 - norm.cdf(abs(att) / se)))
        band = 0.0 if not np.isfinite(se) else z * se
        return {
            "p_value": p_value,
            "standard_error": se,
            "lift_ci_lower": lower,
            "lift_ci_upper": upper,
            "incrementality_ci_lower": lower * t1,
            "incrementality_ci_upper": upper * t1,
            "conformal_band": band,
            "method": "placebo",
        }

    def _fit_predict_weights(self, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray | None:
        """Refit the SDID unit weights on a subset and predict (with the fixed effect).

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
        zeta = self.zeta if self.zeta is not None else self._default_zeta_for(x_train, x_eval.shape[0])
        intercept, weights = self._solve_weights(x_train, y_train, zeta)
        return x_eval @ weights + intercept

    def summarize(self, lift: str) -> None:
        """Print a tabulated summary of the synthetic difference-in-differences results.

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
        table_dict: dict[str, list[Any]] = {
            "Variant": [np.sum(self.results["test"])],
            "Baseline": [np.sum(self.results["counterfactual"])],
        }
        ci_alpha = self._get_ci_print()
        baseline = np.sum(self.results["counterfactual"])
        if lift in ["incremental", "absolute"]:
            table_dict["Metric"] = [self.y_variable]
            table_dict["Lift Type "] = ["Incremental"]
            table_dict["Lift"] = [f"""{ceil(self.results["incrementality"]):,}"""]
            table_dict[f"{ci_alpha} Lower CI"] = [f"""{ceil(self.results["incrementality_ci_lower"]):,}"""]
            table_dict[f"{ci_alpha} Upper CI"] = [f"""{ceil(self.results["incrementality_ci_upper"]):,}"""]
        elif lift == "relative":
            table_dict["Metric"] = [self.y_variable]
            table_dict["Lift Type"] = ["Relative"]
            table_dict["Lift"] = [f"""{round(float(self.results["incrementality"]) * 100 / baseline, 2)}%"""]
            table_dict[f"{ci_alpha} Lower CI"] = [
                f"""{round(self.results["incrementality_ci_lower"] * 100 / baseline, 2)}%"""
            ]
            table_dict[f"{ci_alpha} Upper CI"] = [
                f"""{round(self.results["incrementality_ci_upper"] * 100 / baseline, 2)}%"""
            ]
        elif lift == "revenue":
            table_dict["Metric"] = ["Revenue"]
            table_dict["Lift Type "] = ["Incremental"]
            table_dict["Lift"] = [f"""${round(self.results["incrementality"] * self.msrp, 2):,}"""]
            table_dict[f"{ci_alpha} Lower CI"] = [
                f"""${round(self.results["incrementality_ci_lower"] * self.msrp, 2):,}"""
            ]
            table_dict[f"{ci_alpha} Upper CI"] = [
                f"""${round(self.results["incrementality_ci_upper"] * self.msrp, 2):,}"""
            ]
        else:
            table_dict["Metric"] = ["ROAS"]
            table_dict["Lift Type "] = ["Incremental"]
            roas_lift, roas_ci_lower, roas_ci_upper = self._get_roas()
            table_dict["Lift"] = [f"${round(roas_lift, 2)}"]
            table_dict[f"{ci_alpha} Lower CI"] = [f"${round(roas_ci_lower, 2)}"]
            table_dict[f"{ci_alpha} Upper CI"] = [f"${round(roas_ci_upper, 2)}"]
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
        self._plot_counterfactual(
            self.dates,
            self.actual_pre,
            self.actual_post,
            self.prediction_pre,
            self.prediction_post,
        )
