import abc
from abc import ABC
from collections.abc import Callable
from datetime import date as date_cls
from typing import Any

import narwhals as nw
import numpy as np
import plotly.graph_objects as go
from narwhals.typing import IntoDataFrame
from plotly.subplots import make_subplots


class Estimator(abc.ABC):
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
        """Initialize the geo-causality estimator (abstract base class).

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
        treatment_variable : str, optional, default="is_treatment"
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
        """
        self.data: nw.DataFrame = nw.from_native(data, eager_only=True)
        self.test_geos = test_geos
        self.control_geos = control_geos
        self.pre_period = pre_period
        self.post_period = post_period
        self.geo_variable = geo_variable
        self.date_variable = date_variable
        self.y_variable = y_variable
        if (self.control_geos is None) | (self.test_geos is None):
            self.treatment_variable = treatment_variable
        else:
            self.treatment_variable = "is_test"
        self.alpha = alpha
        self.msrp = msrp
        self.spend = spend
        self.results: dict[str, Any] | None = None

    @abc.abstractmethod
    def pre_process(self) -> "Estimator":
        """Pre-process our data to make it usable for the estimator.

        Returns
        -------
        Itself, so it can be chained with generate()
        """
        pass

    @abc.abstractmethod
    def generate(self) -> "Estimator":
        """Run our models on pre-processed data to estimate causality.

        Returns
        -------
        Itself, so it can be chained with summarize()
        """
        pass

    @abc.abstractmethod
    def summarize(self, lift: str) -> None:
        """Summarize the results of generated models on our pre-processed data.

        Parameters
        ----------
        lift : str
            The kind of uplift we are measuring for geo-causality

        Returns
        -------
        The lift of our campaign
        """
        pass

    def _get_ci_print(self) -> str:
        percent = int((1 - self.alpha) * 100)
        return f"{percent}%"

    @abc.abstractmethod
    def _get_roas(self) -> tuple[float, float, float]:
        """Return our Return on Ad Spend (ROAS) and CIs.

        Returns
        -------
        A tuple containing our ROAS as well as Confidence Intervals
        """
        pass


class EconometricEstimator(Estimator, ABC):
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
        """Initialize the econometric estimator (abstract base class for FixedEffects and Diff-in-Diff).

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
        self.model: Any = None
        # Inference path: "auto" picks conformal for long pre-periods and falls
        # back to jackknife+ for short ones; "conformal" / "jackknife" force one;
        # "bootstrap" uses the parametric bootstrap (needs ``_bootstrap_refit``).
        self.inference_method: str = "auto"
        # Parametric-bootstrap settings (used when inference_method == "bootstrap").
        self.n_boot: int = 1000
        self.bootstrap_seed: int = 0
        # Donor matrices cached by synthetic-control estimators for faithful
        # jackknife+ (the shared leave-one-out loop in ``_block_loo``).
        self._jk_x_pre: np.ndarray | None = None
        self._jk_y_pre: np.ndarray | None = None
        self._jk_x_post: np.ndarray | None = None

    def pre_process(self) -> "EconometricEstimator":
        if self.test_geos is not None:
            if self.treatment_variable is not None:
                self.data: nw.DataFrame = self.data.with_columns(
                    nw.col(self.geo_variable).is_in(self.test_geos).cast(nw.Int64).alias(self.treatment_variable)
                )
        elif self.control_geos is not None:
            if self.treatment_variable is not None:
                self.data: nw.DataFrame = self.data.with_columns(
                    (1 - nw.col(self.geo_variable).is_in(self.control_geos).cast(nw.Int64)).alias(
                        self.treatment_variable
                    )
                )
        else:
            pass
        # Cast the date column to ISO strings so the comparison is backend
        # agnostic: polars `Date` columns cannot be compared to a string
        # literal directly, while pandas `datetime` columns cannot be compared
        # to a `date` object. ISO date strings sort lexicographically.
        date_as_str = nw.col(self.date_variable).cast(nw.String)
        self.data: nw.DataFrame = self.data.with_columns(
            nw.when(date_as_str <= self.pre_period)
            .then(0)
            .otherwise(nw.when(date_as_str >= self.post_period).then(1).otherwise(0))
            .alias("treatment_period")
        )

        return self

    # ------------------------------------------------------------------
    # Conformal inference
    #
    # Synthetic-control style estimators produce a fitted counterfactual but
    # no closed-form standard errors, so we attach distribution-free p-values
    # and confidence intervals via conformal inference. Two complementary
    # tools are provided, both operating purely on the pre-/post-period
    # residuals (actual - counterfactual):
    #   * The Chernozhukov, Wuthrich & Zhu (2021) moving-block permutation
    #     test, which gives an exact p-value for the sharp null of no effect
    #     and confidence intervals for a constant per-period effect via test
    #     inversion.
    #   * Split-conformal prediction bands, calibrated on the pre-period
    #     residuals, which give a pointwise band around the counterfactual.
    # ------------------------------------------------------------------
    @staticmethod
    def _conformal_residuals(
        actual_pre: Any,
        prediction_pre: Any,
        actual_post: Any,
        prediction_post: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute pre- and post-period residuals as flat numpy arrays.

        Parameters
        ----------
        actual_pre, prediction_pre : array-like
            Observed and counterfactual values over the pre-period.
        actual_post, prediction_post : array-like
            Observed and counterfactual values over the post-period.

        Returns
        -------
        A tuple of (pre_period_residuals, post_period_residuals).
        """
        pre = np.asarray(actual_pre, dtype=float).ravel() - np.asarray(prediction_pre, dtype=float).ravel()
        post = np.asarray(actual_post, dtype=float).ravel() - np.asarray(prediction_post, dtype=float).ravel()
        return pre, post

    @staticmethod
    def _block_statistics(residuals: np.ndarray, t1: int, q: float = 1.0) -> np.ndarray:
        """Compute the moving-block test statistic over every cyclic block.

        For each of the ``len(residuals)`` cyclically-contiguous blocks of
        length ``t1`` we compute ``S = mean(|u|^q) ** (1 / q)``.

        Parameters
        ----------
        residuals : numpy array
            The full (pre-period then post-period) residual series.
        t1 : int
            The length of the post-period block.
        q : float, default=1.0
            The exponent of the test statistic.

        Returns
        -------
        An array of block statistics, indexed by the block's start position.
        """
        n = residuals.shape[0]
        offsets = np.arange(t1)
        stats = np.empty(n)
        for start in range(n):
            block = residuals[(start + offsets) % n]
            stats[start] = np.mean(np.abs(block) ** q) ** (1.0 / q)
        return stats

    def _conformal_p_value(self, pre_resid: np.ndarray, post_resid: np.ndarray, q: float = 1.0) -> float:
        """Permutation p-value for the sharp null of no treatment effect.

        Parameters
        ----------
        pre_resid, post_resid : numpy array
            Pre- and post-period residuals.
        q : float, default=1.0
            The exponent of the moving-block test statistic.

        Returns
        -------
        The fraction of cyclic blocks whose statistic is at least as large as
        the observed post-period block.
        """
        full = np.concatenate([pre_resid, post_resid])
        t1 = post_resid.shape[0]
        stats = self._block_statistics(full, t1, q)
        observed = stats[full.shape[0] - t1]
        return float(np.mean(stats >= observed))

    @staticmethod
    def _bisect_threshold(
        p_func: Callable[[float], float],
        x_in: float,
        x_out: float,
        alpha: float,
        iters: int = 64,
    ) -> float:
        """Find the effect boundary where the test inverts at level ``alpha``.

        Assumes ``p_func(x_in) >= alpha`` (inside the acceptance region) and
        searches towards ``x_out`` for the furthest point still accepted.

        Parameters
        ----------
        p_func : callable
            Maps a candidate effect to its permutation p-value.
        x_in : float
            A point known to be inside the acceptance region.
        x_out : float
            A point assumed to be outside the acceptance region.
        alpha : float
            The significance level.
        iters : int, default=64
            Number of bisection iterations.

        Returns
        -------
        The boundary effect value.
        """
        if p_func(x_out) >= alpha:
            return x_out
        for _ in range(iters):
            mid = 0.5 * (x_in + x_out)
            if p_func(mid) >= alpha:
                x_in = mid
            else:
                x_out = mid
        return x_in

    def _conformal_interval(
        self,
        pre_resid: np.ndarray,
        post_resid: np.ndarray,
        q: float = 1.0,
        alpha: float | None = None,
    ) -> tuple[float, float]:
        """Confidence interval for a constant per-period effect by test inversion.

        Inverts the moving-block permutation test: the interval is the set of
        constant per-period effects ``theta`` for which the sharp null is not
        rejected at level ``alpha`` after subtracting ``theta`` from the
        post-period residuals.

        Parameters
        ----------
        pre_resid, post_resid : numpy array
            Pre- and post-period residuals.
        q : float, default=1.0
            The exponent of the moving-block test statistic.
        alpha : float, optional
            The significance level. Defaults to ``self.alpha``.

        Returns
        -------
        A tuple of (lower, upper) bounds on the per-period effect.
        """
        if alpha is None:
            alpha = self.alpha
        center = float(np.mean(post_resid))
        spread = float(np.std(np.concatenate([pre_resid, post_resid])))
        if spread == 0.0:
            spread = abs(center) if center != 0.0 else 1.0

        def p_func(theta: float) -> float:
            return self._conformal_p_value(pre_resid, post_resid - theta, q)

        upper = self._bisect_threshold(p_func, center, center + 10.0 * spread, alpha)
        lower = self._bisect_threshold(p_func, center, center - 10.0 * spread, alpha)
        return lower, upper

    def _split_conformal_band(
        self,
        pre_resid: np.ndarray,
        alpha: float | None = None,
    ) -> float:
        """Half-width of a split-conformal prediction band around the counterfactual.

        Uses the absolute pre-period residuals as the calibration set and
        returns the conformalized ``(1 - alpha)`` quantile, so a pointwise
        ``(1 - alpha)`` prediction interval is ``counterfactual +/- band``.

        Parameters
        ----------
        pre_resid : numpy array
            Pre-period residuals used for calibration.
        alpha : float, optional
            The significance level. Defaults to ``self.alpha``.

        Returns
        -------
        The band half-width.
        """
        if alpha is None:
            alpha = self.alpha
        scores = np.abs(pre_resid)
        n = scores.shape[0]
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return float(np.quantile(scores, level, method="higher"))

    def _conformal_inference(
        self,
        actual_pre: Any,
        prediction_pre: Any,
        actual_post: Any,
        prediction_post: Any,
        q: float = 1.0,
    ) -> dict[str, Any]:
        """Bundles conformal p-value, effect CIs and prediction band.

        Parameters
        ----------
        actual_pre, prediction_pre : array-like
            Observed and counterfactual values over the pre-period.
        actual_post, prediction_post : array-like
            Observed and counterfactual values over the post-period.
        q : float, default=1.0
            The exponent of the moving-block test statistic.

        Returns
        -------
        A dict suitable for ``self.results.update(...)`` containing the
        p-value, per-period ``lift`` CIs, total ``incrementality`` CIs, the
        prediction band half-width, and the ``method`` used for the intervals
        (``"conformal"``, ``"jackknife+"`` or ``"jackknife+ (residual)"``).

        Notes
        -----
        The moving-block permutation test and split-conformal band both need a
        reasonable number of pre-period points: the band quantile saturates and
        the permutation p-value loses resolution when the pre-period is short.
        When that happens (see ``_pre_period_too_short``) the interval falls back
        to jackknife+ (Barber et al., 2021), which reuses every pre-period point
        for both fitting and calibration. An estimator that overrides
        ``_loo_counterfactuals`` gets the faithful refit-based jackknife+;
        otherwise a residual-only approximation is used.
        """
        pre_resid, post_resid = self._conformal_residuals(actual_pre, prediction_pre, actual_post, prediction_post)
        t1 = post_resid.shape[0]
        p_value = self._conformal_p_value(pre_resid, post_resid, q)

        boot = None
        if self.inference_method == "bootstrap":
            fitted_pre = np.asarray(prediction_pre, dtype=float).ravel()
            fitted_post = np.asarray(prediction_post, dtype=float).ravel()
            actual_post_arr = np.asarray(actual_post, dtype=float).ravel()
            boot = self._parametric_bootstrap_interval(fitted_pre, fitted_post, actual_post_arr, pre_resid)
        if boot is not None:
            method = "bootstrap"
            lift_lower, lift_upper, band, p_value = boot
        elif self._pre_period_too_short(pre_resid.shape[0]):
            loo = self._loo_counterfactuals()
            if loo is not None:
                method = "jackknife+"
                lift_lower, lift_upper, band = self._jackknife_plus_interval(
                    loo[0], loo[1], np.asarray(actual_post, dtype=float).ravel(), t1
                )
            else:
                method = "jackknife+ (residual)"
                lift_lower, lift_upper, band = self._jackknife_residual_interval(pre_resid, post_resid)
        else:
            method = "conformal"
            lift_lower, lift_upper = self._conformal_interval(pre_resid, post_resid, q)
            band = self._split_conformal_band(pre_resid)

        return {
            "p_value": p_value,
            "lift_ci_lower": lift_lower,
            "lift_ci_upper": lift_upper,
            "incrementality_ci_lower": lift_lower * t1,
            "incrementality_ci_upper": lift_upper * t1,
            "conformal_band": band,
            "method": method,
        }

    # ------------------------------------------------------------------
    # Jackknife+ fallback
    #
    # When the pre-period is too short for the conformal tools above, we fall
    # back to jackknife+ (Barber, Candes, Ramdas & Tibshirani, 2021), which
    # reuses every pre-period point for both fitting and calibration via
    # leave-one-out and carries a distribution-free >= 1 - 2*alpha coverage
    # guarantee. The faithful version needs per-fold refits, supplied by an
    # estimator overriding ``_loo_counterfactuals``; without it we approximate
    # using the pre-period residuals with the counterfactual held fixed.
    # ------------------------------------------------------------------
    def _pre_period_too_short(self, n_pre: int, alpha: float | None = None) -> bool:
        """Whether the pre-period is too short for the conformal interval.

        Honours an explicit ``inference_method`` override; otherwise the
        pre-period is "too short" exactly when the split-conformal quantile level
        ``ceil((n + 1)(1 - alpha)) / n`` saturates at 1.0, the documented failure
        mode where the band collapses to the largest residual.

        Parameters
        ----------
        n_pre : int
            The number of pre-period points.
        alpha : float, optional
            The significance level. Defaults to ``self.alpha``.

        Returns
        -------
        True if the jackknife+ fallback should be used.
        """
        if self.inference_method == "jackknife":
            return True
        if self.inference_method == "conformal":
            return False
        if alpha is None:
            alpha = self.alpha
        return bool(np.ceil((n_pre + 1) * (1 - alpha)) > n_pre)

    def _loo_counterfactuals(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Leave-one-out pre-period residuals and post-period counterfactuals.

        Returns ``(loo_residuals, loo_post_predictions)`` where ``loo_residuals``
        has shape ``(n_folds,)`` (the absolute held-out pre-period residual of
        each fold) and ``loo_post_predictions`` has shape ``(n_folds, t1)`` (each
        leave-one-out model's counterfactual over the post-period). Returns
        ``None`` -- triggering the residual-only jackknife+ approximation -- when
        the estimator has not cached the donor matrices (``_jk_x_pre`` etc.) and
        implemented ``_fit_predict_weights``.

        Synthetic-control estimators all predict ``donor_matrix @ weights`` (with
        an intercept for the augmented method), so the leave-one-out loop is
        shared in ``_block_loo`` and each estimator only supplies its weight fit.

        Returns
        -------
        The leave-one-out arrays, or None if unsupported.
        """
        x_pre = self._jk_x_pre
        if x_pre is None or self._jk_y_pre is None or self._jk_x_post is None:
            return None
        return self._block_loo(x_pre, self._jk_y_pre, self._jk_x_post)

    def _fit_predict_weights(self, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray | None:
        """Fit this estimator's donor weights on a pre-period subset and predict.

        Default returns ``None`` (no faithful jackknife+). A synthetic-control
        estimator overrides this to fit its weights on ``(x_train, y_train)`` --
        a subset of the pre-period rows -- and return the counterfactual for the
        rows of ``x_eval`` (a held-out pre-period row stacked on the post-period
        donor matrix). Donor-column order matches the cached ``_jk_x_*`` matrices.

        Parameters
        ----------
        x_train : numpy array, shape (n_train, n_donors)
            Pre-period donor rows used to refit the weights.
        y_train : numpy array, shape (n_train,)
            Treated pre-period series on the same rows.
        x_eval : numpy array, shape (n_eval, n_donors)
            Donor rows to predict the counterfactual for.

        Returns
        -------
        The counterfactual for each ``x_eval`` row, or None if unsupported.
        """
        return None

    def _block_loo(
        self,
        x_pre: np.ndarray,
        y_pre: np.ndarray,
        x_post: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Run the leave-one-out refit loop over the pre-period via ``_fit_predict_weights``.

        For each pre-period row, refit the weights on the remaining rows and
        predict both the held-out row (its residual) and the post-period
        counterfactual. Folds the fit declines (returns None) are skipped. The
        synthetic-control weight fits are regularised or simplex-constrained, so
        they remain well posed when donors outnumber the training rows.

        Parameters
        ----------
        x_pre : numpy array, shape (n_pre, n_donors)
            Pre-period donor matrix.
        y_pre : numpy array, shape (n_pre,)
            Treated pre-period series.
        x_post : numpy array, shape (t1, n_donors)
            Post-period donor matrix.

        Returns
        -------
        ``(loo_residuals, loo_post_predictions)``, or None if no fold succeeded.
        """
        n = x_pre.shape[0]
        loo_resid: list[float] = []
        loo_post: list[np.ndarray] = []
        for i in range(n):
            mask = np.arange(n) != i
            eval_mat = np.vstack([x_pre[i][None, :], x_post])
            pred = self._fit_predict_weights(x_pre[mask], y_pre[mask], eval_mat)
            if pred is None:
                continue
            loo_resid.append(abs(float(y_pre[i] - pred[0])))
            loo_post.append(np.asarray(pred[1:], dtype=float))
        if not loo_resid:
            return None
        return np.array(loo_resid), np.array(loo_post)

    @staticmethod
    def _jackknife_quantile(abs_residuals: np.ndarray, alpha: float) -> float:
        """Return the jackknife+ ``(1 - alpha)`` quantile of absolute residuals.

        Uses the ``ceil((1 - alpha)(n + 1))``-th smallest absolute residual,
        widening to the maximum when that rank exceeds ``n``.

        Parameters
        ----------
        abs_residuals : numpy array
            Absolute residuals.
        alpha : float
            The significance level.

        Returns
        -------
        The band half-width.
        """
        scores = np.sort(np.abs(abs_residuals))
        n = scores.shape[0]
        k = int(np.ceil((1 - alpha) * (n + 1)))
        if k > n:
            return float(scores[-1])
        return float(scores[k - 1])

    def _jackknife_residual_interval(
        self,
        pre_resid: np.ndarray,
        post_resid: np.ndarray,
        alpha: float | None = None,
    ) -> tuple[float, float, float]:
        """Residual-only jackknife+ interval for a constant per-period effect.

        Holds the counterfactual fixed and calibrates a band from the pre-period
        residuals (every point used for both fit and calibration), centred on the
        mean post-period residual.

        Parameters
        ----------
        pre_resid, post_resid : numpy array
            Pre- and post-period residuals.
        alpha : float, optional
            The significance level. Defaults to ``self.alpha``.

        Returns
        -------
        A tuple of (lift_lower, lift_upper, band_half_width).
        """
        if alpha is None:
            alpha = self.alpha
        band = self._jackknife_quantile(pre_resid, alpha)
        center = float(np.mean(post_resid))
        return center - band, center + band, band

    def _jackknife_plus_interval(
        self,
        loo_resid: np.ndarray,
        loo_post_pred: np.ndarray,
        actual_post: np.ndarray,
        t1: int,
        alpha: float | None = None,
    ) -> tuple[float, float, float]:
        """Faithful jackknife+ interval from leave-one-out refits.

        For each post-period point the counterfactual bounds combine every
        leave-one-out model's prediction at that point with that model's held-out
        residual; the per-period lift bounds are aggregated into a per-period
        effect CI (so ``lift_ci * t1`` recovers the incrementality CI).

        Parameters
        ----------
        loo_resid : numpy array, shape (n_folds,)
            Absolute held-out pre-period residual of each fold.
        loo_post_pred : numpy array, shape (n_folds, t1)
            Each leave-one-out model's post-period counterfactual.
        actual_post : numpy array, shape (t1,)
            Observed post-period outcome.
        t1 : int
            The number of post-period points.
        alpha : float, optional
            The significance level. Defaults to ``self.alpha``.

        Returns
        -------
        A tuple of (lift_lower, lift_upper, band_half_width).
        """
        if alpha is None:
            alpha = self.alpha
        n = loo_resid.shape[0]
        lo_idx = max(int(np.floor(alpha * (n + 1))), 1) - 1
        hi_idx = min(int(np.ceil((1 - alpha) * (n + 1))), n) - 1
        incr_lower, incr_upper = 0.0, 0.0
        for t in range(t1):
            cf_lower = np.sort(loo_post_pred[:, t] - loo_resid)[lo_idx]
            cf_upper = np.sort(loo_post_pred[:, t] + loo_resid)[hi_idx]
            incr_lower += actual_post[t] - cf_upper
            incr_upper += actual_post[t] - cf_lower
        band = self._jackknife_quantile(loo_resid, alpha)
        return incr_lower / t1, incr_upper / t1, band

    # ------------------------------------------------------------------
    # Parametric bootstrap
    #
    # Mirrors GeoLift's Generalized Synthetic Control inference: hold the
    # estimated counterfactual fixed, draw parametric noise at the pre-period
    # residual scale, rebuild pseudo treated pre-period series, refit the
    # counterfactual (via ``_bootstrap_refit``) and accumulate an incrementality
    # distribution. The CI is the percentile interval; the p-value is the
    # two-sided proportion of no-effect replicates at least as extreme as the
    # observed incrementality. Needs a refit hook, so estimators without one fall
    # back to the conformal / jackknife+ path.
    # ------------------------------------------------------------------
    def _bootstrap_refit(self, treated_pre: np.ndarray) -> np.ndarray | None:
        """Refit the counterfactual over the post-period from a resampled pre-period.

        Delegates to ``_fit_predict_weights``: refit the weights on the cached
        pre-period donor matrix against the resampled treated series and predict
        the post-period donor matrix. Synthetic-control estimators get the
        parametric bootstrap for free this way; an estimator that caches a
        different design (e.g. the factor design) overrides this directly. Returns
        ``None`` -- so the bootstrap falls back -- when the donor matrices are not
        cached or the weight fit is unsupported.

        Parameters
        ----------
        treated_pre : numpy array, shape (n_pre,)
            A resampled treated pre-period series.

        Returns
        -------
        The refit post-period counterfactual, or None if unsupported.
        """
        if self._jk_x_pre is None or self._jk_x_post is None:
            return None
        return self._fit_predict_weights(self._jk_x_pre, treated_pre, self._jk_x_post)

    def _parametric_bootstrap_interval(
        self,
        fitted_pre: np.ndarray,
        fitted_post: np.ndarray,
        actual_post: np.ndarray,
        pre_resid: np.ndarray,
        alpha: float | None = None,
    ) -> tuple[float, float, float, float] | None:
        """Parametric-bootstrap interval and p-value for the incrementality.

        Parameters
        ----------
        fitted_pre, fitted_post : numpy array
            The fitted counterfactual over the pre- and post-period.
        actual_post : numpy array
            Observed post-period outcome.
        pre_resid : numpy array
            Pre-period residuals, whose scale sets the parametric noise.
        alpha : float, optional
            The significance level. Defaults to ``self.alpha``.

        Returns
        -------
        A tuple of (lift_lower, lift_upper, band_half_width, p_value), or None if
        the estimator does not implement ``_bootstrap_refit``.
        """
        if alpha is None:
            alpha = self.alpha
        if self._bootstrap_refit(fitted_pre) is None:
            return None

        sigma = float(np.std(pre_resid))
        if sigma == 0.0:
            sigma = 1.0
        n_pre, t1 = fitted_pre.shape[0], actual_post.shape[0]
        rng = np.random.default_rng(self.bootstrap_seed)
        incr_obs = float(np.sum(actual_post - fitted_post))

        incr_samples = np.empty(self.n_boot)
        null_samples = np.empty(self.n_boot)
        for b in range(self.n_boot):
            cf_post = self._bootstrap_refit(fitted_pre + rng.normal(0.0, sigma, n_pre))
            eps_post = rng.normal(0.0, sigma, t1)
            # Replicate of the observed incrementality (counterfactual + post noise).
            incr_samples[b] = float(np.sum(actual_post - cf_post - eps_post))
            # Replicate under the sharp null of no effect (observed == counterfactual).
            null_samples[b] = float(np.sum(fitted_post + eps_post - cf_post))

        # Center the bootstrap spread on the reported incrementality. A refit on
        # the fitted counterfactual need not reproduce it exactly for penalised or
        # mean-based estimators, which would otherwise bias the percentile
        # interval off the point estimate; centering on incr_obs de-biases it and
        # guarantees the interval brackets the reported lift.
        boot_mean = float(np.mean(incr_samples))
        lower = incr_obs + float(np.percentile(incr_samples, 100 * alpha / 2)) - boot_mean
        upper = incr_obs + float(np.percentile(incr_samples, 100 * (1 - alpha / 2))) - boot_mean
        band = self._jackknife_quantile(pre_resid, alpha)
        p_value = float(np.mean(np.abs(null_samples) >= abs(incr_obs)))
        return lower / t1, upper / t1, band, p_value

    # ------------------------------------------------------------------
    # Shared counterfactual plot
    #
    # Every synthetic-control style estimator draws the same three panels
    # (actual vs counterfactual, pointwise difference, cumulative difference)
    # and carries the same inference outputs, so the figure -- including the
    # confidence bands -- is built once here and each estimator's ``plot()``
    # just supplies its series as arrays.
    # ------------------------------------------------------------------
    def _plot_counterfactual(
        self,
        dates: list[Any],
        actual_pre: np.ndarray,
        actual_post: np.ndarray,
        prediction_pre: np.ndarray,
        prediction_post: np.ndarray,
    ) -> None:
        """Build and show the three-panel counterfactual figure with confidence bands.

        The top panel shows the actual and counterfactual series with the pointwise
        prediction band (``results["conformal_band"]``) shaded around the
        counterfactual; the middle panel shows the pointwise difference with the
        same band shaded around zero (points outside it are the significant
        per-period effects); the bottom panel shows the cumulative post-period
        difference with a band that grows linearly to the reported incrementality
        interval, so its endpoint matches ``summarize()``.

        Parameters
        ----------
        dates : list
            The full (pre- then post-period) date axis.
        actual_pre, actual_post : numpy array
            Observed outcome over the pre- and post-period.
        prediction_pre, prediction_post : numpy array
            Counterfactual outcome over the pre- and post-period.
        """
        if self.results is None:
            raise ValueError("results must not be None")
        actual_pre = np.asarray(actual_pre, dtype=float).ravel()
        actual_post = np.asarray(actual_post, dtype=float).ravel()
        prediction_pre = np.asarray(prediction_pre, dtype=float).ravel()
        prediction_post = np.asarray(prediction_post, dtype=float).ravel()
        actual = np.concatenate([actual_pre, actual_post])
        prediction = np.concatenate([prediction_pre, prediction_post])
        residuals = actual - prediction
        t1 = actual_post.shape[0]

        band_value = self.results.get("conformal_band")
        band = float(band_value) if band_value is not None else 0.0
        has_band = bool(np.isfinite(band)) and band > 0
        band_fill = "rgba(68, 68, 68, 0.2)"
        ci_label = f"{self._get_ci_print()} CI"

        total_fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(
                "Expected vs Counterfactual",
                "Pointwise Difference",
                "Cumulative Difference",
            ),
        )

        # Top panel: band around the counterfactual, then the two series on top.
        if has_band:
            total_fig.add_trace(
                go.Scatter(x=dates, y=prediction + band, mode="lines", line={"width": 0}, showlegend=False),
                row=1,
                col=1,
            )
            total_fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=prediction - band,
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor=band_fill,
                    name=ci_label,
                ),
                row=1,
                col=1,
            )
        total_fig.add_trace(
            go.Scatter(x=dates, y=actual, marker={"color": "blue"}, mode="lines", name="Actual"),
            row=1,
            col=1,
        )
        total_fig.add_trace(
            go.Scatter(x=dates, y=prediction, marker={"color": "red"}, mode="lines", name="Counterfactual"),
            row=1,
            col=1,
        )

        # Middle panel: the pointwise band around zero (the no-effect region).
        if has_band:
            total_fig.add_trace(
                go.Scatter(x=dates, y=np.full(len(dates), band), mode="lines", line={"width": 0}, showlegend=False),
                row=2,
                col=1,
            )
            total_fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=np.full(len(dates), -band),
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor=band_fill,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        total_fig.add_trace(
            go.Scatter(x=dates, y=residuals, marker={"color": "purple"}, mode="lines", name="Difference"),
            row=2,
            col=1,
        )

        # Bottom panel: cumulative post-period difference with a band that grows
        # linearly to the reported incrementality interval.
        post_period_date = date_cls.fromisoformat(self.post_period)
        marketing_start = [d for d in dates if d >= post_period_date]
        cum_resids = np.cumsum(actual_post - prediction_post)
        incr = self.results.get("incrementality")
        incr_lo = self.results.get("incrementality_ci_lower")
        incr_hi = self.results.get("incrementality_ci_upper")
        if incr is not None and incr_lo is not None and incr_hi is not None and t1 > 0:
            frac = np.arange(1, t1 + 1, dtype=float) / t1
            cum_upper = cum_resids + frac * (float(incr_hi) - float(incr))
            cum_lower = cum_resids + frac * (float(incr_lo) - float(incr))
            total_fig.add_trace(
                go.Scatter(x=marketing_start, y=cum_upper, mode="lines", line={"width": 0}, showlegend=False),
                row=3,
                col=1,
            )
            total_fig.add_trace(
                go.Scatter(
                    x=marketing_start,
                    y=cum_lower,
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor="rgba(255, 165, 0, 0.2)",
                    name=ci_label,
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
        total_fig.add_trace(
            go.Scatter(
                x=marketing_start,
                y=cum_resids,
                marker={"color": "orange"},
                mode="lines",
                name="Cumulative Incrementality",
            ),
            row=3,
            col=1,
        )

        for row in (1, 2, 3):
            total_fig.add_vline(
                x=self.post_period,
                line_width=1,
                line_dash="dash",
                line_color="black",
                row=row,
                col=1,
            )
        total_fig.show()


class MLEstimator(Estimator, abc.ABC):
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
        """Initialize the ML estimator (abstract base class for GeoX).

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

        self.pre_control: nw.DataFrame | None = None
        self.post_control: nw.DataFrame | None = None
        self.pre_test: nw.DataFrame | None = None
        self.post_test: nw.DataFrame | None = None
        self.model: Any = None
        self.test_dates: nw.Series | None = None

    def pre_process(self) -> "MLEstimator":
        if self.treatment_variable is not None:
            # Cast the date column to ISO strings so the comparison is backend
            # agnostic: polars `Date` columns cannot be compared to a string
            # literal directly, while pandas `datetime` columns cannot be
            # compared to a `date` object. ISO date strings sort
            # lexicographically.
            date_as_str = nw.col(self.date_variable).cast(nw.String)
            pre_control = self.data.filter((date_as_str <= self.pre_period) & (nw.col(self.treatment_variable) == 0))
            self.pre_control = (
                pre_control.group_by(self.date_variable).agg(nw.col(self.y_variable).sum()).sort(self.date_variable)
            )
            self.pre_control = self.pre_control.drop(self.date_variable)

            pre_test = self.data.filter((date_as_str <= self.pre_period) & (nw.col(self.treatment_variable) == 1))
            self.pre_test = (
                pre_test.group_by(self.date_variable).agg(nw.col(self.y_variable).sum()).sort(self.date_variable)
            )
            self.pre_test = self.pre_test.drop(self.date_variable)

            post_control = self.data.filter((date_as_str >= self.post_period) & (nw.col(self.treatment_variable) == 0))
            self.post_control = (
                post_control.group_by(self.date_variable).agg(nw.col(self.y_variable).sum()).sort(self.date_variable)
            )
            self.test_dates = self.post_control[self.date_variable]
            self.post_control = self.post_control.drop(self.date_variable)

            post_test = self.data.filter((date_as_str >= self.post_period) & (nw.col(self.treatment_variable) == 1))
            self.post_test = (
                post_test.group_by(self.date_variable).agg(nw.col(self.y_variable).sum()).sort(self.date_variable)
            )
            self.post_test = self.post_test.drop(self.date_variable)

        return self
