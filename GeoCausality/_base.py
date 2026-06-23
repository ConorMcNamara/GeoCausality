import abc
from abc import ABC
from collections.abc import Callable
from typing import Any

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrame


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
    ) -> dict[str, float]:
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
        p-value, per-period ``lift`` CIs, total ``incrementality`` CIs and the
        split-conformal prediction band half-width.
        """
        pre_resid, post_resid = self._conformal_residuals(actual_pre, prediction_pre, actual_post, prediction_post)
        t1 = post_resid.shape[0]
        lift_lower, lift_upper = self._conformal_interval(pre_resid, post_resid, q)
        return {
            "p_value": self._conformal_p_value(pre_resid, post_resid, q),
            "lift_ci_lower": lift_lower,
            "lift_ci_upper": lift_upper,
            "incrementality_ci_lower": lift_lower * t1,
            "incrementality_ci_upper": lift_upper * t1,
            "conformal_band": self._split_conformal_band(pre_resid),
        }


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
        """Initialize the ML estimator (abstract base class for GeoX and Synthetic Control).

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
