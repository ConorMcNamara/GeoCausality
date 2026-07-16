"""Matrix Completion (MC-NNM) for geo-experiment causal inference."""

from typing import Any

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrame

from GeoCausality._base import EconometricEstimator

# Floor added to the previous iterate's Frobenius norm so the relative
# convergence check never divides by zero (e.g. on an all-zero panel).
_CONVERGENCE_NORM_EPS = 1e-12


class MatrixCompletion(EconometricEstimator):
    """Run matrix-completion (nuclear-norm) synthetic control for our geo-test.

    Unlike the linear synthetic-control family, which regresses the treated
    series on a weighted combination of donor geos, matrix completion stacks
    every unit (controls **and** the treated series) into a single panel matrix,
    marks the treated unit's post-period cells as missing, and completes the
    whole matrix under a low-rank (nuclear-norm) penalty. The imputed treated
    post-period cells are the counterfactual.

    The estimator standardizes each unit's series over its observed cells (which
    absorbs unit-level location and scale, so the summed treated row and the
    individual donor rows are comparable), then alternates a soft-impute step
    (fill the missing cells, SVD, soft-threshold the singular values) with a
    time fixed-effect update, following Athey, Bayati, Doudchenko, Imbens & Khosravi
    :cite:`athey2021matrix`. The nuclear-norm penalty ``lambda_`` is chosen by
    cross-validation over a warm-started decreasing path unless supplied.

    Inference reuses the shared conformal p-values and confidence intervals from
    ``EconometricEstimator`` (with the residual-only jackknife+ fallback for
    short pre-periods); there is no donor-weight vector, so the weight-based
    jackknife+ and parametric bootstrap are not wired in.
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
        lambda_: float | None = None,
        n_lambda: int = 50,
        lambda_ratio: float = 1e-3,
        max_iter: int = 200,
        tol: float = 1e-5,
        cv_fraction: float = 0.2,
        use_time_effect: bool = True,
        seed: int = 0,
        conformal_q: float = 1.0,
    ) -> None:
        """Initialize the matrix-completion estimator.

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
        lambda_ : float, optional
            The nuclear-norm penalty. When ``None`` (default) it is chosen by
            cross-validation over a decreasing path of ``n_lambda`` values.
        n_lambda : int, default=50
            Number of penalties on the cross-validation path (ignored when
            ``lambda_`` is given).
        lambda_ratio : float, default=1e-3
            The smallest penalty on the path, as a fraction of the largest
            (``lambda_max``). Ignored when ``lambda_`` is given.
        max_iter : int, default=200
            Maximum soft-impute iterations per solve.
        tol : float, default=1e-5
            Relative Frobenius-norm change below which soft-impute is converged.
        cv_fraction : float, default=0.2
            Fraction of observed control cells held out to score each penalty on
            the cross-validation path.
        use_time_effect : bool, default=True
            Whether to fit an additive time (column) fixed effect alongside the
            low-rank term. Unit (row) effects are always absorbed by the per-unit
            standardization.
        seed : int, default=0
            Seed for the cross-validation hold-out mask, so the fit is
            reproducible.
        conformal_q : float, default=1.0
            The exponent of the moving-block test statistic used for conformal
            inference (p-values and confidence intervals).

        Notes
        -----
        Based on Athey, Bayati, Doudchenko, Imbens & Khosravi :cite:`athey2021matrix`.
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
        if not 0.0 < cv_fraction < 1.0:
            raise ValueError(f"cv_fraction must be in (0, 1), got {cv_fraction}")
        if lambda_ is not None and lambda_ < 0.0:
            raise ValueError(f"lambda_ must be non-negative, got {lambda_}")
        self.dates: list[Any] | None = None
        self.prediction_pre: nw.DataFrame | None = None
        self.prediction_post: nw.DataFrame | None = None
        self.actual_pre: nw.DataFrame | None = None
        self.actual_post: nw.DataFrame | None = None
        self.lambda_ = lambda_
        self.n_lambda = n_lambda
        self.lambda_ratio = lambda_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.cv_fraction = cv_fraction
        self.use_time_effect = use_time_effect
        self.seed = seed
        self.conformal_q = conformal_q
        self.daily_x: nw.DataFrame | None = None
        self.daily_y: nw.DataFrame | None = None

    def pre_process(self) -> "MatrixCompletion":
        """Pivot the control geos into a daily outcome matrix and aggregate the test series.

        Returns
        -------
        MatrixCompletion
            Itself, so it can be chained with generate().
        """
        super().pre_process()
        self.dates = sorted(self.data[self.date_variable].unique().to_list())
        if self.treatment_variable is None:
            raise ValueError("treatment_variable must not be None")
        day_x = self.data.filter(nw.col(self.treatment_variable) == 0).select(
            [self.y_variable, self.geo_variable, self.date_variable]
        )
        # Pivot: rows=dates, cols=geos. Sort by date so the pre-period rows are the
        # leading block (the split in ``generate`` relies on this ordering). One
        # pivot over all dates keeps the donor column order consistent across the
        # pre/post split.
        self.daily_x = nw.from_native(
            day_x.to_native().pivot(on=self.geo_variable, index=self.date_variable, values=self.y_variable),
            eager_only=True,
        ).sort(self.date_variable)
        daily_y = (
            self.data.filter(nw.col(self.treatment_variable) == 1)
            .select([self.y_variable, self.date_variable])
            .group_by(self.date_variable)
            .agg(nw.col(self.y_variable).sum())
            .sort(self.date_variable)
        )
        self.daily_y = daily_y
        return self

    def generate(self) -> "MatrixCompletion":
        """Complete the masked panel and compute the counterfactual, lift and inference.

        Returns
        -------
        MatrixCompletion
            Itself, so it can be chained with summarize().
        """
        if self.daily_x is None or self.daily_y is None:
            raise ValueError("pre_process must be called before generate")
        # Split the sorted panel into the leading pre-period block and the
        # trailing post-period block. Compare dates as ISO strings so the split
        # is backend- and format-agnostic (matches the base class; integer years
        # like "1989" are not valid ``date.fromisoformat`` input).
        date_list = self.daily_x[self.date_variable].to_list()
        n_pre = sum(1 for d in date_list if str(d) <= self.pre_period)
        control_mat = self.daily_x.drop(self.date_variable).to_numpy().astype(float)  # (T, n_ctrl)
        treated = self.daily_y[self.y_variable].to_numpy().astype(float)  # (T,)

        # Panel oriented units x time: donors first, aggregated treated last. The
        # treated post-period is masked -- those are the cells we impute.
        panel = np.vstack([control_mat.T, treated[None, :]])  # (n_ctrl + 1, T)
        treated_idx = panel.shape[0] - 1
        mask = np.ones_like(panel, dtype=bool)
        mask[treated_idx, n_pre:] = False

        fitted = self._solve(panel, mask)
        prediction = fitted[treated_idx]

        pre_dates = date_list[:n_pre]
        post_dates = date_list[n_pre:]
        self.actual_pre = self._series_frame(pre_dates, treated[:n_pre])
        self.actual_post = self._series_frame(post_dates, treated[n_pre:])
        self.prediction_pre = self._series_frame(pre_dates, prediction[:n_pre])
        self.prediction_post = self._series_frame(post_dates, prediction[n_pre:])

        actual_post = treated[n_pre:]
        counterfactual_post = prediction[n_pre:]
        self.results = self._finalize_counterfactual_results(
            treated[:n_pre],
            prediction[:n_pre],
            actual_post,
            counterfactual_post,
            q=self.conformal_q,
        )
        return self

    def _solve(self, panel: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Complete the masked panel via MC-NNM and return the fitted matrix.

        Cross-validates (or uses) the nuclear-norm penalty and runs the
        two-way fixed-effect / soft-impute alternation on the raw scale.

        Parameters
        ----------
        panel : numpy array, shape (n_units, n_periods)
            The stacked outcome panel (donor rows then the treated row).
        mask : numpy array of bool, shape (n_units, n_periods)
            True where a cell is observed, False where it must be imputed.

        Returns
        -------
        The fitted full panel on the original scale.
        """
        lambda_ = self.lambda_ if self.lambda_ is not None else self._cv_lambda(panel, mask)
        return self._complete(panel, mask, lambda_)

    def _two_way_fe(self, resid: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fit additive row and column effects on the observed residual cells.

        Alternates row means and (optionally) column means over observed cells --
        the observed-data analogue of two-way demeaning -- which absorbs unit
        level differences (including the larger level of the aggregated treated
        row) and, when ``use_time_effect`` is set, a common time effect.

        Parameters
        ----------
        resid : numpy array
            The residual to demean (outcome minus the current low-rank term).
        mask : numpy array of bool
            True where observed.

        Returns
        -------
        A tuple of (row_effect, col_effect) as 1-D arrays.
        """
        row_obs = mask.sum(axis=1)
        col_obs = mask.sum(axis=0)
        gamma = np.zeros(resid.shape[0])
        delta = np.zeros(resid.shape[1])
        for _ in range(self.max_iter):
            gamma_new = np.where(
                row_obs > 0, ((resid - delta[None, :]) * mask).sum(axis=1) / np.maximum(row_obs, 1), 0.0
            )
            if self.use_time_effect:
                delta = np.where(
                    col_obs > 0, ((resid - gamma_new[:, None]) * mask).sum(axis=0) / np.maximum(col_obs, 1), 0.0
                )
            if np.max(np.abs(gamma_new - gamma)) < self.tol:
                gamma = gamma_new
                break
            gamma = gamma_new
        return gamma, delta

    def _complete(self, panel: np.ndarray, mask: np.ndarray, lambda_: float) -> np.ndarray:
        """Alternate two-way fixed effects and soft-impute at a fixed penalty.

        Parameters
        ----------
        panel : numpy array
            The raw outcome panel.
        mask : numpy array of bool
            True where observed.
        lambda_ : float
            The nuclear-norm penalty (singular-value soft-threshold).

        Returns
        -------
        The completed panel (low-rank term plus the two-way fixed effects).
        """
        obs = np.where(mask, panel, 0.0)
        low_rank = np.zeros_like(panel)
        prev = None
        gamma = np.zeros(panel.shape[0])
        delta = np.zeros(panel.shape[1])
        for _ in range(self.max_iter):
            gamma, delta = self._two_way_fe(panel - low_rank, mask)
            fe = gamma[:, None] + delta[None, :]
            low_rank = self._soft_impute(obs - np.where(mask, fe, 0.0), mask, lambda_, low_rank)
            fitted = low_rank + fe
            if prev is not None:
                denom = np.linalg.norm(prev) + _CONVERGENCE_NORM_EPS
                if np.linalg.norm(fitted - prev) / denom < self.tol:
                    break
            prev = fitted
        return low_rank + gamma[:, None] + delta[None, :]

    def _soft_impute(self, target: np.ndarray, mask: np.ndarray, lambda_: float, warm: np.ndarray) -> np.ndarray:
        """One soft-impute solve: fill missing cells, SVD, soft-threshold, iterate.

        Parameters
        ----------
        target : numpy array
            The matrix to complete (standardized panel minus the time effect).
        mask : numpy array of bool
            True where observed.
        lambda_ : float
            The singular-value soft-threshold.
        warm : numpy array
            Warm-start low-rank estimate (from the previous outer iteration or
            the previous penalty on the path).

        Returns
        -------
        The soft-thresholded low-rank completion.
        """
        low_rank = warm
        prev = None
        for _ in range(self.max_iter):
            filled = np.where(mask, target, low_rank)
            u, s, vt = np.linalg.svd(filled, full_matrices=False)
            s_thresh = np.maximum(s - lambda_, 0.0)
            low_rank = (u * s_thresh) @ vt
            if prev is not None:
                denom = np.linalg.norm(prev) + _CONVERGENCE_NORM_EPS
                if np.linalg.norm(low_rank - prev) / denom < self.tol:
                    break
            prev = low_rank
        return low_rank

    def _lambda_path(self, panel: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Geometric penalty path from ``lambda_max`` down to ``lambda_max * ratio``.

        ``lambda_max`` is the largest singular value of the two-way-demeaned
        observed matrix -- the smallest penalty that shrinks the low-rank term to
        zero -- so the path spans the full range from pure fixed effects to a
        near-interpolating completion.
        """
        gamma, delta = self._two_way_fe(panel, mask)
        demeaned = np.where(mask, panel - gamma[:, None] - delta[None, :], 0.0)
        lambda_max = float(np.linalg.svd(demeaned, compute_uv=False)[0])
        if lambda_max <= 0.0:
            return np.zeros(1)
        return np.geomspace(lambda_max, lambda_max * self.lambda_ratio, self.n_lambda)

    def _cv_lambda(self, panel: np.ndarray, mask: np.ndarray) -> float:
        """Pick the penalty minimizing held-out error on masked observed cells.

        Randomly holds out ``cv_fraction`` of the observed cells, fits the
        completion on the rest along a warm-started decreasing path, and returns
        the penalty with the lowest mean-squared error on the held-out cells.

        Parameters
        ----------
        panel : numpy array
            The raw outcome panel.
        mask : numpy array of bool
            True where observed.

        Returns
        -------
        The selected penalty.
        """
        rng = np.random.default_rng(self.seed)
        obs_idx = np.argwhere(mask)
        n_hold = max(1, int(round(self.cv_fraction * obs_idx.shape[0])))
        hold = obs_idx[rng.choice(obs_idx.shape[0], size=n_hold, replace=False)]
        train_mask = mask.copy()
        train_mask[hold[:, 0], hold[:, 1]] = False

        held_true = panel[hold[:, 0], hold[:, 1]]
        best_lambda, best_err = 0.0, np.inf
        for lambda_ in self._lambda_path(panel, train_mask):
            fitted = self._complete(panel, train_mask, lambda_)
            err = float(np.mean((fitted[hold[:, 0], hold[:, 1]] - held_true) ** 2))
            if err < best_err:
                best_err, best_lambda = err, lambda_
        return best_lambda
