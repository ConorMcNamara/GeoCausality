"""Kernel-ridge (nonlinear-map) Synthetic Control for geo-experiment causal inference."""

from collections.abc import Callable

import narwhals as nw
import numpy as np
import polars as pl
from narwhals.typing import IntoDataFrame

from GeoCausality._base import NonLinearEstimator


class KernelSyntheticControl(NonLinearEstimator):
    """Run kernel-ridge (nonlinear-map) synthetic control for our geo-test.

    Where the linear synthetic-control family (and Tian's
    ``NonlinearSyntheticControl``) build the counterfactual as a *combination of
    donor outcomes*, this learns a nonlinear regression of the treated series on
    the donor outcomes via kernel ridge with a composite **linear + RBF** kernel:

    * each donor column is standardised so the kernel is not dominated by
      high-variance donors;
    * a **linear** kernel term gives the model a global linear backbone, so it
      extrapolates trends (a treated unit whose donors drift outside their
      pre-period range does not collapse to the pre-period mean);
    * an **RBF (Gaussian)** kernel term adds local nonlinear flexibility on top of
      that backbone;
    * the treated series is centred, so its pre-period level is carried by an
      intercept rather than shrunk toward zero;
    * the fitted map is ``f(x) = y_bar + K(x, X_pre) @ (K + lambda I)^{-1} y_c``.

    The bandwidth defaults to the median pairwise distance in standardised donor
    space (the median heuristic) and the ridge penalty ``lambda_`` defaults to the
    one-standard-error rule over a leave-one-out grid; both can be pinned. This is
    the estimator to reach for when the treated unit relates to the donors
    *nonlinearly* within a reasonably stationary regime; for strongly trending
    panels prefer ``NonlinearSyntheticControl`` (Tian 2023) or the linear family.

    Notes
    -----
    Kernel ridge / RKHS synthetic control. A pure linear kernel
    (``linear_weight -> inf`` relative to the RBF, or a large bandwidth) recovers
    a ridge-regularised linear synthetic control. Inference (conformal p-values
    and intervals, with the jackknife+ fallback for short pre-periods) is
    inherited unchanged from :class:`NonLinearEstimator`.
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
        bandwidth: float | None = None,
        lambda_: float | None = None,
        linear_weight: float = 1.0,
        conformal_q: float = 1.0,
    ) -> None:
        """Initialize the kernel-ridge synthetic control estimator.

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
        bandwidth : float, optional
            The RBF kernel length-scale (in standardised donor space). If not
            given, it is set to the median pairwise distance of the pre-period
            donor rows (the median heuristic).
        lambda_ : float, optional
            The ridge penalty. If not given, it is chosen by the one-standard-error
            rule over a leave-one-out cross-validation grid.
        linear_weight : float, default=1.0
            The weight of the (normalised) linear kernel term relative to the RBF
            term. ``0.0`` gives a pure RBF kernel (no linear extrapolation
            backbone); larger values lean more on the linear trend.
        conformal_q : float, default=1.0
            The exponent of the moving-block test statistic used for conformal
            inference (p-values and confidence intervals).
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
            conformal_q,
        )
        self.bandwidth = bandwidth
        self.lambda_ = lambda_
        self.linear_weight = linear_weight
        # Resolved (fitted) hyper-parameters. Set on the full pre-period fit and
        # reused across jackknife+/bootstrap refits so every fold shares one model.
        self._fit_bandwidth: float | None = None
        self._fit_lambda: float | None = None

    def pre_process(self) -> "KernelSyntheticControl":
        """Tag the pre/post periods and cache the full date axis for plotting.

        Returns
        -------
        KernelSyntheticControl
            Itself, so it can be chained with generate().
        """
        super().pre_process()
        self.dates = sorted(self.data[self.date_variable].unique().to_list())
        return self

    def generate(self) -> "KernelSyntheticControl":
        """Fit the kernel-ridge counterfactual and compute lift and inference.

        Returns
        -------
        KernelSyntheticControl
            Itself, so it can be chained with summarize().
        """
        if self.treatment_variable is None:
            raise ValueError("treatment_variable must not be None")
        # Re-resolve hyper-parameters on each generate() so a re-run does not reuse
        # a stale bandwidth/lambda from a previous fit.
        self._fit_bandwidth = None
        self._fit_lambda = None

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
        # Pivot: rows=dates, cols=geos (same donor-column order for pre and post).
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
        # Cache the donor matrices for the shared faithful jackknife+ / bootstrap loops.
        self._jk_x_pre = control_pre_mat
        self._jk_x_post = control_post_mat
        self._jk_y_pre = self.actual_pre[self.y_variable].to_numpy()

        # Fit once on the full pre-period, then predict both windows.
        predictor = self._fit_model(control_pre_mat, self._jk_y_pre)
        self.model = predictor
        prediction_pre_arr = predictor(control_pre_mat)
        prediction_post_arr = predictor(control_post_mat)

        self.prediction_pre = nw.from_native(
            pl.DataFrame(
                {
                    self.date_variable: control_pre_pivot[self.date_variable].to_native(),
                    self.y_variable: np.asarray(prediction_pre_arr).flatten(),
                }
            ),
            eager_only=True,
        )
        self.prediction_post = nw.from_native(
            pl.DataFrame(
                {
                    self.date_variable: control_post_pivot[self.date_variable].to_native(),
                    self.y_variable: np.asarray(prediction_post_arr).flatten(),
                }
            ),
            eager_only=True,
        )
        self.results = {
            "test": self.actual_post[self.y_variable].to_numpy(),
            "counterfactual": self.prediction_post[self.y_variable].to_numpy(),
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

    def _fit_model(self, x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        """Fit kernel ridge on ``(x, y)`` and return a predictor closure.

        Standardises the donor columns, centres the target, resolves the bandwidth
        and ridge penalty (once, on the first / full-pre-period fit, then reused),
        and solves ``(K + lambda I) coef = y_centred`` for the composite linear+RBF
        kernel. The returned closure carries the training rows, so the
        counterfactual for a new donor matrix is ``y_bar + K(x_new, x_train) @ coef``.

        Parameters
        ----------
        x : numpy array, shape (n_periods, n_donors)
            Donor outcome matrix.
        y : numpy array, shape (n_periods,)
            Treated (aggregated) outcome on the same rows.

        Returns
        -------
        A callable mapping a donor matrix to its counterfactual.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std == 0.0] = 1.0
        x_scaled = (x - mean) / std

        if self._fit_bandwidth is None:
            self._fit_bandwidth = (
                float(self.bandwidth) if self.bandwidth is not None else self._median_bandwidth(x_scaled)
            )
        bandwidth = self._fit_bandwidth
        if self._fit_lambda is None:
            self._fit_lambda = (
                float(self.lambda_) if self.lambda_ is not None else self._select_lambda(x_scaled, y, bandwidth)
            )
        lambda_ = self._fit_lambda

        y_bar = float(y.mean())
        y_centered = y - y_bar
        kernel = self._kernel_matrix(x_scaled, x_scaled, bandwidth)
        coef = np.linalg.solve(kernel + lambda_ * np.identity(kernel.shape[0]), y_centered)

        def predict(x_new: np.ndarray) -> np.ndarray:
            x_new = np.asarray(x_new, dtype=float)
            x_new_scaled = (x_new - mean) / std
            return y_bar + self._kernel_matrix(x_new_scaled, x_scaled, bandwidth) @ coef

        return predict

    @staticmethod
    def _squared_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Pairwise squared Euclidean distances between the rows of ``a`` and ``b``."""
        a_sq = np.sum(a**2, axis=1)[:, None]
        b_sq = np.sum(b**2, axis=1)[None, :]
        return np.clip(a_sq + b_sq - 2.0 * a @ b.T, 0.0, None)

    def _kernel_matrix(self, a: np.ndarray, b: np.ndarray, bandwidth: float) -> np.ndarray:
        """Composite linear + RBF kernel between the rows of ``a`` and ``b``.

        The RBF term provides local nonlinear flexibility; the (donor-count
        normalised) linear term provides a global backbone so the counterfactual
        extrapolates trends instead of reverting to the pre-period mean.
        """
        kernel = np.exp(-self._squared_dists(a, b) / (2.0 * bandwidth**2))
        if self.linear_weight > 0.0:
            n_features = max(a.shape[1], 1)
            kernel = kernel + self.linear_weight * (a @ b.T) / n_features
        return kernel

    def _median_bandwidth(self, x_scaled: np.ndarray) -> float:
        """Median pairwise distance of the (standardised) donor rows (median heuristic)."""
        n = x_scaled.shape[0]
        if n < 2:
            return 1.0
        upper = np.triu_indices(n, k=1)
        dists = np.sqrt(self._squared_dists(x_scaled, x_scaled)[upper])
        positive = dists[dists > 0.0]
        if positive.size == 0:
            return 1.0
        return float(np.median(positive))

    @staticmethod
    def _generate_lambdas(eigvals: np.ndarray, n_lambda: int = 20, lambda_min_ratio: float = 1e-06) -> np.ndarray:
        """Log-spaced (ascending) ridge-penalty grid scaled to the kernel spectrum.

        Parameters
        ----------
        eigvals : numpy array
            The eigenvalues of the pre-period kernel matrix.
        n_lambda : int, default=20
            The number of candidate penalties to return.
        lambda_min_ratio : float, default=1e-06
            The smallest penalty as a fraction of the largest eigenvalue.

        Returns
        -------
        An ascending array of candidate ridge penalties.
        """
        lambda_max = float(np.max(eigvals)) if eigvals.size else 1.0
        if lambda_max <= 0.0:
            lambda_max = 1.0
        return np.geomspace(lambda_max * lambda_min_ratio, lambda_max, n_lambda)

    def _select_lambda(self, x_scaled: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
        """Select the ridge penalty by the one-standard-error rule.

        Uses the closed-form leave-one-out residual for a kernel smoother --
        ``e_i / (1 - S_ii)`` where ``S = K (K + lambda I)^{-1}`` -- evaluated over a
        grid via the eigendecomposition of the kernel matrix, so the whole sweep
        costs one ``eigh`` plus O(n^2) per candidate. The chosen penalty is the
        largest whose mean LOO error is within one standard error of the minimum,
        which favours more regularisation (less post-period extrapolation) than the
        bare CV minimum -- mirroring ``AugmentedSyntheticControl._select_lambda``.

        Parameters
        ----------
        x_scaled : numpy array, shape (n_periods, n_donors)
            Standardised donor matrix.
        y : numpy array, shape (n_periods,)
            Treated series on the same rows.
        bandwidth : float
            The resolved RBF length-scale.

        Returns
        -------
        The selected ridge penalty.
        """
        kernel = self._kernel_matrix(x_scaled, x_scaled, bandwidth)
        n = kernel.shape[0]
        y_centered = y - float(y.mean())
        eigvals, eigvecs = np.linalg.eigh(kernel)
        eigvals = np.clip(eigvals, 0.0, None)
        lambdas = self._generate_lambdas(eigvals)
        if n < 3:
            # Too few points for a stable one-SE rule; fall back to the grid median.
            return float(np.median(lambdas))
        coords = eigvecs.T @ y_centered
        eigvecs_sq = eigvecs**2
        fold_err = np.empty((n, lambdas.shape[0]))
        for j, lambda_ in enumerate(lambdas):
            shrink = eigvals / (eigvals + lambda_)
            y_hat = eigvecs @ (shrink * coords)  # S @ y_centered
            leverage = eigvecs_sq @ shrink  # diagonal of S
            denom = np.clip(1.0 - leverage, 1e-8, None)
            fold_err[:, j] = ((y_centered - y_hat) / denom) ** 2
        mean_err = fold_err.mean(axis=0)
        se = fold_err.std(axis=0, ddof=1) / np.sqrt(n)
        j_min = int(np.argmin(mean_err))
        within = np.flatnonzero(mean_err <= mean_err[j_min] + se[j_min])
        return float(lambdas[within[-1]])

    def plot(self) -> None:
        """Plot the actual results, the counterfactual, and the pointwise and cumulative differences.

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
            self.actual_pre[self.y_variable].to_numpy(),
            self.actual_post[self.y_variable].to_numpy(),
            self.prediction_pre[self.y_variable].to_numpy(),
            self.prediction_post[self.y_variable].to_numpy(),
        )
