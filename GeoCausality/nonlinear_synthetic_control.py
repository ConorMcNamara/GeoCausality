"""Nonlinear Synthetic Control (Tian 2023) for geo-experiment causal inference."""

from typing import Any

import narwhals as nw
import numpy as np
import polars as pl
from narwhals.typing import IntoDataFrame
from scipy.optimize import Bounds, LinearConstraint, minimize

from GeoCausality._base import EconometricEstimator


class NonlinearSyntheticControl(EconometricEstimator):
    """Run Nonlinear Synthetic Control (Tian 2023) for our geo-test.

    Tian (2023) generalises Abadie, Diamond & Hainmueller's synthetic control to
    the case where the untreated outcome is a *strictly monotonic nonlinear*
    function of a latent linear index, ``y = F(x'beta + mu'lambda + eps)``. The
    key result is that because ``F`` is monotonic, matching the treated unit's
    *observed* pre-period outcomes to a weighted combination of donor outcomes
    implicitly matches the latent index -- so the counterfactual is still the
    familiar linear-in-weights ``sum_j w_j y_jt`` and no link function has to be
    specified. The "nonlinear" adaptation is entirely in how the weights are
    solved for (Tian 2023, eq. 7)::

        min_w  ||z_1 - sum_j w_j z_j||^2
               + a * sum_j |w_j| * ||z_1 - z_j||          (distance-weighted L1)
               + b * sum_j w_j^2                          (ridge L2)
        s.t.   sum_j w_j = 1

    where ``z_1`` is the treated unit's pre-period trajectory and ``z_j`` each
    donor's. Two departures from canonical SC make it work under a nonlinear
    ``F``:

    * **weights are affine, not simplex** -- they sum to one but may be negative,
      so the synthetic unit can match a treated unit outside the donor convex
      hull (as ``AugmentedSyntheticControl`` also allows);
    * **a distance-weighted L1 penalty** concentrates weight on donors close to
      the treated unit (nearest-neighbour matching as ``a -> inf``), while the
      **L2 ridge** spreads it out (difference-in-differences as ``b -> inf``).

    Unlike ``KernelSyntheticControl`` -- which learns a genuinely nonlinear map of
    the donor outcomes and can revert to the pre-period mean when extrapolating --
    this keeps a linear combination of donor *outcomes*, so it tracks trends and
    reproduces the canonical Proposition 99 result.

    Notes
    -----
    Tian, W. (2023). "The Synthetic Control Method with Nonlinear Outcomes."
    arXiv:2306.01967. The eigenvalue penalty scaling follows the ``mlsynth``
    reference implementation; the penalties are chosen by rolling-origin
    cross-validation (see ``_select_penalties``).
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
        a: float | None = None,
        b: float | None = None,
        cv_grid_step: float = 0.1,
        conformal_q: float = 1.0,
    ) -> None:
        """Initialize the nonlinear synthetic control estimator.

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
        a : float, optional
            The distance-weighted L1 penalty. If either ``a`` or ``b`` is not
            given, both are selected by cross-validation.
        b : float, optional
            The ridge (L2) penalty. If either ``a`` or ``b`` is not given, both are
            selected by cross-validation.
        cv_grid_step : float, default=0.1
            The step of the ``[0, 1]`` multiplier grid searched when ``a`` / ``b``
            are cross-validated. Smaller is more faithful but more expensive
            (the cost is quadratic in the number of grid points).
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
        )
        self.dates: list[Any] | None = None
        self.prediction_pre: nw.DataFrame | None = None
        self.prediction_post: nw.DataFrame | None = None
        self.actual_pre: nw.DataFrame | None = None
        self.actual_post: nw.DataFrame | None = None
        self.a = a
        self.b = b
        self.cv_grid_step = cv_grid_step
        self.conformal_q = conformal_q
        # Penalties resolved on the full pre-period fit and reused across the
        # jackknife+/bootstrap refits, so every fold shares one (a, b) regime.
        self._fit_a: float | None = None
        self._fit_b: float | None = None

    def pre_process(self) -> "NonlinearSyntheticControl":
        """Mark the treatment periods and record the sorted date axis.

        Returns
        -------
        NonlinearSyntheticControl
            Itself, so it can be chained with generate().
        """
        super().pre_process()
        self.dates = sorted(self.data[self.date_variable].unique().to_list())
        return self

    def generate(self) -> "NonlinearSyntheticControl":
        """Build the counterfactual from the NSC weights and compute lift and inference.

        Returns
        -------
        NonlinearSyntheticControl
            Itself, so it can be chained with summarize().
        """
        if self.treatment_variable is None:
            raise ValueError("treatment_variable must not be None")
        # Re-resolve penalties on each generate() so a re-run does not reuse a
        # stale (a, b) from a previous fit.
        self._fit_a = None
        self._fit_b = None

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

        # Resolve (a, b): use the supplied values, else rolling-origin CV over the
        # treated unit's pre-period.
        if self.a is not None and self.b is not None:
            self._fit_a, self._fit_b = float(self.a), float(self.b)
        else:
            self._fit_a, self._fit_b = self._select_penalties(self._jk_y_pre, control_pre_mat)

        self.model = self._create_model(self._jk_y_pre, control_pre_mat)
        prediction_pre_arr = control_pre_mat @ self.model
        prediction_post_arr = control_post_mat @ self.model

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

    def _fit_predict_weights(self, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray | None:
        """Refit the NSC weights on a pre-period subset and predict.

        The penalties ``(self._fit_a, self._fit_b)`` are held at their full-fit
        values (not re-selected per fold) so the leave-one-out counterfactuals
        reflect the same model.

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
        return x_eval @ self._create_model(y_train, x_train)

    def _create_model(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Solve the Tian (2023) NSC weights over the pre-period.

        Parameters
        ----------
        y : numpy array, shape (n_pre,)
            The treated unit's pre-period trajectory (``z_1``).
        x : numpy array, shape (n_pre, n_donors)
            The donor pre-period matrix (rows = dates, cols = donor geos).

        Returns
        -------
        The affine (sum-to-one, possibly negative) donor weights.
        """
        a = self._fit_a if self._fit_a is not None else 0.0
        b = self._fit_b if self._fit_b is not None else 0.0
        weights, _converged = self._solve_nsc(np.asarray(y, dtype=float).ravel(), np.asarray(x, dtype=float), a, b)
        return weights

    @staticmethod
    def _solve_nsc(y: np.ndarray, x: np.ndarray, a: float, b: float) -> tuple[np.ndarray, bool]:
        """Solve the NSC quadratic program.

        Minimises ``||y - x w||^2 + a sum_j d_j |w_j| + b ||w||^2`` subject to
        ``sum_j w_j = 1`` (signed weights). The non-smooth ``|w_j|`` is handled by
        splitting ``w = p - n`` with ``p, n >= 0`` (so ``|w_j| = p_j + n_j`` at the
        optimum), turning the problem into a smooth bound-constrained QP that SLSQP
        solves reliably.

        Parameters
        ----------
        y : numpy array, shape (n_pre,)
            Treated pre-period trajectory.
        x : numpy array, shape (n_pre, n_donors)
            Donor pre-period matrix.
        a : float
            Distance-weighted L1 penalty.
        b : float
            Ridge (L2) penalty.

        Returns
        -------
        weights : numpy array, shape (n_donors,)
            The donor weight vector.
        converged : bool
            Whether SLSQP reached a genuine optimum. When it does not (an
            ill-conditioned or over-penalised cell), the solver falls back toward
            the uniform-weight ``x0``; the caller uses this flag to keep the
            cross-validated penalty search off those degenerate cells.
        """
        n_donors = x.shape[1]
        distance = np.linalg.norm(y.reshape(-1, 1) - x, axis=0)
        # Tiny ridge floor keeps the QP strictly convex when b == 0 and donors
        # outnumber pre-period points.
        gram = x.T @ x + (b + 1e-8) * np.identity(n_donors)
        cross = x.T @ y

        def loss(pn: np.ndarray) -> float:
            weights = pn[:n_donors] - pn[n_donors:]
            l1_penalty = a * np.sum(distance * (pn[:n_donors] + pn[n_donors:]))
            return float(weights @ gram @ weights - 2.0 * cross @ weights + l1_penalty)

        def jac(pn: np.ndarray) -> np.ndarray:
            weights = pn[:n_donors] - pn[n_donors:]
            grad_w = 2.0 * gram @ weights - 2.0 * cross
            return np.concatenate([grad_w + a * distance, -grad_w + a * distance])

        bounds = Bounds(lb=np.zeros(2 * n_donors), ub=np.full(2 * n_donors, np.inf))
        constraints = LinearConstraint(A=np.concatenate([np.ones(n_donors), -np.ones(n_donors)]), lb=1.0, ub=1.0)
        x0 = np.concatenate([np.full(n_donors, 1.0 / n_donors), np.zeros(n_donors)])
        res = minimize(
            fun=loss,
            x0=x0,
            jac=jac,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": 500, "ftol": 1e-10},
        )
        solution = res["x"]
        return solution[:n_donors] - solution[n_donors:], bool(res.success)

    def _select_penalties(self, y_pre: np.ndarray, pre_mat: np.ndarray) -> tuple[float, float]:
        """Choose ``(a, b)`` by a full 2-D grid, rolling-origin cross-validation.

        Searches the eigenvalue-scaled multipliers ``a*, b* in {0, step, ..., 1}``
        for the pair minimising a *rolling-origin* validation error on the treated
        unit's pre-period: over expanding windows, fit the weights on the periods
        so far and score the forecast of the whole remaining tail. This mirrors the
        actual estimand -- projecting the counterfactual across the post-period --
        so it does not
        over-regularise toward the uniform-weight (difference-in-differences) limit
        the way a leave-one-donor-out score does when the treated unit is an
        outlier (that limit fits the "average" donor but flips the sign on a
        high/low-level treated unit). Averaging over many split points also
        stabilises the choice: a single split leaves the signed-weight fit
        underdetermined when donors outnumber the training rows, so the argmin
        would otherwise wander with the donor column order.

        Tian (2023) proposes a coordinate descent, but the CV surface here is not
        coordinate-separable (the L1 and ridge penalties interact), so a plain
        coordinate descent gets trapped short of the optimum; the full grid is
        robust and the cost is quadratic in the (coarse) grid size.

        Parameters
        ----------
        y_pre : numpy array, shape (n_pre,)
            The treated unit's pre-period trajectory.
        pre_mat : numpy array, shape (n_pre, n_donors)
            Donor pre-period matrix.

        Returns
        -------
        The selected ``(a, b)`` raw penalties.
        """
        eigvals = np.sort(np.linalg.eigvalsh(pre_mat.T @ pre_mat))
        n_donors = pre_mat.shape[1]
        grid = np.round(np.arange(0.0, 1.0 + 1e-9, self.cv_grid_step), 4)
        # Prefer the best *converged* cell. Cells where SLSQP fails to reach an
        # optimum collapse toward uniform weights; their CV score is not only
        # meaningless but also platform-dependent (the exact stuck point varies
        # with the BLAS backend), so the raw argmin over all cells is not
        # reproducible across machines. Restricting the choice to converged cells
        # keeps the selection stable and off the degenerate fallback; ``best_any``
        # is a defensive fallback for the (unobserved) case where nothing converges.
        best_conv_score, best_conv = np.inf, None
        best_any_score, best_any = np.inf, (0.0, 0.0)
        for b_star in grid:
            b = self._scale_penalty(float(b_star), eigvals, n_donors)
            for a_star in grid:
                a = self._scale_penalty(float(a_star), eigvals, n_donors, shift=b)
                score, converged = self._cv_score(a, b, y_pre, pre_mat)
                if converged and score < best_conv_score:
                    best_conv_score, best_conv = score, (a, b)
                if score < best_any_score:
                    best_any_score, best_any = score, (a, b)
        return best_conv if best_conv is not None else best_any

    @staticmethod
    def _scale_penalty(star: float, eigvals: np.ndarray, n_donors: int, shift: float = 0.0) -> float:
        """Eigenvalue-scale a ``[0, 1]`` multiplier to a raw penalty (Tian 2023).

        Parameters
        ----------
        star : float
            The multiplier in ``[0, 1]``.
        eigvals : numpy array
            Ascending eigenvalues of the donor Gram matrix ``x'x``.
        n_donors : int
            The number of donors.
        shift : float, default=0.0
            Added to the selected eigenvalue before scaling. Used for the L1
            multiplier, whose spectrum is that of ``x'x + b I`` (i.e. shifted by
            the ridge penalty ``b``).

        Returns
        -------
        The raw penalty.
        """
        if star <= 0.0:
            return 0.0
        idx = min(n_donors, int(np.ceil(n_donors * star)))
        return star * (float(eigvals[idx - 1]) + shift)

    def _cv_score(
        self, a: float, b: float, y_pre: np.ndarray, pre_mat: np.ndarray, min_train_fraction: float = 0.6
    ) -> tuple[float, bool]:
        """Rolling-origin, predict-to-horizon validation error for a candidate ``(a, b)``.

        Over expanding windows -- training on the first ``k`` pre-periods -- for
        every origin ``k`` from ``min_train_fraction`` of the pre-period to its end,
        the weights forecast the *entire remaining tail* ``[k:]`` and the mean
        squared error of that multi-step forecast is recorded; the score is the
        average over origins. Forecasting the whole tail (rather than a single step)
        mirrors the estimand -- projecting the counterfactual across the full
        post-period -- which is what lands the estimate on the published Prop 99
        magnitude; averaging over origins keeps the penalty choice stable (a single
        split is underdetermined and column-order sensitive).

        Parameters
        ----------
        a, b : float
            Candidate raw penalties.
        y_pre : numpy array, shape (n_pre,)
            Treated pre-period trajectory.
        pre_mat : numpy array, shape (n_pre, n_donors)
            Donor pre-period matrix.
        min_train_fraction : float, default=0.6
            The first origin trains on this fraction of the pre-period.

        Returns
        -------
        score : float
            The mean multi-step forecast squared error over the rolling origins.
        converged : bool
            Whether SLSQP converged on *every* rolling-origin fit. A candidate is
            only trusted for selection when all its folds reach a genuine optimum.
        """
        y_pre = np.asarray(y_pre, dtype=float).ravel()
        n_pre = y_pre.shape[0]
        if n_pre < 2:
            weights, converged = self._solve_nsc(y_pre, pre_mat, a, b)
            return float(np.mean((y_pre - pre_mat @ weights) ** 2)), converged
        start = min(max(int(round(n_pre * min_train_fraction)), 2), n_pre - 1)
        errors = np.empty(n_pre - start)
        all_converged = True
        for k in range(start, n_pre):
            weights, converged = self._solve_nsc(y_pre[:k], pre_mat[:k], a, b)
            all_converged = all_converged and converged
            errors[k - start] = float(np.mean((y_pre[k:] - pre_mat[k:] @ weights) ** 2))
        return float(np.mean(errors)), all_converged

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
