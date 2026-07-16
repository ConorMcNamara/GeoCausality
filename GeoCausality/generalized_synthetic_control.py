"""Generalized Synthetic Control method for geo-experiment causal inference."""

from typing import Any

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
from narwhals.typing import IntoDataFrame

from GeoCausality._base import EconometricEstimator
from GeoCausality.utils import HoldoutSplitter


class GeneralizedSyntheticControl(EconometricEstimator):
    """Run generalized synthetic control (interactive fixed effects) for our geo-test.

    Where classic synthetic control builds the counterfactual as a convex blend of
    donor geos, the generalized method of Xu (2017) models the outcome with an
    *interactive fixed effects* (latent factor) structure
    ``Y_it = lambda_i' f_t + eps_it``: a small number of latent time factors
    ``f_t`` shared across geos, each weighted by a geo-specific loading
    ``lambda_i``. The factors are learned from the control geos (which are never
    treated), the treated unit's loadings are recovered from its pre-period fit to
    those factors, and the counterfactual is the loadings projected through the
    factors over the post-period. This relaxes the parallel-trends assumption of
    two-way fixed effects, to which it reduces when the factor structure is
    trivial.
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
        n_factors: int | None = None,
        max_factors: int = 5,
        holdout_len: int = 1,
        factor_selection: str = "er",
        conformal_q: float = 1.0,
    ) -> None:
        """Initialize the generalized synthetic control estimator.

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
        n_factors : int, optional
            The number of latent factors to use. If None, it is selected
            automatically by ``factor_selection``, capped at ``max_factors``.
        max_factors : int, default=5
            The largest number of factors considered during selection.
        holdout_len : int, default=1
            Pre-period margin reserved when capping the factor count: the count is
            bounded by ``n_pre - holdout_len - 1`` so the loading fit keeps spare
            degrees of freedom. Also the block length held out per fold when
            ``factor_selection="cv"``.
        factor_selection : {"er", "cv"}, default="er"
            How the factor count is chosen when ``n_factors`` is None.
            ``"er"`` uses the eigenvalue-ratio criterion (Ahn & Horenstein, 2013)
            on the control panel's spectrum -- the factor count is a property of
            the donor pool, so this is robust. ``"cv"`` minimises held-out
            treated pre-period error via cross-validation; it can over-select
            factors and overfit the counterfactual, so it is opt-in.
        conformal_q : float, default=1.0
            The exponent of the moving-block test statistic used for conformal
            inference (p-values and confidence intervals).

        Notes
        -----
        Based on Xu, Yiqing. "Generalized Synthetic Control Method: Causal
        Inference with Interactive Fixed Effects Models." Political Analysis 25.1
        (2017): 57-76.
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
        if n_factors is not None and n_factors < 0:
            raise ValueError(f"n_factors must be non-negative, got {n_factors}")
        if factor_selection not in ("er", "cv"):
            raise ValueError(f"factor_selection must be 'er' or 'cv', got {factor_selection!r}")
        self.n_factors = n_factors
        self.max_factors = max_factors
        self.holdout_len = holdout_len
        self.factor_selection = factor_selection
        self.conformal_q = conformal_q
        self.dates: list[Any] | None = None
        self.n_factors_selected: int | None = None
        self.prediction_pre: nw.DataFrame | None = None
        self.prediction_post: nw.DataFrame | None = None
        self.actual_pre: nw.DataFrame | None = None
        self.actual_post: nw.DataFrame | None = None
        # Factor design over all periods, pre-period length and treated series,
        # cached for the leave-one-out jackknife+ refits.
        self._jk_design: np.ndarray | None = None
        self._jk_n_pre: int | None = None
        self._jk_y1: np.ndarray | None = None

    def pre_process(self) -> "GeneralizedSyntheticControl":
        """Aggregate the treated series and record the sorted date axis.

        Returns
        -------
        GeneralizedSyntheticControl
            Itself, so it can be chained with generate().
        """
        super().pre_process()
        self.dates = sorted(self.data[self.date_variable].unique().to_list())
        return self

    def generate(self) -> "GeneralizedSyntheticControl":
        """Fit the factor model and build the counterfactual, lift and inference.

        Returns
        -------
        GeneralizedSyntheticControl
            Itself, so it can be chained with summarize().
        """
        if self.treatment_variable is None:
            raise ValueError("treatment_variable must not be None")
        # Treated series, aggregated across test geos, split into pre / post.
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
        # Control outcome matrix over all periods: rows = dates, cols = geos.
        control_all = (
            self.data.filter(nw.col(self.treatment_variable) == 0)
            .select([self.y_variable, self.date_variable, self.geo_variable])
            .group_by([self.date_variable, self.geo_variable])
            .agg(nw.col(self.y_variable).sum())
            .sort([self.date_variable, self.geo_variable])
        )
        control_pivot = nw.from_native(
            control_all.to_native().pivot(on=self.geo_variable, index=self.date_variable, values=self.y_variable),
            eager_only=True,
        ).sort(self.date_variable)
        y0 = control_pivot.drop(self.date_variable).to_numpy()  # T x N
        n_pre = self.actual_pre.shape[0]
        y1_all = np.concatenate(
            [self.actual_pre[self.y_variable].to_numpy(), self.actual_post[self.y_variable].to_numpy()]
        )

        prediction_all = self._fit_factor_model(y0, y1_all, n_pre)

        self.prediction_pre = nw.from_native(
            pl.DataFrame(
                {
                    self.date_variable: self.actual_pre[self.date_variable].to_native(),
                    self.y_variable: prediction_all[:n_pre],
                }
            ),
            eager_only=True,
        )
        self.prediction_post = nw.from_native(
            pl.DataFrame(
                {
                    self.date_variable: self.actual_post[self.date_variable].to_native(),
                    self.y_variable: prediction_all[n_pre:],
                }
            ),
            eager_only=True,
        )
        self.results = {
            "test": self.actual_post[self.y_variable].to_numpy(),
            "counterfactual": self.prediction_post[self.y_variable].to_numpy(),
            "lift": self.actual_post[self.y_variable].to_numpy() - self.prediction_post[self.y_variable].to_numpy(),
            "n_factors": self.n_factors_selected,
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

    def _fit_factor_model(self, y0: np.ndarray, y1_all: np.ndarray, n_pre: int) -> np.ndarray:
        """Fit the interactive-fixed-effects counterfactual for the treated series.

        Factors are the leading left singular vectors of the column-centred
        control matrix (estimated from controls over all periods, since controls
        are never treated). An intercept plus the chosen number of factors are fit
        to the treated unit's pre-period by least squares, and the resulting
        loadings are projected through the factors over every period.

        Parameters
        ----------
        y0 : numpy array, shape (T, N)
            Control outcome matrix, rows = dates, cols = control geos.
        y1_all : numpy array, shape (T,)
            Treated series over all periods (pre then post).
        n_pre : int
            The number of pre-period dates.

        Returns
        -------
        The counterfactual treated series over all periods, shape (T,).
        """
        # Column-centre so the factors capture common temporal dynamics; the
        # intercept added below carries the treated unit's level.
        y0_centered = y0 - y0.mean(axis=0, keepdims=True)
        u, s, _ = np.linalg.svd(y0_centered, full_matrices=False)

        max_r = max(0, min(self.max_factors, u.shape[1], n_pre - self.holdout_len - 1))
        if self.n_factors is not None:
            r = min(self.n_factors, max_r)
        elif self.factor_selection == "cv":
            r = self._select_n_factors(u[:n_pre], y1_all[:n_pre], max_r)
        else:
            r = self._eigenvalue_ratio_factors(s, max_r)
        self.n_factors_selected = r

        factors = self._design(u, r)
        beta = self._ols(factors[:n_pre], y1_all[:n_pre])
        self._jk_design = factors
        self._jk_n_pre = n_pre
        self._jk_y1 = y1_all
        return factors @ beta

    def _loo_counterfactuals(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Leave-one-out pre-period residuals and post-period counterfactuals.

        The latent factors are estimated from the controls and so are unchanged by
        dropping a treated pre-period day; only the loading regression is refit.
        Each fold therefore costs a single least-squares solve, making faithful
        jackknife+ cheap for this estimator.

        Returns
        -------
        ``(loo_residuals, loo_post_predictions)`` of shapes ``(n_folds,)`` and
        ``(n_folds, t1)``, or None if the model has not been fit.
        """
        if self._jk_design is None or self._jk_n_pre is None or self._jk_y1 is None:
            return None
        design, n_pre, y = self._jk_design, self._jk_n_pre, self._jk_y1
        pre_x, pre_y, post_x = design[:n_pre], y[:n_pre], design[n_pre:]
        n_params = pre_x.shape[1]
        loo_resid, loo_post = [], []
        for i in range(n_pre):
            mask = np.arange(n_pre) != i
            if mask.sum() <= n_params:
                continue  # underdetermined fold; skip
            beta = self._ols(pre_x[mask], pre_y[mask])
            loo_resid.append(abs(float(pre_y[i] - pre_x[i] @ beta)))
            loo_post.append(post_x @ beta)
        if not loo_resid:
            return None
        return np.array(loo_resid), np.array(loo_post)

    def _bootstrap_refit(self, treated_pre: np.ndarray) -> np.ndarray | None:
        """Refit the counterfactual over the post-period from a resampled pre-period.

        The latent factors come from the controls and are unchanged by resampling
        the treated series, so each bootstrap replicate is a single least-squares
        refit of the loadings on the cached design.

        Parameters
        ----------
        treated_pre : numpy array, shape (n_pre,)
            A bootstrap-resampled treated pre-period series.

        Returns
        -------
        The refit post-period counterfactual, or None if the model is not fit.
        """
        if self._jk_design is None or self._jk_n_pre is None:
            return None
        design, n_pre = self._jk_design, self._jk_n_pre
        beta = self._ols(design[:n_pre], treated_pre)
        return design[n_pre:] @ beta

    @staticmethod
    def _design(u: np.ndarray, r: int) -> np.ndarray:
        """Assemble the design matrix: an intercept plus ``r`` leading factors.

        Parameters
        ----------
        u : numpy array, shape (T, k)
            Left singular vectors of the centred control matrix.
        r : int
            The number of leading factors to include.

        Returns
        -------
        The design matrix of shape (T, r + 1).
        """
        intercept = np.ones((u.shape[0], 1))
        if r == 0:
            return intercept
        return np.column_stack([intercept, u[:, :r]])

    @staticmethod
    def _ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Least-squares solution, robust to rank deficiency.

        Parameters
        ----------
        x : numpy array, shape (n, p)
            Design matrix.
        y : numpy array, shape (n,)
            Target vector.

        Returns
        -------
        The coefficient vector of shape (p,).
        """
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        return beta

    @staticmethod
    def _eigenvalue_ratio_factors(singular_values: np.ndarray, max_r: int) -> int:
        """Choose the factor count by the eigenvalue-ratio criterion.

        Selects the number of latent factors from the control panel's own spectrum
        (Ahn & Horenstein, 2013): with eigenvalues ``mu_k`` (the squared singular
        values of the centred control matrix in descending order), the estimated
        factor count is the ``k`` in ``1..max_r`` that maximises the adjacent
        ratio ``mu_k / mu_{k+1}``. A genuine factor leaves a large gap before the
        noise eigenvalues, so the ratio spikes at the true count.

        This deliberately replaces a treated-pre-period cross-validation: minimising
        held-out *treated* error rewards ever-richer factor subspaces that fit the
        pre-period but overfit the counterfactual (they reconstruct the treated
        unit's post-period and wash out the effect). The factor *count* is a
        property of the donor panel's covariance, so we estimate it there instead.

        Parameters
        ----------
        singular_values : numpy array
            Singular values of the centred control matrix, descending.
        max_r : int
            The largest factor count to consider.

        Returns
        -------
        The selected factor count, in ``0..max_r``.
        """
        if max_r < 1:
            return 0
        eig = np.asarray(singular_values, dtype=float)[: max_r + 1] ** 2
        if eig.shape[0] < 2:
            return min(1, max_r)
        denom = np.where(eig[1:] <= 0.0, np.finfo(float).tiny, eig[1:])
        ratios = eig[:-1] / denom  # ratios[k-1] = mu_k / mu_{k+1} for k = 1..len
        return int(np.argmax(ratios)) + 1

    def _select_n_factors(self, u_pre: np.ndarray, y1_pre: np.ndarray, max_r: int) -> int:
        """Choose the factor count by cross-validation on the pre-period.

        Opt-in alternative to the eigenvalue-ratio criterion (``factor_selection
        ="cv"``). For each candidate factor count, a moving block of the
        pre-period is held out (via :class:`~GeoCausality.utils.HoldoutSplitter`),
        the loadings are fit on the remainder and scored on the block; the count
        with the smallest mean held-out error wins. Because more factors keep
        improving the pre-period fit, this can over-select and overfit the
        counterfactual -- prefer ``"er"`` unless you specifically want CV.

        Parameters
        ----------
        u_pre : numpy array, shape (n_pre, k)
            Pre-period rows of the control left singular vectors.
        y1_pre : numpy array, shape (n_pre,)
            Treated series over the pre-period.
        max_r : int
            The largest factor count to consider.

        Returns
        -------
        The selected factor count.
        """
        best_r, best_err = 0, np.inf
        for r in range(max_r + 1):
            design = self._design(u_pre, r)
            df = pd.DataFrame(design)
            ser = pd.Series(y1_pre)
            errors, folds = 0.0, 0
            for x_train, x_holdout, y_train, y_holdout in HoldoutSplitter(df, ser, self.holdout_len):
                if x_train.shape[0] <= x_train.shape[1]:
                    continue  # underdetermined fold; skip
                beta = self._ols(x_train.to_numpy(), y_train.to_numpy())
                pred = x_holdout.to_numpy() @ beta
                errors += float(np.sum((y_holdout.to_numpy() - pred) ** 2))
                folds += 1
            if folds == 0:
                continue
            mse = errors / folds
            if mse < best_err:
                best_err, best_r = mse, r
        return best_r
