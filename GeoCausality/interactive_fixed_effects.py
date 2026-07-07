"""Interactive fixed effects (Bai 2009) panel model for geo-experiment causal inference."""

from datetime import date as date_cls

import narwhals as nw
import numpy as np
import plotly.graph_objects as go
import polars as pl
from narwhals.typing import IntoDataFrame
from plotly.subplots import make_subplots

from GeoCausality._base import EconometricEstimator


class InteractiveFixedEffects(EconometricEstimator):
    """Run an interactive fixed effects (Bai 2009) panel model for our geo-test.

    Where two-way fixed effects assumes a common time shock ``xi_t`` that hits
    every geo equally, the interactive fixed effects model lets a small number of
    latent time factors ``f_t`` load onto each geo with a geo-specific weight
    ``lambda_i``::

        Y_it = delta * D_it + alpha_i + xi_t + lambda_i' f_t + eps_it

    where ``D_it`` is the treatment indicator (1 for a treated geo in the
    post-period) and ``delta`` is the average treatment effect on the treated.
    The additive two-way fixed effects (``alpha_i``, ``xi_t``) and the factors
    ``f_t`` are estimated from the control geos by the Bai (2009) alternating
    algorithm: given the factors, the additive effects are the two-way means of the
    de-factored outcome; given the additive effects, the factors are the leading
    principal components of the de-meaned outcome. The two updates alternate to
    convergence. This differs from :class:`~GeoCausality.fixed_effects.FixedEffects`,
    which drops the interactive term and fits only ``alpha_i + xi_t``; it relaxes
    the parallel-trends assumption, to which it reduces when the factor count is
    zero.

    The transferable time structure -- an intercept, the common time effect and the
    factors -- is then projected onto the treated geos' pre-period by least squares,
    and the fitted loadings extrapolate the no-treatment counterfactual over the
    post-period. Estimating the factors from the never-treated donor pool (rather
    than the full panel) avoids a spare factor adopting the treated units and
    washing out the effect, and matches the generalized synthetic control method
    (Xu 2017); the same conformal-inference contract as the synthetic-control
    estimators then applies to the treated series. This estimator additionally
    carries explicit additive two-way fixed effects and the full Bai alternating
    fit, where
    :class:`~GeoCausality.generalized_synthetic_control.GeneralizedSyntheticControl`
    uses a single column-centred SVD.

    Two estimation modes are offered via ``method`` (see ``__init__``). The default
    ``"projection"`` is the robust control-only counterfactual described above.
    ``"coefficient"`` instead fits the full-panel Bai (2009) model with the
    treatment effect as a genuine regression coefficient ``delta`` estimated
    jointly with the factors -- letting the treated geos' pre-period inform the
    factors, which ``GeneralizedSyntheticControl`` structurally cannot do -- at the
    cost of weaker identification when few geos are treated. Both report the
    per-treated-cell effect as ``self.att``.
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
        max_iter: int = 500,
        tol: float = 1e-6,
        conformal_q: float = 1.0,
        method: str = "projection",
    ) -> None:
        """Initialize the interactive fixed effects estimator.

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
            automatically by the eigenvalue-ratio criterion, capped at
            ``max_factors``.
        max_factors : int, default=5
            The largest number of factors considered during selection.
        max_iter : int, default=500
            The maximum number of alternating-least-squares iterations.
        tol : float, default=1e-6
            Convergence tolerance on the treatment coefficient between iterations.
        conformal_q : float, default=1.0
            The exponent of the moving-block test statistic used for conformal
            inference (p-values and confidence intervals).
        method : {"projection", "coefficient"}, default="projection"
            How the treatment effect is estimated.

            ``"projection"`` estimates the additive fixed effects and factors from
            the control geos alone and projects the treated geos' pre-period onto
            that time structure to build the counterfactual (Xu, 2017). It is
            robust because the treated geos never enter the factor estimation.

            ``"coefficient"`` is the full-panel Bai (2009) estimator: the treatment
            effect is a genuine regression coefficient ``delta`` on the treatment
            indicator, estimated *jointly* with the factors over the whole panel
            (the treated geos' pre-period informs the factors, which
            ``GeneralizedSyntheticControl`` never does). The treated post-period
            cells are held out as missing so the factors cannot simply absorb the
            effect; ``delta`` is the average residual over those cells. This is
            weakly identified when there are few treated geos or the factor count
            is over-selected -- a spare factor can adopt the treated geos and
            attenuate ``delta`` -- so ``"projection"`` is the default.

            Both expose the per-treated-cell effect as ``self.att``.

        Notes
        -----
        Based on Bai, Jushan. "Panel Data Models with Interactive Fixed Effects."
        Econometrica 77.4 (2009): 1229-1279.
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
            raise ValueError("n_factors must be non-negative")
        if method not in ("projection", "coefficient"):
            raise ValueError(f"method must be 'projection' or 'coefficient', got {method!r}")
        self.method = method
        self.n_factors = n_factors
        self.max_factors = max_factors
        self.max_iter = max_iter
        self.tol = tol
        self.conformal_q = conformal_q
        self.n_factors_selected: int | None = None
        self.att: float | None = None
        self.dates: list | None = None
        self.actual_pre: nw.DataFrame | None = None
        self.actual_post: nw.DataFrame | None = None
        self.prediction_pre: nw.DataFrame | None = None
        self.prediction_post: nw.DataFrame | None = None

    def pre_process(self) -> "InteractiveFixedEffects":
        """Assign the treatment period / membership columns and record the date axis.

        Returns
        -------
        InteractiveFixedEffects
            Itself, so it can be chained with generate().
        """
        super().pre_process()
        self.dates = sorted(self.data[self.date_variable].unique().to_list())
        return self

    def generate(self) -> "InteractiveFixedEffects":
        """Fit the interactive fixed effects model and build the counterfactual, lift and inference.

        Returns
        -------
        InteractiveFixedEffects
            Itself, so it can be chained with summarize().
        """
        if self.treatment_variable is None:
            raise ValueError("treatment_variable must not be None")
        if self.dates is None:
            raise ValueError("dates must not be None; call pre_process() first")
        # Balanced (date x geo) panel of the outcome; rows = dates, cols = geos.
        panel = (
            self.data.select([self.y_variable, self.date_variable, self.geo_variable])
            .group_by([self.date_variable, self.geo_variable])
            .agg(nw.col(self.y_variable).sum())
            .sort([self.date_variable, self.geo_variable])
        )
        pivot = nw.from_native(
            panel.to_native().pivot(on=self.geo_variable, index=self.date_variable, values=self.y_variable),
            eager_only=True,
        ).sort(self.date_variable)
        geos = [c for c in pivot.columns if c != self.date_variable]
        y = pivot.drop(self.date_variable).to_numpy()  # T x N

        # Treated-geo mask (column aligned) and the per-date treatment period.
        treated_geos = set(self.data.filter(nw.col(self.treatment_variable) == 1)[self.geo_variable].unique().to_list())
        treated_mask = np.array([g in treated_geos for g in geos])
        period = (
            self.data.select([self.date_variable, "treatment_period"])
            .group_by(self.date_variable)
            .agg(nw.col("treatment_period").max())
            .sort(self.date_variable)
        )
        treatment_period = period["treatment_period"].to_numpy().astype(float)  # T
        n_pre = int(np.sum(treatment_period == 0))

        # Observed treated series (summed over test geos) and its counterfactual.
        actual = y[:, treated_mask].sum(axis=1)
        if self.method == "coefficient":
            predicted, r = self._fit_coefficient(y, treated_mask, treatment_period, n_pre)
        else:
            predicted, r = self._fit_interactive_fe(y, treated_mask, n_pre)
        self.n_factors_selected = r

        pre_dates = self.dates[:n_pre]
        post_dates = self.dates[n_pre:]
        self.actual_pre = self._series_frame(pre_dates, actual[:n_pre])
        self.actual_post = self._series_frame(post_dates, actual[n_pre:])
        self.prediction_pre = self._series_frame(pre_dates, predicted[:n_pre])
        self.prediction_post = self._series_frame(post_dates, predicted[n_pre:])

        lift = actual[n_pre:] - predicted[n_pre:]
        n_treated = int(np.sum(treated_mask))
        n_post = len(post_dates)
        # Per treated-cell average treatment effect on the treated.
        self.att = float(np.sum(lift) / (n_treated * n_post)) if n_treated * n_post > 0 else 0.0
        self.results = {
            "test": self.actual_post[self.y_variable].to_numpy(),
            "counterfactual": self.prediction_post[self.y_variable].to_numpy(),
            "lift": lift,
            "att": self.att,
            "n_factors": r,
        }
        self.results["incrementality"] = float(np.sum(self.results["lift"]))
        self.results.update(
            self._conformal_inference(
                actual[:n_pre],
                predicted[:n_pre],
                actual[n_pre:],
                predicted[n_pre:],
                q=self.conformal_q,
            )
        )
        return self

    def _fit_interactive_fe(self, y: np.ndarray, treated_mask: np.ndarray, n_pre: int) -> tuple[np.ndarray, int]:
        """Fit the interactive fixed effects model and return the treated counterfactual.

        The factor structure is estimated from the *control* geos alone, which are
        never treated: including the treated geos would let a spare factor adopt
        them and, with a free factor count, reconstruct their post-period
        trajectory -- washing out the treatment effect (a well-known degeneracy of
        full-panel interactive fixed effects with few treated units). On the
        control block the additive two-way (unit + time) fixed effects and the
        ``r`` leading factors are estimated by the Bai (2009) alternating
        algorithm: given the factors, the additive effects are the two-way means of
        the de-factored outcome; given the additive effects, the factors are the
        ``r`` leading principal components of the de-meaned outcome. The two
        updates alternate to convergence.

        The transferable time structure -- an intercept, the common time effect and
        the ``r`` factors -- is then projected onto the treated geos' pre-period by
        least squares, and the fitted loadings extrapolate the counterfactual over
        every period (Xu, 2017). The factor count is chosen from the control
        panel's own spectrum, so it reflects the never-treated donor pool.

        Parameters
        ----------
        y : numpy array, shape (T, N)
            Outcome panel, rows = dates, cols = geos.
        treated_mask : numpy array, shape (N,)
            Boolean mask of the treated geos (columns).
        n_pre : int
            The number of pre-period dates.

        Returns
        -------
        ``(counterfactual, r)``: the treated (summed over test geos) counterfactual
        series over every period (shape ``(T,)``) and the number of factors used.
        """
        y0 = y[:, ~treated_mask]  # control block, T x N0
        t, n0 = y0.shape
        max_r = max(0, min(self.max_factors, min(t, n0) - 1, n_pre - 1))
        if self.n_factors is not None:
            r = min(self.n_factors, max_r)
        else:
            # The factor count is a property of the donor pool, so it is chosen
            # from the twoway-demeaned control panel's spectrum.
            _, s0, _ = np.linalg.svd(self._twoway_means(y0)[0], full_matrices=False)
            r = self._eigenvalue_ratio_factors(s0, max_r)

        factor = np.zeros_like(y0)
        f_mat = np.zeros((t, r))
        time_effect = np.zeros(t)
        for _ in range(self.max_iter):
            means, time_effect = self._twoway_means(y0 - factor)
            if r > 0:
                u, s, vt = np.linalg.svd(y0 - means, full_matrices=False)
                f_mat = u[:, :r] * s[:r]
                new_factor = f_mat @ vt[:r, :]
            else:
                new_factor = np.zeros_like(y0)
            if np.max(np.abs(new_factor - factor)) <= self.tol:
                factor = new_factor
                _, time_effect = self._twoway_means(y0 - factor)
                break
            factor = new_factor

        # Transferable time structure: intercept (absorbs the treated level), the
        # common time effect, and the r control-estimated factors.
        columns = [np.ones(t), time_effect]
        if r > 0:
            columns.extend(f_mat[:, k] for k in range(r))
        design = np.column_stack(columns)
        treated = y[:, treated_mask].sum(axis=1)
        beta = self._ols(design[:n_pre], treated[:n_pre])
        return design @ beta, r

    def _fit_coefficient(
        self, y: np.ndarray, treated_mask: np.ndarray, treatment_period: np.ndarray, n_pre: int
    ) -> tuple[np.ndarray, int]:
        """Full-panel Bai (2009) fit: treatment effect as a jointly-estimated coefficient.

        The treatment indicator ``D_it`` is 1 for a treated geo in the post-period.
        The model ``Y_it = delta * D_it + alpha_i + xi_t + lambda_i' f_t`` is fit
        over the *whole* panel by expectation-maximisation: the treated post-period
        cells (``D == 1``) are held out as missing so the factors cannot trivially
        absorb the treatment, the additive two-way effects and ``r`` factors are
        re-estimated on the completed matrix, and the held-out cells are re-imputed
        with the fresh counterfactual. At convergence ``delta`` is the average
        residual over the held-out cells -- the coefficient on ``D``. Unlike
        ``projection`` and unlike ``GeneralizedSyntheticControl``, the treated
        geos' pre-period observations enter the factor estimation, so this is a
        genuine full-panel joint estimate; the price is weak identification when
        few geos are treated (a spare factor can adopt them). The factor count is
        chosen from the control panel's spectrum.

        Parameters
        ----------
        y : numpy array, shape (T, N)
            Outcome panel, rows = dates, cols = geos.
        treated_mask : numpy array, shape (N,)
            Boolean mask of the treated geos (columns).
        treatment_period : numpy array, shape (T,)
            1 for post-period dates, 0 otherwise.
        n_pre : int
            The number of pre-period dates.

        Returns
        -------
        ``(counterfactual, r)``: the treated (summed over test geos) counterfactual
        series over every period (shape ``(T,)``) and the number of factors used.
        """
        missing = (treatment_period[:, None] * treated_mask[None, :]) > 0.0  # treated post cells
        y0 = y[:, ~treated_mask]
        t, n0 = y0.shape
        max_r = max(0, min(self.max_factors, min(t, n0) - 1, n_pre - 1))
        if self.n_factors is not None:
            r = min(self.n_factors, max_r)
        else:
            _, s0, _ = np.linalg.svd(self._twoway_means(y0)[0], full_matrices=False)
            r = self._eigenvalue_ratio_factors(s0, max_r)

        # Initialise the missing cells with the two-way additive fit of the
        # observed cells so the first factor estimate is not distorted by them.
        z = y.copy()
        if missing.any():
            obs = np.where(missing, np.nan, y)
            grand = np.nanmean(obs)
            row = np.nanmean(obs, axis=1, keepdims=True)
            col = np.nanmean(obs, axis=0, keepdims=True)
            init = np.where(np.isnan(row), grand, row) + np.where(np.isnan(col), grand, col) - grand
            z = np.where(missing, init, y)

        common = np.zeros_like(y)
        delta = 0.0
        for _ in range(self.max_iter):
            resid, _ = self._twoway_means(z)
            means = z - resid
            if r > 0:
                u, s, vt = np.linalg.svd(resid, full_matrices=False)
                factor_term = (u[:, :r] * s[:r]) @ vt[:r, :]
            else:
                factor_term = np.zeros_like(y)
            common = means + factor_term
            delta_new = float(np.mean((y - common)[missing])) if missing.any() else 0.0
            z = np.where(missing, common, y)
            if abs(delta_new - delta) <= self.tol * (1.0 + abs(delta)):
                break
            delta = delta_new
        return common[:, treated_mask].sum(axis=1), r

    @staticmethod
    def _twoway_means(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Additive two-way (unit + time) fixed-effects fit of a panel.

        Parameters
        ----------
        w : numpy array, shape (T, N)
            The matrix to decompose.

        Returns
        -------
        ``(residual, time_effect)`` where ``residual = w - (grand + time + unit)``
        is the twoway-demeaned matrix and ``time_effect`` is the per-date effect
        (row mean minus the grand mean), shape ``(T,)``.
        """
        grand = w.mean()
        time_effect = w.mean(axis=1) - grand  # T
        unit_effect = w.mean(axis=0) - grand  # N
        means = grand + time_effect[:, None] + unit_effect[None, :]
        return w - means, time_effect

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

        With eigenvalues ``mu_k`` (the squared singular values of the demeaned
        panel in descending order), the estimated factor count is the ``k`` in
        ``1..max_r`` that maximises the adjacent ratio ``mu_k / mu_{k+1}`` (Ahn &
        Horenstein, 2013): a genuine factor leaves a large gap before the noise
        eigenvalues, so the ratio spikes at the true count.

        Parameters
        ----------
        singular_values : numpy array
            Singular values of the demeaned panel, descending.
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

    def _series_frame(self, dates: list, values: np.ndarray) -> nw.DataFrame:
        """Wrap a date axis and value array into a narwhals frame.

        Parameters
        ----------
        dates : list
            The date axis.
        values : numpy array
            The values aligned to ``dates``.

        Returns
        -------
        A narwhals data frame with the date and outcome columns.
        """
        return nw.from_native(
            pl.DataFrame({self.date_variable: dates, self.y_variable: values}),
            eager_only=True,
        )

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
