"""CausalImpact-style estimator: a Bayesian structural time-series counterfactual."""

from typing import Any

import narwhals as nw
import numpy as np
import statsmodels.api as sm
from narwhals.typing import IntoDataFrame
from scipy.stats import norm

from GeoCausality._base import EconometricEstimator


class CausalImpact(EconometricEstimator):
    """Estimate geo-test lift with a structural time-series counterfactual.

    Where :class:`~GeoCausality.synthetic_control.SyntheticControl` builds the
    counterfactual as a static, simplex-weighted blend of donor geos, CausalImpact
    fits a structural time-series model (via statsmodels'
    :class:`~statsmodels.tsa.statespace.structural.UnobservedComponents`): a
    time-varying level/trend plus an optional seasonal component and a regression on
    the donor geos. The post-period forecast, conditioned on the donors' observed
    post-period values, is the counterfactual, and the forecast's own error variance
    yields the confidence band -- no placebo/permutation step is required.

    This mirrors Brodersen et al.'s CausalImpact, which uses a Bayesian structural
    time series (BSTS) fit by MCMC; statsmodels fits the same state-space model by
    maximum likelihood, so this is the frequentist structural-TS cousin.
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
        level: str = "local level",
        seasonal: int | None = None,
        conformal_q: float = 1.0,
    ) -> None:
        """Initialize the CausalImpact estimator.

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
        level : str, default="local level"
            The trend specification of the structural model, passed to
            ``UnobservedComponents``. ``"local level"`` (a random-walk baseline) is
            the robust default; ``"local linear trend"`` adds a stochastic slope for
            series with drift, at the cost of wider post-period extrapolation.
        seasonal : int, optional
            The seasonal period (e.g. ``7`` for weekly seasonality in daily data).
            ``None`` omits the seasonal component.
        conformal_q : float, default=1.0
            The exponent of the moving-block test statistic, used only when
            ``inference_method`` routes through conformal inference.

        Notes
        -----
        Based on Brodersen, Gallusser, Koehler, Remy & Scott :cite:`causalimpact2015`.

        The default ``inference_method`` (``"auto"``) uses the structural model's
        native forecast intervals. Setting ``inference_method`` to ``"conformal"``,
        ``"jackknife"`` or ``"bootstrap"`` instead routes the pre-/post-period
        residuals through the shared conformal machinery for parity with the
        synthetic-control estimators.
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
        self.level = level
        self.seasonal = seasonal
        self.conformal_q = conformal_q
        # Posterior-simulation settings for the native structural inference: the
        # number of counterfactual paths drawn from the fitted state distribution
        # and the seed that makes them reproducible.
        self.n_sim: int = 1000
        self.sim_seed: int = 0
        self.synthetic_control_df: nw.DataFrame | None = None
        self.synthetic_test_df: nw.DataFrame | None = None
        self.actual_pre: np.ndarray | None = None
        self.actual_post: np.ndarray | None = None
        self.prediction_pre: np.ndarray | None = None
        self.prediction_post: np.ndarray | None = None
        self.dates: list[Any] | None = None

    def pre_process(self) -> "CausalImpact":
        """Aggregate the pre-/post-period treated series and donor matrices.

        Builds the same donor design as SyntheticControl: the treated series joined
        to a wide (date x donor-geo) matrix, split into the pre-period training frame
        and the post-period evaluation frame.

        Returns
        -------
        CausalImpact
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

    def generate(self) -> "CausalImpact":
        """Fit the structural model, build the counterfactual, and compute lift and inference.

        Returns
        -------
        CausalImpact
            Itself, so it can be chained with summarize().
        """
        if self.synthetic_control_df is None:
            raise ValueError("synthetic_control_df must not be None")
        if self.synthetic_test_df is None:
            raise ValueError("synthetic_test_df must not be None")
        train_x = self.synthetic_control_df.drop([self.date_variable, self.y_variable]).to_numpy()
        test_x = self.synthetic_test_df.drop([self.date_variable, self.y_variable]).to_numpy()
        self.actual_pre = self.synthetic_control_df[self.y_variable].to_numpy()
        self.actual_post = self.synthetic_test_df[self.y_variable].to_numpy()

        # Fit the structural time series on the pre-period, with the donor geos as
        # exogenous regressors, then forecast the post-period conditioned on the
        # donors' observed post-period values -- that forecast is the counterfactual.
        self.model = sm.tsa.UnobservedComponents(
            endog=np.asarray(self.actual_pre, dtype=float),
            exog=np.asarray(train_x, dtype=float),
            level=self.level,
            seasonal=self.seasonal,
        ).fit(disp=False)
        self.prediction_pre = np.asarray(self.model.get_prediction().predicted_mean, dtype=float)
        forecast = self.model.get_forecast(steps=len(self.actual_post), exog=np.asarray(test_x, dtype=float))
        self.prediction_post = np.asarray(forecast.predicted_mean, dtype=float)

        self.results = {
            "test": self.actual_post,
            "counterfactual": self.prediction_post,
            "lift": self.actual_post - self.prediction_post,
        }
        self.results["incrementality"] = float(np.sum(self.results["lift"]))
        if self.inference_method in ("auto", "structural"):
            self.results.update(
                self._structural_inference(self.actual_post, self.prediction_post, np.asarray(test_x, dtype=float))
            )
        else:
            self.results.update(
                self._conformal_inference(
                    self.actual_pre,
                    self.prediction_pre,
                    self.actual_post,
                    self.prediction_post,
                    q=self.conformal_q,
                )
            )
        return self

    def _structural_inference(
        self,
        actual_post: np.ndarray,
        prediction_post: np.ndarray,
        exog_post: np.ndarray,
    ) -> dict[str, Any]:
        """Confidence band, effect CIs and p-value from posterior counterfactual draws.

        Draws ``n_sim`` counterfactual post-period paths from the fitted model
        (:meth:`_simulate_counterfactual`) and forms the cumulative-effect
        distribution ``sum(actual) - sum(path)`` across draws. This is the proper
        analogue of CausalImpact's posterior predictive: the interval is the
        percentile interval of the cumulative-effect draws (so it captures the
        serial correlation in the state and the observation noise, not just summed
        per-period variances), and the p-value is the two-sided posterior tail-area
        probability -- twice the smaller of the mass on either side of a zero effect.

        The point ``incrementality`` (set in ``generate``) stays the deterministic
        ``sum(actual - forecast mean)``; the draws provide only its uncertainty.

        Parameters
        ----------
        actual_post : numpy array
            Observed post-period outcome.
        prediction_post : numpy array
            Counterfactual (forecast mean) over the post-period.
        exog_post : numpy array
            Donor (exogenous) matrix over the post-period, conditioning the draws.

        Returns
        -------
        A dict suitable for ``self.results.update(...)`` with the effect CIs, the
        per-period ``lift`` CIs, the plot's ``conformal_band`` half-width, the
        ``p_value`` and ``method = "structural-ts"``.
        """
        t1 = len(actual_post)
        z = float(norm.ppf(1 - self.alpha / 2))
        sims = self._simulate_counterfactual(exog_post, t1)  # (n_sim, t1)
        actual_cum = float(np.sum(actual_post))
        effect_draws = actual_cum - sims.sum(axis=1)  # cumulative-effect posterior
        lower = float(np.percentile(effect_draws, 100 * self.alpha / 2))
        upper = float(np.percentile(effect_draws, 100 * (1 - self.alpha / 2)))
        # Two-sided posterior tail-area probability of a zero cumulative effect.
        p_one = float(np.mean(effect_draws <= 0))
        p_value = 2.0 * min(p_one, 1.0 - p_one)
        # Representative pointwise half-width for the shared three-panel plot, from
        # the per-period spread of the simulated counterfactual paths.
        band = float(z * np.mean(sims.std(axis=0)))
        return {
            "incrementality_ci_lower": lower,
            "incrementality_ci_upper": upper,
            "lift_ci_lower": lower / t1,
            "lift_ci_upper": upper / t1,
            "conformal_band": band,
            "p_value": p_value,
            "method": "structural-ts",
        }

    def _simulate_counterfactual(self, exog_post: np.ndarray, steps: int) -> np.ndarray:
        """Draw post-period counterfactual paths from the fitted state posterior.

        Each draw seeds the simulation with an initial state sampled from the fitted
        model's final predicted-state distribution (mean ``predicted_state`` and
        covariance ``predicted_state_cov`` at the end of the pre-period), so the
        paths carry the uncertainty in where the level/trend ended -- not just the
        forward measurement and state shocks -- then rolls forward ``steps`` periods
        conditioned on the donors' observed post-period values.

        Parameters
        ----------
        exog_post : numpy array, shape (steps, n_donors)
            Donor matrix over the post-period.
        steps : int
            The number of post-period periods to simulate.

        Returns
        -------
        A ``(n_sim, steps)`` array of simulated counterfactual observation paths.
        """
        res = self.model
        state_mean = np.asarray(res.predicted_state[:, -1], dtype=float)
        state_cov = np.asarray(res.predicted_state_cov[:, :, -1], dtype=float)
        state_cov = 0.5 * (state_cov + state_cov.T)  # symmetrise against round-off
        # Derive independent, reproducible seeds for the initial-state draws and for
        # each path's forward shocks: statsmodels' simulate() otherwise draws those
        # shocks from the global RNG, which would make runs non-reproducible.
        init_seq, sim_seq = np.random.SeedSequence(self.sim_seed).spawn(2)
        init_rng = np.random.default_rng(init_seq)
        init_states = init_rng.multivariate_normal(state_mean, state_cov, size=self.n_sim, check_valid="ignore")
        sim_seeds = sim_seq.generate_state(self.n_sim)
        sims = np.empty((self.n_sim, steps))
        for m in range(self.n_sim):
            sims[m] = np.asarray(
                res.simulate(
                    steps,
                    anchor="end",
                    exog=exog_post,
                    initial_state=init_states[m],
                    random_state=int(sim_seeds[m]),
                ),
                dtype=float,
            ).ravel()
        return sims
