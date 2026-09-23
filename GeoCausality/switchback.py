"""Switchback (crossover) experiment estimator for geo-experiment causal inference."""

from typing import Any

import narwhals as nw
import numpy as np
import plotly.graph_objects as go
from linearmodels.panel import PanelOLS
from narwhals.typing import IntoDataFrame
from tabulate import tabulate  # type: ignore

from GeoCausality._base import Estimator


class Switchback(Estimator):
    """Run a switchback (crossover) experiment analysis for geo-tests.

    In a switchback design, treatment assignment alternates over time within
    geos rather than being fixed for the entire experiment. Each geo serves as
    both treatment and control at different time windows, giving stronger
    within-unit identification when the number of geos is small.

    The estimator fits a two-way fixed-effects panel regression with geo and
    time fixed effects.  Cluster-robust standard errors (clustered at the geo
    level) account for within-geo serial correlation.  An optional washout
    period can be specified to discard observations immediately after each
    treatment switch, mitigating carryover bias.  Carryover can also be modeled
    explicitly by including lagged treatment indicators.

    A randomization-inference p-value is available via ``permutation_test()``:
    the switchback schedule is re-randomised across geos and the empirical null
    distribution of test statistics is compared to the observed statistic.

    Notes
    -----
    Based on Bojinov & Shephard (2019), "Time Series Experiments and Causal
    Estimands: Exact Randomization Tests and Trading" and Xiong et al. (2019),
    "Optimal Experimental Design for Staggered Rollouts" (Uber switchback
    framework).
    """

    def __init__(
        self,
        data: IntoDataFrame,
        geo_variable: str = "geo",
        date_variable: str = "date",
        y_variable: str = "y",
        treatment_variable: str = "is_treatment",
        alpha: float = 0.1,
        msrp: float = 0.0,
        spend: float = 0.0,
        washout: int = 0,
        carryover_lags: int = 0,
    ) -> None:
        """Initialize the switchback estimator.

        Parameters
        ----------
        data : pandas or polars data frame
            Panel data with a time-varying binary treatment column.
        geo_variable : str
            Column identifying the geographic unit.
        date_variable : str
            Column identifying the time period.
        y_variable : str
            Outcome column.
        treatment_variable : str
            Column with the time-varying binary treatment indicator (1 = treated
            in this period, 0 = control in this period).
        alpha : float, default=0.1
            Significance level.
        msrp : float, default=0.0
            Average revenue per unit, for revenue/ROAS lifts.
        spend : float, default=0.0
            Campaign spend, for ROAS and cost-per lifts.
        washout : int, default=0
            Number of periods to discard after each treatment switch within a
            geo. Setting ``washout=1`` drops the first observation after every
            on/off or off/on transition, reducing carryover bias at the cost of
            sample size.
        carryover_lags : int, default=0
            Number of lagged treatment indicators to include in the model.
            When > 0, the model includes ``treatment_lag_1`` through
            ``treatment_lag_{carryover_lags}`` as additional regressors. The
            main treatment coefficient then captures the *contemporaneous*
            effect net of carryover.
        """
        super().__init__(
            data,
            geo_variable,
            None,
            None,
            treatment_variable,
            date_variable,
            "1900-01-01",
            "1900-01-02",
            y_variable,
            alpha,
            msrp,
            spend,
        )
        self.washout = washout
        self.carryover_lags = carryover_lags
        self.model: Any = None
        self.panel: nw.DataFrame | None = None
        self.n_treated_obs: int = 0

    def pre_process(self) -> "Switchback":
        """Prepare the panel data: add washout flags and carryover lags.

        Returns
        -------
        Switchback
            Itself, so it can be chained with ``generate()``.
        """
        df = self.data.sort([self.geo_variable, self.date_variable])
        treat_var: str = self.treatment_variable or "is_treatment"

        if self.washout > 0:
            treat_col = nw.col(treat_var)
            shifted = treat_col.shift(1).over(self.geo_variable)
            switch = (treat_col != shifted) & ~shifted.is_null()
            df = df.with_columns(switch.alias("_switch"))
            washout_mask = nw.lit(False)
            for lag in range(self.washout):
                washout_mask = washout_mask | nw.col("_switch").shift(lag).over(self.geo_variable).fill_null(False)
            df = df.filter(~washout_mask).drop("_switch")

        if self.carryover_lags > 0:
            for lag in range(1, self.carryover_lags + 1):
                lag_col = nw.col(treat_var).shift(lag).over(self.geo_variable).fill_null(0)
                df = df.with_columns(lag_col.alias(f"treatment_lag_{lag}"))

        self.panel = df
        self.n_treated_obs = int(df.filter(nw.col(treat_var) == 1).shape[0])
        return self

    def generate(self) -> "Switchback":
        """Fit the two-way fixed-effects model with cluster-robust standard errors.

        Returns
        -------
        Switchback
            Itself, so it can be chained with ``summarize()``.
        """
        if self.panel is None:
            raise ValueError("Call pre_process() before generate()")

        treat_var: str = self.treatment_variable or "is_treatment"
        exog_terms: list[str] = [treat_var]
        for lag in range(1, self.carryover_lags + 1):
            exog_terms.append(f"treatment_lag_{lag}")
        exog_formula = " + ".join(exog_terms)
        formula = f"{self.y_variable} ~ {exog_formula} + EntityEffects + TimeEffects"

        panel_pd = self.panel.to_pandas().set_index([self.geo_variable, self.date_variable])
        model = PanelOLS.from_formula(formula, data=panel_pd)
        self.model = model.fit(cov_type="clustered", cluster_entity=True)

        lift = float(self.model.params[treat_var])
        cis = self.model.conf_int(1 - self.alpha)
        ci_lower = float(cis.loc[treat_var, "lower"])
        ci_upper = float(cis.loc[treat_var, "upper"])
        p_value = float(self.model.pvalues[treat_var])

        self.results = {
            "lift": lift,
            "lift_ci_lower": ci_lower,
            "lift_ci_upper": ci_upper,
            "incrementality": lift * self.n_treated_obs,
            "incrementality_ci_lower": ci_lower * self.n_treated_obs,
            "incrementality_ci_upper": ci_upper * self.n_treated_obs,
            "p_value": p_value,
            "n_treated_obs": self.n_treated_obs,
        }

        if self.carryover_lags > 0:
            carryover: dict[str, dict[str, float]] = {}
            for lag in range(1, self.carryover_lags + 1):
                key = f"treatment_lag_{lag}"
                carryover[key] = {
                    "estimate": float(self.model.params[key]),
                    "p_value": float(self.model.pvalues[key]),
                    "ci_lower": float(cis.loc[key, "lower"]),
                    "ci_upper": float(cis.loc[key, "upper"]),
                }
            self.results["carryover"] = carryover

        return self

    def _get_roas(self) -> tuple[float, float, float]:
        if self.results is None:
            raise ValueError("results must not be None")
        incr = self.results["incrementality"]
        roas = self._safe_ratio(incr * self.msrp, self.spend)
        lo = self._safe_ratio(self.results["incrementality_ci_lower"] * self.msrp, self.spend)
        hi = self._safe_ratio(self.results["incrementality_ci_upper"] * self.msrp, self.spend)
        return roas, lo, hi

    def summarize(self, lift: str = "incremental") -> None:
        """Print a tabulated summary of the switchback results.

        Parameters
        ----------
        lift : str
            The kind of lift to report. One of ``"absolute"``, ``"relative"``,
            ``"incremental"``, ``"cost-per"``, ``"revenue"`` or ``"roas"``.
        """
        if self.results is None:
            raise ValueError("Call generate() before summarize()")
        lift = self._validate_lift(lift)
        ci_alpha = self._get_ci_print()
        lo_key, hi_key = f"{ci_alpha} Lower CI", f"{ci_alpha} Upper CI"

        point = self.results["lift"]
        ci_lo = self.results["lift_ci_lower"]
        ci_hi = self.results["lift_ci_upper"]
        incr = self.results["incrementality"]
        incr_lo = self.results["incrementality_ci_lower"]
        incr_hi = self.results["incrementality_ci_upper"]

        table_dict: dict[str, list[Any]] = {
            "Design": ["Switchback"],
            "Treated Obs": [self.results["n_treated_obs"]],
        }

        if lift in ("incremental", "absolute"):
            table_dict["Metric"] = [self.y_variable]
            table_dict["Lift Type"] = ["Incremental" if lift == "incremental" else "Absolute"]
            if lift == "incremental":
                cells = self._format_lift_cells(lift, incr, incr_lo, incr_hi)
            else:
                cells = self._format_lift_cells(lift, point, ci_lo, ci_hi)
        elif lift == "relative":
            table_dict["Metric"] = [self.y_variable]
            table_dict["Lift Type"] = ["Relative"]
            cells = self._format_lift_cells(lift, incr, incr_lo, incr_hi, relative_divisor=incr)
        elif lift == "revenue":
            table_dict["Metric"] = ["Revenue"]
            table_dict["Lift Type"] = ["Incremental"]
            rev = incr * self.msrp
            cells = self._format_lift_cells(lift, rev, incr_lo * self.msrp, incr_hi * self.msrp)
        elif lift == "roas":
            table_dict["Metric"] = ["ROAS"]
            table_dict["Lift Type"] = ["Incremental"]
            rev = incr * self.msrp
            roas = self._safe_ratio(rev, self.spend)
            roas_lo = self._safe_ratio(incr_lo * self.msrp, self.spend)
            roas_hi = self._safe_ratio(incr_hi * self.msrp, self.spend)
            cells = self._format_lift_cells(lift, roas, roas_lo, roas_hi)
        else:
            table_dict["Metric"] = ["Cost-per"]
            table_dict["Lift Type"] = ["Incremental"]
            from math import ceil as mceil

            cost = self._safe_ratio(self.spend, mceil(incr)) if incr > 0 else float(np.inf)
            cost_lo = self._safe_ratio(self.spend, mceil(incr_hi)) if incr_hi > 0 else float(np.inf)
            cost_hi = self._safe_ratio(self.spend, mceil(incr_lo)) if incr_lo > 0 else float(np.inf)
            cells = self._format_lift_cells(lift, cost, cost_lo, cost_hi)

        table_dict["Lift"], table_dict[lo_key], table_dict[hi_key] = cells
        table_dict["p_value"] = [self.results["p_value"]]
        print(tabulate(table_dict, headers="keys", tablefmt="grid"))

        if self.carryover_lags > 0 and "carryover" in self.results:
            print("\nCarryover effects:")
            carry_dict: dict[str, list[Any]] = {
                "Lag": [],
                "Estimate": [],
                lo_key: [],
                hi_key: [],
                "p_value": [],
            }
            for lag in range(1, self.carryover_lags + 1):
                key = f"treatment_lag_{lag}"
                c = self.results["carryover"][key]
                carry_dict["Lag"].append(lag)
                carry_dict["Estimate"].append(f"{c['estimate']:,.4f}")
                carry_dict[lo_key].append(f"{c['ci_lower']:,.4f}")
                carry_dict[hi_key].append(f"{c['ci_upper']:,.4f}")
                carry_dict["p_value"].append(f"{c['p_value']:.6f}")
            print(tabulate(carry_dict, headers="keys", tablefmt="grid"))

    def permutation_test(
        self,
        n_permutations: int = 1000,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Run a randomization inference test by permuting the switchback schedule.

        For each permutation, the treatment assignment within each geo is
        circularly shifted by a random offset, preserving the autocorrelation
        structure of the schedule. The two-way FE model is refit and its
        treatment coefficient collected. The empirical p-value is the fraction
        of permuted coefficients at least as extreme as the observed one.

        Parameters
        ----------
        n_permutations : int, default=1000
            Number of random schedule permutations.
        seed : int, default=0
            Random seed for reproducibility.

        Returns
        -------
        dict
            ``observed_effect``, ``permutation_effects`` (array),
            ``empirical_p_value``, ``n_permutations``.
        """
        if self.results is None:
            raise ValueError("Call generate() before permutation_test()")
        if self.panel is None:
            raise ValueError("Panel data not available")

        rng = np.random.default_rng(seed)
        observed = self.results["lift"]
        panel_pd = self.panel.to_pandas()
        geos = panel_pd[self.geo_variable].unique()
        treat_var: str = self.treatment_variable or "is_treatment"

        exog_terms: list[str] = [treat_var]
        for lag in range(1, self.carryover_lags + 1):
            exog_terms.append(f"treatment_lag_{lag}")
        exog_formula = " + ".join(exog_terms)
        formula = f"{self.y_variable} ~ {exog_formula} + EntityEffects + TimeEffects"

        perm_effects: list[float] = []
        for _ in range(n_permutations):
            perm_df = panel_pd.copy()
            for geo in geos:
                mask = perm_df[self.geo_variable] == geo
                treat_vals = perm_df.loc[mask, treat_var].values
                shift = rng.integers(1, len(treat_vals))
                perm_df.loc[mask, treat_var] = np.roll(treat_vals, shift)
                for lag in range(1, self.carryover_lags + 1):
                    perm_df.loc[mask, f"treatment_lag_{lag}"] = np.roll(perm_df.loc[mask, treat_var].values, lag)
            perm_indexed = perm_df.set_index([self.geo_variable, self.date_variable])
            try:
                perm_model = PanelOLS.from_formula(formula, data=perm_indexed)
                perm_fit = perm_model.fit(cov_type="clustered", cluster_entity=True)
                perm_effects.append(float(perm_fit.params[treat_var]))
            except Exception:
                continue

        perm_arr = np.asarray(perm_effects, dtype=float)
        n_extreme = np.sum(np.abs(perm_arr) >= abs(observed))
        empirical_p = float((n_extreme + 1) / (len(perm_arr) + 1))

        result = {
            "observed_effect": observed,
            "permutation_effects": perm_arr,
            "empirical_p_value": empirical_p,
            "n_permutations": len(perm_arr),
        }
        self.results["permutation_test"] = result
        return result

    # ------------------------------------------------------------------
    # Power analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_autocorrelation(residuals_by_geo: list[np.ndarray]) -> float:
        """Estimate pooled AR(1) autocorrelation from per-geo residual series."""
        num = 0.0
        den = 0.0
        for r in residuals_by_geo:
            if len(r) < 2:
                continue
            r_dm = r - r.mean()
            num += float(np.dot(r_dm[1:], r_dm[:-1]))
            den += float(np.dot(r_dm, r_dm))
        return num / den if den > 0 else 0.0

    def power_analytical(
        self,
        mde: float | None = None,
        n_geos: int | None = None,
        n_periods: int | None = None,
        sigma: float | None = None,
        rho: float | None = None,
        treatment_fraction: float = 0.5,
        power_target: float = 0.8,
    ) -> dict[str, float]:
        """Analytical power calculation for the switchback design.

        When ``mde`` is provided, returns the statistical power for that effect
        size. When ``mde`` is ``None``, inverts the formula to return the
        minimum detectable effect at ``power_target``.

        If ``sigma`` and ``rho`` (residual standard deviation and AR(1)
        autocorrelation) are not supplied, they are estimated from the fitted
        model's residuals. Similarly, ``n_geos`` and ``n_periods`` default to
        the values in the current panel.

        Parameters
        ----------
        mde : float or None
            The effect size to evaluate. If ``None``, the MDE at
            ``power_target`` power is returned instead.
        n_geos : int or None
            Number of geos. Defaults to the current data.
        n_periods : int or None
            Number of time periods per geo (after washout). Defaults to the
            current data.
        sigma : float or None
            Residual standard deviation. Estimated from the model if not given.
        rho : float or None
            Within-geo AR(1) autocorrelation. Estimated from the model if not
            given.
        treatment_fraction : float, default=0.5
            Proportion of periods each geo spends in treatment.
        power_target : float, default=0.8
            Target power when computing the MDE (ignored when ``mde`` is
            given).

        Returns
        -------
        dict
            ``mde``, ``power``, ``sigma``, ``rho``, ``n_geos``,
            ``n_periods``, ``se_tau``.
        """
        from scipy.stats import norm

        if self.panel is None or self.model is None:
            raise ValueError("Call pre_process().generate() before power_analytical()")

        panel_pd = self.panel.to_pandas()
        geos = panel_pd[self.geo_variable].unique()

        if n_geos is None:
            n_geos = len(geos)
        if n_periods is None:
            n_periods = int(panel_pd.groupby(self.geo_variable).size().median())

        if sigma is None:
            sigma = float(np.sqrt(self.model.resids.var()))
        if rho is None:
            resids = self.model.resids
            residuals_by_geo: list[np.ndarray] = []
            for geo in geos:
                geo_resids = resids.xs(geo, level=self.geo_variable)
                residuals_by_geo.append(geo_resids.values)
            rho = self._estimate_autocorrelation(residuals_by_geo)

        p = treatment_fraction
        effective_n = n_geos * n_periods * p * (1 - p)
        inflation = 1 + (n_periods - 1) * rho
        var_tau = sigma**2 * inflation / effective_n
        se_tau = float(np.sqrt(var_tau))

        z_alpha = float(norm.ppf(1 - self.alpha / 2))

        if mde is not None:
            z_power = (abs(mde) / se_tau) - z_alpha
            power = float(norm.cdf(z_power))
        else:
            z_beta = float(norm.ppf(power_target))
            mde = (z_alpha + z_beta) * se_tau
            power = power_target

        return {
            "mde": mde,
            "power": power,
            "sigma": sigma,
            "rho": rho,
            "n_geos": n_geos,
            "n_periods": n_periods,
            "se_tau": se_tau,
        }

    def power_simulation(
        self,
        mde: float,
        n_simulations: int = 500,
        n_geos: int | None = None,
        n_periods: int | None = None,
        sigma: float | None = None,
        rho: float | None = None,
        treatment_fraction: float = 0.5,
        switching_freq: int = 7,
        seed: int = 0,
        n_jobs: int = 1,
    ) -> dict[str, Any]:
        """Simulation-based power calculation for the switchback design.

        Generates synthetic switchback panels under a given effect size, fits
        the two-way FE model on each, and counts how often the null is
        rejected at level ``alpha``.

        Parameters
        ----------
        mde : float
            The effect size (treatment coefficient) to simulate.
        n_simulations : int, default=500
            Number of Monte Carlo replications.
        n_geos : int or None
            Number of geos. Defaults to the current data.
        n_periods : int or None
            Number of time periods per geo. Defaults to the current data.
        sigma : float or None
            Residual standard deviation. Estimated from the model if not given.
        rho : float or None
            Within-geo AR(1) autocorrelation. Estimated from the model if not
            given.
        treatment_fraction : float, default=0.5
            Approximate proportion of periods in treatment.
        switching_freq : int, default=7
            Number of consecutive periods in each on/off window.
        seed : int, default=0
            Random seed for reproducibility.
        n_jobs : int, default 1
            Number of parallel workers for the simulation loop. ``-1`` uses
            all available cores.  ``1`` (the default) runs sequentially with
            no parallelism overhead.

        Returns
        -------
        dict
            ``mde``, ``power``, ``n_simulations``, ``n_rejections``,
            ``sigma``, ``rho``, ``power_ci_lower``, ``power_ci_upper``.
        """
        if self.panel is None or self.model is None:
            raise ValueError("Call pre_process().generate() before power_simulation()")

        panel_pd = self.panel.to_pandas()
        geos_arr = panel_pd[self.geo_variable].unique()
        treat_var: str = self.treatment_variable or "is_treatment"

        if n_geos is None:
            n_geos = len(geos_arr)
        if n_periods is None:
            n_periods = int(panel_pd.groupby(self.geo_variable).size().median())

        if sigma is None:
            sigma = float(np.sqrt(self.model.resids.var()))
        if rho is None:
            resids = self.model.resids
            residuals_by_geo: list[np.ndarray] = []
            for geo in geos_arr:
                geo_resids = resids.xs(geo, level=self.geo_variable)
                residuals_by_geo.append(geo_resids.values)
            rho = self._estimate_autocorrelation(residuals_by_geo)

        import pandas as pd

        alpha_val = self.alpha
        ss = np.random.SeedSequence(seed)
        sim_seeds = ss.generate_state(n_simulations)

        def _run_sim(sim_seed: int) -> bool:
            sim_rng = np.random.default_rng(sim_seed)
            rows = []
            for g in range(n_geos):
                geo_fe = sim_rng.normal(0, 10)
                phase = sim_rng.integers(0, 2)
                noise = np.zeros(n_periods)
                noise[0] = sim_rng.normal(0, sigma)
                for t in range(1, n_periods):
                    noise[t] = rho * noise[t - 1] + sim_rng.normal(0, sigma * np.sqrt(1 - rho**2))
                for t in range(n_periods):
                    time_fe = 2.0 * np.sin(2 * np.pi * t / 30)
                    treat = int((t // switching_freq + phase) % 2)
                    y = geo_fe + time_fe + mde * treat + noise[t]
                    rows.append({"geo": f"g{g}", "date": t, treat_var: treat, "y": y})
            sim_df = pd.DataFrame(rows)
            sim_indexed = sim_df.set_index(["geo", "date"])
            formula = f"y ~ {treat_var} + EntityEffects + TimeEffects"
            try:
                sim_model = PanelOLS.from_formula(formula, data=sim_indexed)
                sim_fit = sim_model.fit(cov_type="clustered", cluster_entity=True)
                return float(sim_fit.pvalues[treat_var]) < alpha_val
            except Exception:
                return False

        if n_jobs == 1:
            outcomes = [_run_sim(int(s)) for s in sim_seeds]
        else:
            from joblib import Parallel, delayed

            outcomes = Parallel(n_jobs=n_jobs)(delayed(_run_sim)(int(s)) for s in sim_seeds)

        n_rejections = sum(outcomes)
        power = n_rejections / n_simulations
        from scipy.stats import norm

        z = float(norm.ppf(0.975))
        se_power = np.sqrt(power * (1 - power) / n_simulations)
        return {
            "mde": mde,
            "power": power,
            "n_simulations": n_simulations,
            "n_rejections": n_rejections,
            "sigma": sigma,
            "rho": rho,
            "power_ci_lower": max(0.0, power - z * se_power),
            "power_ci_upper": min(1.0, power + z * se_power),
        }

    def plot(self) -> None:
        """Plot the treatment schedule and outcome time series per geo.

        Returns
        -------
        An interactive Plotly figure showing the switchback pattern.
        """
        if self.panel is None:
            raise ValueError("Call pre_process() before plot()")

        import pandas as pd

        panel_pd: pd.DataFrame = self.panel.to_pandas()
        geos = sorted(panel_pd[self.geo_variable].unique())
        treat_var: str = self.treatment_variable or "is_treatment"

        fig = go.Figure()
        for geo in geos:
            geo_data = panel_pd.loc[panel_pd[self.geo_variable] == geo].sort_values(by=self.date_variable)
            dates = geo_data[self.date_variable]
            y = geo_data[self.y_variable]
            treat = geo_data[treat_var]
            colors = ["red" if t == 1 else "blue" for t in treat]
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=y,
                    mode="lines+markers",
                    marker={"color": colors, "size": 4},
                    name=str(geo),
                )
            )

        fig.update_layout(
            title="Switchback Design: Outcome by Geo",
            xaxis_title=self.date_variable,
            yaxis_title=self.y_variable,
        )
        fig.show()
