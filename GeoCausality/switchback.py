"""Switchback (crossover) experiment estimator for geo-experiment causal inference."""

from typing import Any

import narwhals as nw
import numpy as np
import plotly.graph_objects as go
from linearmodels.panel import PanelOLS
from narwhals.typing import IntoDataFrame
from tabulate import tabulate  # type: ignore

from GeoCausality._base import Estimator
from GeoCausality._plotting import apply_theme, get_theme


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
    each geo's schedule is circularly shifted at random and the empirical null
    distribution of test statistics is compared to the observed statistic.
    This is exact only if the real assignment mechanism was a uniformly random
    circular shift of each geo's schedule; for other designs it is an
    approximation.

    Notes
    -----
    The estimator is two-way fixed-effects regression, not the Horvitz-Thompson
    estimator of Bojinov, Simchi-Levi & Zhao (2023), "Design and Analysis of
    Switchback Experiments": it does not weight by exposure probabilities and is
    biased under carryover beyond the modelled lags. The design motivation
    follows Bojinov & Shephard (2019), "Time Series Experiments and Causal
    Estimands: Exact Randomization Tests and Trading", and Xiong et al. (2019),
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
        self._exposure_counts: dict[str, float] = {}
        self._exposed_outcome: float = 0.0

    def pre_process(self) -> "Switchback":
        """Prepare the panel data: add washout flags and carryover lags.

        Returns
        -------
        Switchback
            Itself, so it can be chained with ``generate()``.
        """
        df = self.data.sort([self.geo_variable, self.date_variable])
        treat_var: str = self.treatment_variable or "is_treatment"
        full = self._add_lags(df)
        self.panel = self._apply_washout(full)
        # Exposure counts over the full schedule (washed-out periods were still
        # treated), used to turn the per-period coefficients into a total effect.
        self.n_treated_obs = int(full.filter(nw.col(treat_var) == 1).shape[0])
        self._exposure_counts = {term: float(full[term].sum()) for term in [treat_var, *self._lag_terms()]}
        exposed = nw.col(treat_var) == 1
        for term in self._lag_terms():
            exposed = exposed | (nw.col(term) == 1)
        self._exposed_outcome = float(full.filter(exposed)[self.y_variable].sum())
        return self

    def _lag_terms(self) -> list[str]:
        """Names of the lagged-treatment regressors."""
        return [f"treatment_lag_{lag}" for lag in range(1, self.carryover_lags + 1)]

    def _add_lags(self, df: nw.DataFrame) -> nw.DataFrame:
        """Add lagged treatment indicators, computed on the full (pre-washout) schedule.

        Lags must be built before any rows are dropped: after a washout filter,
        ``shift`` would read the previous *kept* row rather than period ``t - k``.

        Parameters
        ----------
        df : narwhals DataFrame
            The panel sorted by geo and date.

        Returns
        -------
        The panel with ``treatment_lag_k`` columns (0 before a geo's first period).
        """
        treat_var: str = self.treatment_variable or "is_treatment"
        for lag in range(1, self.carryover_lags + 1):
            lag_col = nw.col(treat_var).shift(lag).over(self.geo_variable).fill_null(0)
            df = df.with_columns(lag_col.alias(f"treatment_lag_{lag}"))
        return df

    def _apply_washout(self, df: nw.DataFrame) -> nw.DataFrame:
        """Drop the ``washout`` periods following every treatment switch within a geo.

        Parameters
        ----------
        df : narwhals DataFrame
            The panel sorted by geo and date.

        Returns
        -------
        The panel without the washout periods.
        """
        if self.washout <= 0:
            return df
        treat_var: str = self.treatment_variable or "is_treatment"
        treat_col = nw.col(treat_var)
        shifted = treat_col.shift(1).over(self.geo_variable)
        switch = (treat_col != shifted) & ~shifted.is_null()
        df = df.with_columns(switch.alias("_switch"))
        washout_mask = nw.lit(False)
        for lag in range(self.washout):
            washout_mask = washout_mask | nw.col("_switch").shift(lag).over(self.geo_variable).fill_null(False)
        return df.filter(~washout_mask).drop("_switch")

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

        # Total incrementality is the effect summed over every exposed period of
        # the full schedule (washed-out periods were still treated): the
        # contemporaneous effect times the treated periods plus each carryover
        # effect times the periods it applies to. Its interval uses the same
        # critical value as the coefficient interval.
        terms = [treat_var, *self._lag_terms()]
        counts = np.array([self._exposure_counts[term] for term in terms])
        coefs = np.array([float(self.model.params[term]) for term in terms])
        cov = np.asarray(self.model.cov.loc[terms, terms], dtype=float)
        incrementality = float(counts @ coefs)
        incr_se = float(np.sqrt(max(counts @ cov @ counts, 0.0)))
        lift_se = float(self.model.std_errors[treat_var])
        crit = (ci_upper - lift) / lift_se if lift_se > 0 else 0.0

        self.results = {
            "lift": lift,
            "lift_ci_lower": ci_lower,
            "lift_ci_upper": ci_upper,
            "incrementality": incrementality,
            "incrementality_ci_lower": incrementality - crit * incr_se,
            "incrementality_ci_upper": incrementality + crit * incr_se,
            "baseline": self._exposed_outcome - incrementality,
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
            # Relative to the untreated baseline of the exposed periods, not to
            # the incrementality itself (which would always read 100%).
            cells = self._format_lift_cells(lift, incr, incr_lo, incr_hi, relative_divisor=self.results["baseline"])
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
        structure of the schedule. The p-value is exact only when the real
        schedule was assigned by such a random shift; otherwise it approximates
        the design's randomization distribution. The two-way FE model is refit and its
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

        import pandas as pd

        rng = np.random.default_rng(seed)
        observed = self.results["lift"]
        # Permute the full (pre-washout) schedule, then rebuild lags and re-apply
        # the washout, so each permuted panel is processed exactly like the real
        # one (no lag values wrapped around from the end of the series).
        full_pd = self.data.sort([self.geo_variable, self.date_variable]).to_pandas()
        treat_var: str = self.treatment_variable or "is_treatment"

        exog_terms: list[str] = [treat_var]
        for lag in range(1, self.carryover_lags + 1):
            exog_terms.append(f"treatment_lag_{lag}")
        exog_formula = " + ".join(exog_terms)
        formula = f"{self.y_variable} ~ {exog_formula} + EntityEffects + TimeEffects"

        # Rows are sorted by geo, so each geo is a contiguous block, in the
        # order of ``unique()``.
        geo_codes = pd.factorize(full_pd[self.geo_variable], use_na_sentinel=False)[0]
        lengths = np.bincount(geo_codes)
        block_start = np.repeat(np.cumsum(lengths) - lengths, lengths)
        position = np.arange(len(full_pd)) - block_start
        treat_vals = full_pd[treat_var].to_numpy()
        fast_inputs = self._fast_refit_inputs(full_pd, treat_var)

        perm_effects: list[float] = []
        for _ in range(n_permutations):
            # One shift per geo, drawn in the same order as rolling each geo in
            # turn; row i of a geo of length n takes row (i - shift) mod n.
            shifts = rng.integers(1, lengths)
            source = block_start + (position - shifts[geo_codes]) % lengths[geo_codes]
            perm_treat = treat_vals[source]
            effect = None
            if fast_inputs is not None:
                y, entity, time = fast_inputs
                x, keep = self._schedule_design(perm_treat.astype(float), position)
                effect = self._twfe_treatment_coefficient(y[keep], x[keep], entity[keep], time[keep])
            if effect is None:
                perm_df = full_pd.copy()
                perm_df[treat_var] = perm_treat
                try:
                    effect = self._panelols_permutation_effect(perm_df, formula)
                except Exception:
                    continue
            perm_effects.append(effect)

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

    def _panelols_permutation_effect(self, perm_df: Any, formula: str) -> float:
        """Refit the two-way FE model with ``PanelOLS`` on a permuted schedule.

        This is the reference path for ``permutation_test``: it is used whenever
        the closed-form refit in ``_twfe_treatment_coefficient`` declines a
        permutation, and it raises if ``PanelOLS`` cannot fit the panel.

        Parameters
        ----------
        perm_df : pandas DataFrame
            The full panel, sorted by geo and date, with the permuted schedule.
        formula : str
            The ``PanelOLS`` formula used by ``generate()``.

        Returns
        -------
        The treatment coefficient.
        """
        treat_var: str = self.treatment_variable or "is_treatment"
        perm_panel = self._apply_washout(self._add_lags(nw.from_native(perm_df, eager_only=True)))
        perm_indexed = perm_panel.to_pandas().set_index([self.geo_variable, self.date_variable])
        perm_fit = PanelOLS.from_formula(formula, data=perm_indexed).fit(cov_type="clustered", cluster_entity=True)
        return float(perm_fit.params[treat_var])

    def _fast_refit_inputs(self, full_pd: Any, treat_var: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Outcome and integer geo/date codes for the closed-form permutation refit.

        Parameters
        ----------
        full_pd : pandas DataFrame
            The full panel, sorted by geo and date.
        treat_var : str
            The treatment column.

        Returns
        -------
        ``(y, geo_codes, date_codes)``, or ``None`` when the panel has missing,
        non-finite or non-numeric values or duplicate geo-date rows, which
        ``PanelOLS`` handles itself (by dropping rows or failing).
        """
        import pandas as pd

        keys = [self.geo_variable, self.date_variable]
        y_col, treat_col = full_pd[self.y_variable], full_pd[treat_var]
        for col in (y_col, treat_col):
            if not pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
                return None
        y = y_col.to_numpy(dtype=float)
        if not (np.isfinite(y).all() and np.isfinite(treat_col.to_numpy(dtype=float)).all()):
            return None
        if full_pd[keys].isna().any().any() or full_pd.duplicated(keys).any():
            return None
        geo_codes = pd.factorize(full_pd[self.geo_variable])[0]
        date_codes = pd.factorize(full_pd[self.date_variable])[0]
        return y, geo_codes, date_codes

    def _schedule_design(self, treat: np.ndarray, position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Treatment and lag regressors plus the washout keep-mask, in NumPy.

        Mirrors ``_add_lags`` followed by ``_apply_washout`` on a panel sorted by
        geo and date.

        Parameters
        ----------
        treat : numpy array
            The treatment schedule, one value per row.
        position : numpy array of int
            Each row's period index within its geo (0 for a geo's first row).

        Returns
        -------
        ``(x, keep)``: the ``(n_rows, 1 + carryover_lags)`` regressors
        ``[treatment, treatment_lag_1, ...]`` and the rows kept after washout.
        """

        def shifted(values: np.ndarray, lag: int, fill: Any) -> np.ndarray:
            out = np.full_like(values, fill)
            out[lag:] = values[: max(len(values) - lag, 0)]
            out[position < lag] = fill
            return out

        x = np.column_stack([treat] + [shifted(treat, lag, 0.0) for lag in range(1, self.carryover_lags + 1)])
        keep = np.ones(len(treat), dtype=bool)
        if self.washout > 0:
            switch = treat != shifted(treat, 1, np.nan)
            switch[position == 0] = False
            for lag in range(self.washout):
                keep &= ~shifted(switch, lag, False)
        return x, keep

    @staticmethod
    def _twfe_treatment_coefficient(
        y: np.ndarray,
        x: np.ndarray,
        entity: np.ndarray,
        time: np.ndarray,
    ) -> float | None:
        """First coefficient of ``y ~ x + EntityEffects + TimeEffects``, without ``PanelOLS``.

        By Frisch-Waugh-Lovell the slope coefficients are those of the OLS of
        ``y`` on ``x`` once both are residualised on the two sets of fixed
        effects. As in ``PanelOLS``, the effect with more levels is swept out
        exactly by group demeaning; the other enters as dummies (first level
        dropped), partialled out through their small normal equations. This is
        exact on unbalanced panels and costs one ``(levels x levels)``
        Cholesky factorisation per call.

        Parameters
        ----------
        y : numpy array
            Outcome, one value per row.
        x : numpy array
            ``(n_rows, k)`` regressors; the coefficient on column 0 is returned.
        entity, time : numpy array of int
            Non-negative integer codes of each row's geo and date.

        Returns
        -------
        The coefficient, or ``None`` when ``PanelOLS`` should decide instead: a
        regressor (nearly) absorbed by the effects or collinear with the others,
        disconnected effects, too few observations, or more than 2,000 levels
        in both effects.
        """
        n_obs, n_x = x.shape
        codes: list[np.ndarray] = []
        counts: list[np.ndarray] = []
        for c in (entity, time):
            count = np.bincount(c)
            present = count > 0
            codes.append((np.cumsum(present) - 1)[c])
            counts.append(count[present].astype(float))
        n_entity, n_time = len(counts[0]), len(counts[1])
        if min(n_entity, n_time) < 2 or n_obs - n_entity - n_time + 1 - n_x <= 0:
            return None
        # Demean by the larger effect (as ``PanelOLS`` does); dummies for the smaller.
        absorb, dummy = (0, 1) if n_entity > n_time else (1, 0)
        ca, cb = codes[absorb], codes[dummy]
        na, nb = counts[absorb], counts[dummy]
        if len(nb) > 2000 or len(na) * len(nb) > 5e7:
            return None

        z = np.column_stack([y, x])
        for j in range(z.shape[1]):
            z[:, j] -= (np.bincount(ca, weights=z[:, j]) / na)[ca]
        # Dummies for levels 1.. of the smaller effect, demeaned within the
        # larger: their cross-products with z are per-level sums of z, and their
        # Gram matrix is diag(nb) - B' diag(1 / na) B for the incidence matrix B.
        cross = np.column_stack([np.bincount(cb, weights=z[:, j], minlength=len(nb)) for j in range(z.shape[1])])[1:]
        incidence = np.bincount(ca * len(nb) + cb, minlength=len(na) * len(nb)).reshape(len(na), len(nb))
        incidence = incidence.astype(float)
        gram = np.diag(nb) - incidence.T @ (incidence / na[:, None])
        gram = gram[1:, 1:]
        try:
            chol = np.linalg.cholesky(gram)
        except np.linalg.LinAlgError:
            return None
        # A pivot near zero means the effects are not connected (not identified).
        if np.any(np.diag(chol) ** 2 <= 1e-10 * np.diag(gram)):
            return None
        half = np.linalg.solve(chol, cross)
        moments = z.T @ z - half.T @ half

        resid_gram, resid_cross = moments[1:, 1:], moments[1:, 0]
        raw_ss = np.einsum("ij,ij->j", x, x)
        resid_ss = np.diag(resid_gram)
        if np.any(resid_ss <= 1e-8 * raw_ss):
            return None
        corr = resid_gram / np.sqrt(np.outer(resid_ss, resid_ss))
        if np.linalg.eigvalsh(corr)[0] <= 1e-8:
            return None
        return float(np.linalg.solve(resid_gram, resid_cross)[0])

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
        switching_freq: int = 7,
    ) -> dict[str, float]:
        """Analytical power calculation for the switchback design.

        When ``mde`` is provided, returns the statistical power for that effect
        size. When ``mde`` is ``None``, inverts the formula to return the
        minimum detectable effect at ``power_target``.

        The standard error is the exact sampling standard deviation of the
        two-way fixed-effects treatment coefficient, ``sqrt(d' Sigma d) / (d' d)``,
        where ``d`` is the two-way demeaned treatment schedule and ``Sigma`` is
        block-diagonal with an AR(1) correlation within each geo. The schedule
        mirrors ``power_simulation``: blocks of ``switching_freq`` treated
        periods alternate with control blocks (sized so each geo is treated for
        ``treatment_fraction`` of the periods), with alternate geos starting in
        opposite phases.

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
        switching_freq : int, default=7
            Number of consecutive periods in each treated window.

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

        se_tau = self._twfe_standard_error(n_geos, n_periods, sigma, rho, treatment_fraction, switching_freq)

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

    @staticmethod
    def _schedule(
        n_periods: int,
        treatment_fraction: float,
        switching_freq: int,
        phases: np.ndarray,
    ) -> np.ndarray:
        """Build block switchback schedules.

        Each geo alternates ``switching_freq`` treated periods with control
        blocks sized so the geo is treated for ``treatment_fraction`` of the
        time. A geo with phase 1 is shifted by one treated block (it starts in
        control).

        Parameters
        ----------
        n_periods : int
            Number of periods per geo.
        treatment_fraction : float
            Share of periods treated, strictly between 0 and 1.
        switching_freq : int
            Length of each treated block.
        phases : numpy array of int
            One phase (0 or 1) per geo.

        Returns
        -------
        A ``(n_geos, n_periods)`` 0/1 array.
        """
        if not 0.0 < treatment_fraction < 1.0:
            raise ValueError("treatment_fraction must be strictly between 0 and 1")
        on = max(int(switching_freq), 1)
        off = max(int(round(on * (1 - treatment_fraction) / treatment_fraction)), 1)
        t = np.arange(n_periods)
        shifts = np.asarray(phases, dtype=int)[:, None] * on
        return (((t[None, :] + shifts) % (on + off)) < on).astype(float)

    @staticmethod
    def _twfe_standard_error(
        n_geos: int,
        n_periods: int,
        sigma: float,
        rho: float,
        treatment_fraction: float,
        switching_freq: int,
    ) -> float:
        """Compute the standard error of the two-way FE treatment effect under AR(1) errors.

        Parameters
        ----------
        n_geos, n_periods : int
            Panel dimensions.
        sigma : float
            Marginal standard deviation of the errors.
        rho : float
            Within-geo AR(1) autocorrelation (clipped to (-1, 1)).
        treatment_fraction : float
            Share of periods each geo is treated.
        switching_freq : int
            Length of each treated window.

        Returns
        -------
        ``sqrt(d' Sigma d) / (d' d)`` for the two-way demeaned schedule ``d``.
        """
        from scipy.linalg import toeplitz

        # Alternate geos start in opposite phases, balancing each period.
        phases = np.arange(n_geos) % 2
        schedule = Switchback._schedule(n_periods, treatment_fraction, switching_freq, phases)
        demeaned = schedule - schedule.mean(axis=1, keepdims=True) - schedule.mean(axis=0, keepdims=True)
        demeaned += schedule.mean()
        rho = float(np.clip(rho, -0.99, 0.99))
        corr = toeplitz(rho ** np.arange(n_periods))
        ss = float(np.sum(demeaned**2))
        if ss == 0.0:
            return float("inf")
        quad = float(np.einsum("gt,ts,gs->", demeaned, corr, demeaned))
        return float(sigma * np.sqrt(max(quad, 0.0)) / ss)

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
            Proportion of periods each simulated geo spends in treatment.
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
            ``sigma``, ``rho``, ``power_ci_lower``, ``power_ci_upper`` (a
            ``1 - alpha`` Monte Carlo interval for the power).
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
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
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
        # Alternate geos start in opposite phases, as in ``power_analytical``; random
        # phases can give every geo the same schedule, which is collinear with
        # ``TimeEffects`` and makes the fit fail.
        schedule = self._schedule(n_periods, treatment_fraction, switching_freq, np.arange(n_geos) % 2)
        treat_int = schedule.astype(int)
        # Scalar ``np.sin`` per period: an array call may differ in the last bit.
        time_fe = np.array([2.0 * np.sin(2 * np.pi * t / 30) for t in range(n_periods)])
        geo_labels = np.repeat(np.array([f"g{g}" for g in range(n_geos)], dtype=object), n_periods)
        dates = np.tile(np.arange(n_periods), n_geos)
        innovation_sd = sigma * np.sqrt(1 - rho**2)

        def _run_sim(sim_seed: int) -> bool:
            sim_rng = np.random.default_rng(sim_seed)
            # Per geo: its effect, the first noise term, then the innovations,
            # drawn in that order. ``normal(0, s)`` is ``s`` times the next
            # standard normal, so one array of draws reproduces the scalar calls.
            draws = sim_rng.standard_normal((n_geos, n_periods + 1))
            geo_fe = 10 * draws[:, 0]
            noise = np.zeros((n_geos, n_periods))
            noise[:, 0] = sigma * draws[:, 1]
            for t in range(1, n_periods):
                noise[:, t] = rho * noise[:, t - 1] + innovation_sd * draws[:, t + 1]
            y = geo_fe[:, None] + time_fe[None, :] + mde * treat_int + noise
            sim_df = pd.DataFrame({"geo": geo_labels, "date": dates, treat_var: treat_int.ravel(), "y": y.ravel()})
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

        # Monte Carlo interval for the simulated power, at the estimator's level.
        z = float(norm.ppf(1 - alpha_val / 2))
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

    def plot(self, dark_mode: bool = False) -> None:
        """Plot the treatment schedule and outcome time series per geo.

        Parameters
        ----------
        dark_mode : bool, default=False
            Render with a dark background and lighter series colours.

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

        theme = get_theme(dark_mode)
        fig = go.Figure()
        for geo in geos:
            geo_data = panel_pd.loc[panel_pd[self.geo_variable] == geo].sort_values(by=self.date_variable)
            dates = geo_data[self.date_variable]
            y = geo_data[self.y_variable]
            treat = geo_data[treat_var]
            colors = [theme.counterfactual if t == 1 else theme.actual for t in treat]
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
        apply_theme(fig, theme)
        fig.show()
