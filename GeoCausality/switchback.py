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

        if self.washout > 0:
            treat_col = nw.col(self.treatment_variable)
            shifted = treat_col.shift(1).over(self.geo_variable)
            switch = (treat_col != shifted) & ~shifted.is_null()
            df = df.with_columns(switch.alias("_switch"))
            washout_mask = nw.lit(False)
            for lag in range(self.washout):
                washout_mask = washout_mask | nw.col("_switch").shift(lag).over(self.geo_variable).fill_null(False)
            df = df.filter(~washout_mask).drop("_switch")

        if self.carryover_lags > 0:
            for lag in range(1, self.carryover_lags + 1):
                lag_col = nw.col(self.treatment_variable).shift(lag).over(self.geo_variable).fill_null(0)
                df = df.with_columns(lag_col.alias(f"treatment_lag_{lag}"))

        self.panel = df
        self.n_treated_obs = int(df.filter(nw.col(self.treatment_variable) == 1).shape[0])
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

        exog_terms = [self.treatment_variable]
        for lag in range(1, self.carryover_lags + 1):
            exog_terms.append(f"treatment_lag_{lag}")
        exog_formula = " + ".join(exog_terms)
        formula = f"{self.y_variable} ~ {exog_formula} + EntityEffects + TimeEffects"

        panel_pd = self.panel.to_pandas().set_index([self.geo_variable, self.date_variable])
        model = PanelOLS.from_formula(formula, data=panel_pd)
        self.model = model.fit(cov_type="clustered", cluster_entity=True)

        lift = float(self.model.params[self.treatment_variable])
        cis = self.model.conf_int(1 - self.alpha)
        ci_lower = float(cis.loc[self.treatment_variable, "lower"])
        ci_upper = float(cis.loc[self.treatment_variable, "upper"])
        p_value = float(self.model.pvalues[self.treatment_variable])

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

        exog_terms = [self.treatment_variable]
        for lag in range(1, self.carryover_lags + 1):
            exog_terms.append(f"treatment_lag_{lag}")
        exog_formula = " + ".join(exog_terms)
        formula = f"{self.y_variable} ~ {exog_formula} + EntityEffects + TimeEffects"

        perm_effects: list[float] = []
        for _ in range(n_permutations):
            perm_df = panel_pd.copy()
            for geo in geos:
                mask = perm_df[self.geo_variable] == geo
                treat_vals = perm_df.loc[mask, self.treatment_variable].values
                shift = rng.integers(1, len(treat_vals))
                perm_df.loc[mask, self.treatment_variable] = np.roll(treat_vals, shift)
                for lag in range(1, self.carryover_lags + 1):
                    perm_df.loc[mask, f"treatment_lag_{lag}"] = np.roll(
                        perm_df.loc[mask, self.treatment_variable].values, lag
                    )
            perm_indexed = perm_df.set_index([self.geo_variable, self.date_variable])
            try:
                perm_model = PanelOLS.from_formula(formula, data=perm_indexed)
                perm_fit = perm_model.fit(cov_type="clustered", cluster_entity=True)
                perm_effects.append(float(perm_fit.params[self.treatment_variable]))
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

    def plot(self) -> None:
        """Plot the treatment schedule and outcome time series per geo.

        Returns
        -------
        An interactive Plotly figure showing the switchback pattern.
        """
        if self.panel is None:
            raise ValueError("Call pre_process() before plot()")

        panel_pd = self.panel.to_pandas()
        geos = sorted(panel_pd[self.geo_variable].unique())

        fig = go.Figure()
        for geo in geos:
            geo_data = panel_pd[panel_pd[self.geo_variable] == geo].sort_values(self.date_variable)
            dates = geo_data[self.date_variable]
            y = geo_data[self.y_variable]
            treat = geo_data[self.treatment_variable]
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
