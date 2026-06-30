"""Pre-experiment power analysis and Minimum Detectable Effect (MDE) for geo experiments.

This is the GeoCausality analog of GeoLift's ``GeoLiftPower``. Every estimator in
the package is a *post*-experiment tool: it measures the lift of an experiment
that has already run. ``PowerAnalysis`` answers the *pre*-experiment question --
"given this geo split and our historical data, what lift could we actually
detect, and for how long must we run?" -- by simulating placebo experiments on
clean pre-period history.

The mechanism, adapted to GeoCausality's conformal inference:

1. Restrict to the clean **pre-period history** (no real treatment effect).
2. For a grid of candidate **effect sizes** and **test durations**, repeatedly
   carve a placebo experiment out of the history: a sliding window of
   ``duration`` days becomes a fake post-period, every earlier day becomes the
   fake pre-period, and the candidate effect is injected into the test geos over
   the fake post-period.
3. Run the chosen estimator on each placebo experiment and record whether its
   conformal ``p_value`` clears ``alpha``.
4. **Power** is the fraction of placebos detected; the **MDE** is the smallest
   effect reaching a target power (default 0.8).

Running the grid at ``effect = 0`` also calibrates the false-positive rate
against ``alpha`` -- a free validation of the underlying inference.
"""

from typing import Any

import narwhals as nw
import numpy as np
import plotly.graph_objects as go
from narwhals.typing import IntoDataFrame
from tabulate import tabulate  # type: ignore

from GeoCausality._base import Estimator
from GeoCausality.augmented_synthetic_control import AugmentedSyntheticControl


class PowerAnalysis:
    """Simulate placebo experiments to estimate power and the Minimum Detectable Effect."""

    def __init__(
        self,
        data: IntoDataFrame,
        geo_variable: str = "geo",
        test_geos: list[str] | None = None,
        control_geos: list[str] | None = None,
        treatment_variable: str | None = "is_treatment",
        date_variable: str = "date",
        pre_period: str = "2021-01-01",
        y_variable: str = "y",
        alpha: float = 0.1,
        estimator: type[Estimator] = AugmentedSyntheticControl,
        estimator_kwargs: dict[str, Any] | None = None,
        injection: str = "multiplicative",
        min_pre_periods: int = 2,
        seed: int = 0,
    ) -> None:
        """Initialize the power-analysis simulator.

        Parameters
        ----------
        data : pandas or polars data frame
            Our geo-based time-series data. Only rows on or before ``pre_period``
            are used, so the simulation never sees a real treatment effect.
        geo_variable : str
            The name of the variable representing our geo-data.
        test_geos : list, optional
            The geos that would be assigned treatment. If not provided, derived
            from ``treatment_variable``.
        control_geos : list, optional
            The geos that would be withheld from treatment. If not provided,
            every geo that is not a test geo is treated as a control.
        treatment_variable : str, optional, default="is_treatment"
            If test and control geos are not provided, the column denoting which
            is test (coded 1) and control (coded 0).
        date_variable : str
            The name of the variable representing our dates.
        pre_period : str
            The last date of clean history. Every simulated experiment is carved
            from the dates on or before this value.
        y_variable : str
            The name of the variable representing the outcome metric.
        alpha : float, default=0.1
            The significance level. A placebo is "detected" when its conformal
            ``p_value`` is at most ``alpha``.
        estimator : type, default=AugmentedSyntheticControl
            The estimator class used to analyse each placebo experiment. Power is
            measured with the same method that will analyse the real experiment.
        estimator_kwargs : dict, optional
            Extra keyword arguments forwarded to ``estimator`` (e.g. ``sv_count``
            for ``RobustSyntheticControl``).
        injection : str, default="multiplicative"
            How the candidate effect is injected into the test geos.
            ``"multiplicative"`` scales the outcome by ``(1 + effect)``;
            ``"additive"`` adds ``effect`` to each post-period observation.
        min_pre_periods : int, default=2
            The minimum number of fake pre-period dates each placebo must retain
            so the estimator has something to fit.
        seed : int, default=0
            Seed for the placebo-window sampler, so results are reproducible.
        """
        self.data: nw.DataFrame = nw.from_native(data, eager_only=True)
        self.geo_variable = geo_variable
        self.date_variable = date_variable
        self.y_variable = y_variable
        self.pre_period = pre_period
        self.alpha = alpha
        self.estimator = estimator
        self.estimator_kwargs = estimator_kwargs or {}
        injection = injection.casefold()
        if injection not in ("multiplicative", "additive"):
            raise ValueError(f"injection must be 'multiplicative' or 'additive', got {injection!r}")
        self.injection = injection
        self.min_pre_periods = min_pre_periods
        self.seed = seed

        self.test_geos, self.control_geos = self._resolve_geos(test_geos, control_geos, treatment_variable)
        self.history: list[str] = self._clean_history()
        self.power_curve: list[dict[str, Any]] | None = None
        self.mde_table: list[dict[str, Any]] | None = None
        self.results: dict[str, Any] | None = None

    def _resolve_geos(
        self,
        test_geos: list[str] | None,
        control_geos: list[str] | None,
        treatment_variable: str | None,
    ) -> tuple[list[str], list[str]]:
        """Resolve test and control geos into explicit lists.

        Parameters
        ----------
        test_geos, control_geos : list, optional
            Explicit geo assignments, if provided.
        treatment_variable : str, optional
            Fallback column flagging test (1) vs. control (0) geos.

        Returns
        -------
        A tuple of (test_geos, control_geos) as concrete lists.
        """
        all_geos = self.data[self.geo_variable].unique().to_list()
        if test_geos is None:
            if treatment_variable is None:
                raise ValueError("Provide either test_geos or a treatment_variable")
            test_geos = self.data.filter(nw.col(treatment_variable) == 1)[self.geo_variable].unique().to_list()
        if not test_geos:
            raise ValueError("No test geos were identified")
        if control_geos is None:
            control_geos = [g for g in all_geos if g not in set(test_geos)]
        if not control_geos:
            raise ValueError("No control geos were identified")
        return list(test_geos), list(control_geos)

    def _clean_history(self) -> list[str]:
        """Return the sorted ISO-string dates on or before ``pre_period``.

        Returns
        -------
        The clean-history dates as lexicographically sorted strings, matching the
        backend-agnostic comparison the estimators use internally.
        """
        dates = (
            self.data.select(nw.col(self.date_variable).cast(nw.String).alias("d"))
            .unique(subset=["d"])
            .sort("d")["d"]
            .to_list()
        )
        return [d for d in dates if d <= self.pre_period]

    def _placebo_starts(self, duration: int, n_sims: int) -> np.ndarray:
        """Sample start indices for placebo post-windows of length ``duration``.

        A valid start leaves at least ``min_pre_periods`` fake pre-period dates
        before it and ``duration`` dates at or after it, all within the clean
        history. When more simulations are requested than there are distinct
        valid windows, windows are sampled with replacement.

        Parameters
        ----------
        duration : int
            The number of fake post-period dates.
        n_sims : int
            The number of placebo experiments to draw.

        Returns
        -------
        An array of ``n_sims`` start indices into ``self.history``.
        """
        h = len(self.history)
        last_start = h - duration
        if last_start < self.min_pre_periods:
            raise ValueError(
                f"Not enough clean history for duration={duration}: need "
                f"{self.min_pre_periods + duration} dates, have {h}"
            )
        valid = np.arange(self.min_pre_periods, last_start + 1)
        rng = np.random.default_rng(self.seed + duration)
        if n_sims <= valid.size:
            # Evenly spaced placements give broad, deterministic coverage.
            idx = np.linspace(0, valid.size - 1, n_sims).round().astype(int)
            return valid[idx]
        return rng.choice(valid, size=n_sims, replace=True)

    def _simulate_one(self, effect: float, duration: int, start_idx: int) -> bool:
        """Run a single placebo experiment and report whether it was detected.

        Parameters
        ----------
        effect : float
            The candidate effect injected into the test geos.
        duration : int
            The number of fake post-period dates.
        start_idx : int
            Index into ``self.history`` where the fake post-period begins.

        Returns
        -------
        True if the estimator's conformal ``p_value`` is at most ``alpha``.
        """
        post_dates = self.history[start_idx : start_idx + duration]
        pre_boundary = self.history[start_idx - 1]
        post_boundary = self.history[start_idx]
        last_post = post_dates[-1]

        date_str = nw.col(self.date_variable).cast(nw.String)
        in_post = nw.col(self.geo_variable).is_in(self.test_geos) & date_str.is_in(post_dates)
        if self.injection == "multiplicative":
            injected = nw.when(in_post).then(nw.col(self.y_variable) * (1.0 + effect))
        else:
            injected = nw.when(in_post).then(nw.col(self.y_variable) + effect)
        injected = injected.otherwise(nw.col(self.y_variable))

        sub = self.data.filter(date_str <= last_post).with_columns(injected.alias(self.y_variable))

        model = self.estimator(
            sub.to_native(),
            geo_variable=self.geo_variable,
            test_geos=self.test_geos,
            control_geos=self.control_geos,
            date_variable=self.date_variable,
            pre_period=pre_boundary,
            post_period=post_boundary,
            y_variable=self.y_variable,
            alpha=self.alpha,
            **self.estimator_kwargs,
        )
        model.pre_process().generate()
        if model.results is None:
            raise ValueError("model.results must not be None")
        return float(model.results["p_value"]) <= self.alpha

    def simulate(
        self,
        effect_sizes: list[float],
        durations: list[int],
        n_sims: int = 100,
    ) -> "PowerAnalysis":
        """Estimate power over a grid of effect sizes and test durations.

        Parameters
        ----------
        effect_sizes : list of float
            Candidate effects to inject. With multiplicative injection these are
            fractional lifts (e.g. ``0.10`` for +10%). Include ``0.0`` to
            calibrate the false-positive rate against ``alpha``.
        durations : list of int
            Candidate experiment lengths, in number of post-period dates.
        n_sims : int, default=100
            The number of placebo experiments per (effect, duration) cell.

        Returns
        -------
        PowerAnalysis
            Itself, so it can be chained with mde() / summarize() / plot().
        """
        curve: list[dict[str, Any]] = []
        for duration in durations:
            starts = self._placebo_starts(duration, n_sims)
            for effect in effect_sizes:
                detected = sum(self._simulate_one(effect, duration, int(s)) for s in starts)
                curve.append(
                    {
                        "effect": float(effect),
                        "duration": int(duration),
                        "n_sims": int(starts.size),
                        "n_detected": int(detected),
                        "power": detected / starts.size,
                    }
                )
        self.power_curve = curve
        self.results = {"power_curve": curve}
        return self

    def mde(self, target_power: float = 0.8) -> "PowerAnalysis":
        """Compute the Minimum Detectable Effect per duration at a target power.

        For each duration the MDE is the smallest effect whose power reaches
        ``target_power``, linearly interpolated between the two bracketing tested
        effects. When no tested effect reaches the target, the MDE is ``None``.

        Parameters
        ----------
        target_power : float, default=0.8
            The power level the MDE must achieve.

        Returns
        -------
        PowerAnalysis
            Itself, so it can be chained with summarize() / plot().
        """
        if self.power_curve is None:
            raise ValueError("Call simulate() before mde()")
        table: list[dict[str, Any]] = []
        durations = sorted({row["duration"] for row in self.power_curve})
        for duration in durations:
            points = sorted(
                ((r["effect"], r["power"]) for r in self.power_curve if r["duration"] == duration),
                key=lambda p: p[0],
            )
            table.append({"duration": duration, "mde": self._interpolate_mde(points, target_power)})
        self.mde_table = table
        if self.results is None:
            self.results = {}
        self.results["mde"] = table
        self.results["target_power"] = target_power
        return self

    @staticmethod
    def _interpolate_mde(points: list[tuple[float, float]], target_power: float) -> float | None:
        """Linearly interpolate the effect at which power first reaches the target.

        Parameters
        ----------
        points : list of (effect, power)
            Tested effects and their measured power, sorted by effect.
        target_power : float
            The power level to solve for.

        Returns
        -------
        The interpolated MDE, or None if no tested effect reaches the target.
        """
        prev: tuple[float, float] | None = None
        for effect, power in points:
            if power >= target_power:
                if prev is None or power == prev[1]:
                    return effect
                # Interpolate between the last sub-target point and this one.
                prev_e, prev_p = prev
                frac = (target_power - prev_p) / (power - prev_p)
                return prev_e + frac * (effect - prev_e)
            prev = (effect, power)
        return None

    def summarize(self) -> None:
        """Print the power curve and, if computed, the MDE table.

        Raises
        ------
        ValueError
            If ``simulate()`` has not been run yet.
        """
        if self.power_curve is None:
            raise ValueError("Call simulate() before summarize()")
        unit = "x" if self.injection == "multiplicative" else self.y_variable
        power_table = {
            "Duration": [r["duration"] for r in self.power_curve],
            f"Effect ({unit})": [r["effect"] for r in self.power_curve],
            "Sims": [r["n_sims"] for r in self.power_curve],
            "Detected": [r["n_detected"] for r in self.power_curve],
            "Power": [f"{r['power']:.0%}" for r in self.power_curve],
        }
        print(tabulate(power_table, headers="keys", tablefmt="grid"))

        if self.mde_table is not None:
            target = self.results.get("target_power", 0.8) if self.results else 0.8
            mde_table = {
                "Duration": [r["duration"] for r in self.mde_table],
                f"MDE @ {target:.0%} power": ["n/a" if r["mde"] is None else f"{r['mde']:.3f}" for r in self.mde_table],
            }
            print(tabulate(mde_table, headers="keys", tablefmt="grid"))

    def plot(self) -> None:
        """Plot power as a function of effect size, one line per duration.

        Raises
        ------
        ValueError
            If ``simulate()`` has not been run yet.
        """
        if self.power_curve is None:
            raise ValueError("Call simulate() before plot()")
        fig = go.Figure()
        durations = sorted({r["duration"] for r in self.power_curve})
        for duration in durations:
            points = sorted(
                ((r["effect"], r["power"]) for r in self.power_curve if r["duration"] == duration),
                key=lambda p: p[0],
            )
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in points],
                    y=[p[1] for p in points],
                    mode="lines+markers",
                    name=f"{duration} periods",
                )
            )
        target = self.results.get("target_power", 0.8) if self.results else 0.8
        fig.add_hline(y=target, line_dash="dash", annotation_text=f"target {target:.0%}")
        unit = "multiplicative" if self.injection == "multiplicative" else "additive"
        fig.update_layout(
            title="Power curve",
            xaxis_title=f"Effect size ({unit})",
            yaxis_title="Power",
            yaxis_range=[0, 1],
        )
        fig.show()
