"""Pre-experiment market (test-geo) selection for geo experiments.

This is the GeoCausality analog of GeoLift's ``GeoLiftMarketSelection``. Where
``PowerAnalysis`` answers "how detectable is *this* split?", ``MarketSelection``
answers the prior question: "*which* geos should we treat?" It enumerates
candidate test-geo sets, scores each one, and returns a ranked recommendation.

Each candidate is scored on two axes, both computed on clean pre-period history:

* **Power** -- via :class:`~GeoCausality.power.PowerAnalysis`, the fraction of
  placebo experiments at the target effect / duration that are detected. This is
  the scoring engine; market selection is a search loop around it.
* **Pre-period fit** -- how tightly the chosen estimator reconstructs the
  candidate test geos before any treatment, read from the estimator's
  split-conformal band (smaller is better). A candidate that the donors cannot
  reproduce pre-period will give untrustworthy lift no matter its power.

The two are normalised across candidates and combined into a single score
(``fit_weight`` trades them off), and the candidates are returned best-first.
"""

import warnings
from itertools import combinations
from math import comb
from typing import Any

import narwhals as nw
import numpy as np
import plotly.graph_objects as go
from narwhals.typing import IntoDataFrame
from tabulate import tabulate  # type: ignore

from GeoCausality._base import Estimator
from GeoCausality.augmented_synthetic_control import AugmentedSyntheticControl
from GeoCausality.power import PowerAnalysis


class MarketSelection:
    """Search candidate test-geo sets and rank them by power and pre-period fit."""

    def __init__(
        self,
        data: IntoDataFrame,
        geo_variable: str = "geo",
        date_variable: str = "date",
        pre_period: str = "2021-01-01",
        y_variable: str = "y",
        alpha: float = 0.1,
        estimator: type[Estimator] = AugmentedSyntheticControl,
        estimator_kwargs: dict[str, Any] | None = None,
        injection: str = "multiplicative",
        min_pre_periods: int = 2,
        fit_weight: float = 0.25,
        max_candidates: int = 200,
        seed: int = 0,
    ) -> None:
        """Initialize the market-selection search.

        Parameters
        ----------
        data : pandas or polars data frame
            Our geo-based time-series data. Only rows on or before ``pre_period``
            are used, so the search never sees a real treatment effect.
        geo_variable : str
            The name of the variable representing our geo-data.
        date_variable : str
            The name of the variable representing our dates.
        pre_period : str
            The last date of clean history each candidate is evaluated on.
        y_variable : str
            The name of the variable representing the outcome metric.
        alpha : float, default=0.1
            The significance level forwarded to the power simulation.
        estimator : type, default=AugmentedSyntheticControl
            The estimator class used to score each candidate. The pre-period fit
            metric requires an estimator that exposes a ``conformal_band`` (the
            synthetic-control family); for others, ranking falls back to power.
        estimator_kwargs : dict, optional
            Extra keyword arguments forwarded to ``estimator``.
        injection : str, default="multiplicative"
            How the candidate effect is injected (see ``PowerAnalysis``).
        min_pre_periods : int, default=2
            The minimum number of fake pre-period dates each placebo must retain.
        fit_weight : float, default=0.25
            Weight in ``[0, 1]`` on pre-period fit when combining it with power
            into the score. ``0`` ranks on power alone.
        max_candidates : int, default=200
            The most candidate test sets to evaluate. When the enumeration would
            exceed this, a seeded random sample is taken and a warning is issued.
        seed : int, default=0
            Seed for candidate sampling and the underlying power simulation.
        """
        self.data: nw.DataFrame = nw.from_native(data, eager_only=True)
        self.geo_variable = geo_variable
        self.date_variable = date_variable
        self.y_variable = y_variable
        self.pre_period = pre_period
        self.alpha = alpha
        self.estimator = estimator
        self.estimator_kwargs = estimator_kwargs or {}
        self.injection = injection
        self.min_pre_periods = min_pre_periods
        if not 0.0 <= fit_weight <= 1.0:
            raise ValueError(f"fit_weight must be in [0, 1], got {fit_weight}")
        self.fit_weight = fit_weight
        self.max_candidates = max_candidates
        self.seed = seed

        self.all_geos: list[str] = self.data[self.geo_variable].unique().sort().to_list()
        self.rankings: list[dict[str, Any]] | None = None
        self.results: dict[str, Any] | None = None

    def _candidate_sets(
        self,
        n_test_geos: list[int],
        include: list[str],
        exclude: list[str],
    ) -> list[tuple[str, ...]]:
        """Enumerate (or sample) candidate test-geo sets.

        ``include`` geos are forced into every set; ``exclude`` geos never appear
        in a test set (they remain available as controls). The remaining "free"
        geos fill each set up to the requested sizes. When the full enumeration
        would exceed ``max_candidates``, a seeded sample is drawn and a warning is
        emitted so the truncation is never silent.

        Parameters
        ----------
        n_test_geos : list of int
            Candidate test-set sizes.
        include, exclude : list of str
            Geos forced into / barred from every test set.

        Returns
        -------
        A list of candidate test-geo tuples.
        """
        include_set, exclude_set = set(include), set(exclude)
        unknown = (include_set | exclude_set) - set(self.all_geos)
        if unknown:
            raise ValueError(f"Unknown geos in include/exclude: {sorted(unknown)}")
        if include_set & exclude_set:
            raise ValueError(f"Geos cannot be both included and excluded: {sorted(include_set & exclude_set)}")
        free = [g for g in self.all_geos if g not in include_set and g not in exclude_set]

        # Count the full enumeration first so we can warn before materialising it.
        per_size = {}
        for n in n_test_geos:
            k = n - len(include_set)
            if k < 0 or k > len(free):
                continue
            per_size[n] = comb(len(free), k)
        if not per_size:
            raise ValueError("No valid candidate sizes: check n_test_geos against include/exclude and geo count")
        total = sum(per_size.values())

        rng = np.random.default_rng(self.seed)
        if total <= self.max_candidates:
            candidates = []
            for n in per_size:
                k = n - len(include_set)
                for combo in combinations(free, k):
                    candidates.append(tuple(sorted(include_set.union(combo))))
            return candidates

        warnings.warn(
            f"{total} candidate test sets exceed max_candidates={self.max_candidates}; "
            f"evaluating a random sample of {self.max_candidates}.",
            stacklevel=2,
        )
        # Sample sizes proportionally to how many sets each contributes.
        sizes = list(per_size)
        weights = np.array([per_size[n] for n in sizes], dtype=float)
        weights /= weights.sum()
        seen: set[tuple[str, ...]] = set()
        attempts = 0
        while len(seen) < self.max_candidates and attempts < self.max_candidates * 50:
            attempts += 1
            n = int(rng.choice(sizes, p=weights))
            k = n - len(include_set)
            combo = tuple(rng.choice(free, size=k, replace=False))
            seen.add(tuple(sorted(include_set.union(combo))))
        return list(seen)

    def _pre_fit(self, power: PowerAnalysis, duration: int) -> float | None:
        """Scaled pre-period fit of the estimator for a candidate split.

        Runs a single no-effect fit on the tail of the clean history and returns
        the estimator's split-conformal band normalised by the test geos' mean
        pre-period outcome, so the metric is scale-free and comparable across
        candidates. Smaller is a tighter fit. Returns ``None`` when the estimator
        does not expose a ``conformal_band``.

        Parameters
        ----------
        power : PowerAnalysis
            A configured power analysis for the candidate (reused for its already
            resolved geos and clean history).
        duration : int
            The placebo post-window length, matching the power simulation.

        Returns
        -------
        The scaled conformal band, or None if unavailable.
        """
        history = power.history
        pre_boundary = history[-duration - 1]
        post_boundary = history[-duration]
        date_str = nw.col(self.date_variable).cast(nw.String)
        sub = self.data.filter(date_str <= history[-1])

        model = self.estimator(
            sub.to_native(),
            geo_variable=self.geo_variable,
            test_geos=power.test_geos,
            control_geos=power.control_geos,
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
        band = model.results.get("conformal_band")
        if band is None:
            return None
        scale = sub.filter(nw.col(self.geo_variable).is_in(power.test_geos) & (date_str <= pre_boundary))[
            self.y_variable
        ].mean()
        scale = abs(float(scale)) if scale else 1.0
        return float(band) / (scale or 1.0)

    def search(
        self,
        n_test_geos: list[int],
        effect_size: float = 0.10,
        duration: int = 14,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        n_sims: int = 100,
    ) -> "MarketSelection":
        """Score and rank candidate test-geo sets.

        Parameters
        ----------
        n_test_geos : list of int
            Candidate test-set sizes to consider.
        effect_size : float, default=0.10
            The effect the power simulation injects (a fractional lift under
            multiplicative injection).
        duration : int, default=14
            The experiment length, in number of post-period dates.
        include, exclude : list of str, optional
            Geos forced into / barred from every test set.
        n_sims : int, default=100
            Placebo experiments per candidate, forwarded to the power simulation.

        Returns
        -------
        MarketSelection
            Itself, so it can be chained with summarize() / plot().
        """
        candidates = self._candidate_sets(n_test_geos, include or [], exclude or [])
        records: list[dict[str, Any]] = []
        for test_geos in candidates:
            control_geos = [g for g in self.all_geos if g not in set(test_geos)]
            pa = PowerAnalysis(
                self.data.to_native(),
                geo_variable=self.geo_variable,
                test_geos=list(test_geos),
                control_geos=control_geos,
                date_variable=self.date_variable,
                pre_period=self.pre_period,
                y_variable=self.y_variable,
                alpha=self.alpha,
                estimator=self.estimator,
                estimator_kwargs=self.estimator_kwargs,
                injection=self.injection,
                min_pre_periods=self.min_pre_periods,
                seed=self.seed,
            ).simulate(effect_sizes=[effect_size], durations=[duration], n_sims=n_sims)
            curve = pa.power_curve
            if curve is None:
                raise ValueError("power_curve must not be None")
            records.append(
                {
                    "test_geos": list(test_geos),
                    "n_test": len(test_geos),
                    "n_control": len(control_geos),
                    "power": curve[0]["power"],
                    "pre_fit": self._pre_fit(pa, duration),
                }
            )

        self.rankings = self._score(records)
        self.results = {
            "rankings": self.rankings,
            "effect_size": effect_size,
            "duration": duration,
            "n_candidates": len(records),
        }
        return self

    def _score(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Combine power and pre-period fit into a single score, best-first.

        Power is already in ``[0, 1]``. Pre-period fit is min-max normalised
        across candidates and inverted (tighter fit -> higher), then blended with
        power via ``fit_weight``. When fit is unavailable or constant, the score
        is power alone.

        Parameters
        ----------
        records : list of dict
            Per-candidate raw metrics from ``search``.

        Returns
        -------
        The records with a ``score`` key, sorted by score descending.
        """
        bands = [r["pre_fit"] for r in records]
        usable = [b for b in bands if b is not None]
        lo, hi = (min(usable), max(usable)) if usable else (0.0, 0.0)
        use_fit = self.fit_weight > 0.0 and len(usable) == len(records) and hi > lo

        for r in records:
            if use_fit:
                norm = (r["pre_fit"] - lo) / (hi - lo)
                r["score"] = (1 - self.fit_weight) * r["power"] + self.fit_weight * (1 - norm)
            else:
                r["score"] = r["power"]
        # Break ties on power, then on a tighter pre-period fit.
        return sorted(
            records,
            key=lambda r: (r["score"], r["power"], -(r["pre_fit"] if r["pre_fit"] is not None else np.inf)),
            reverse=True,
        )

    def summarize(self, top: int = 10) -> None:
        """Print the ranked candidate test sets.

        Parameters
        ----------
        top : int, default=10
            The number of top-ranked candidates to display.

        Raises
        ------
        ValueError
            If ``search()`` has not been run yet.
        """
        if self.rankings is None:
            raise ValueError("Call search() before summarize()")
        rows = self.rankings[:top]
        table = {
            "Rank": list(range(1, len(rows) + 1)),
            "Test Geos": [", ".join(r["test_geos"]) for r in rows],
            "Power": [f"{r['power']:.0%}" for r in rows],
            "Pre-fit": ["n/a" if r["pre_fit"] is None else f"{r['pre_fit']:.3f}" for r in rows],
            "Score": [f"{r['score']:.3f}" for r in rows],
        }
        print(tabulate(table, headers="keys", tablefmt="grid"))

    def plot(self, top: int = 10) -> None:
        """Plot the score of the top-ranked candidate test sets.

        Parameters
        ----------
        top : int, default=10
            The number of top-ranked candidates to display.

        Raises
        ------
        ValueError
            If ``search()`` has not been run yet.
        """
        if self.rankings is None:
            raise ValueError("Call search() before plot()")
        rows = self.rankings[:top][::-1]  # best at the top of a horizontal bar
        labels = [", ".join(r["test_geos"]) for r in rows]
        fig = go.Figure(
            go.Bar(
                x=[r["score"] for r in rows],
                y=labels,
                orientation="h",
                text=[f"power {r['power']:.0%}" for r in rows],
            )
        )
        fig.update_layout(
            title="Top candidate test markets",
            xaxis_title="Score",
            yaxis_title="Test geos",
            xaxis_range=[0, 1],
        )
        fig.show()
