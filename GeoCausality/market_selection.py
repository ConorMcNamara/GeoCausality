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
from narwhals.typing import IntoDataFrame
from tabulate import tabulate  # type: ignore

from GeoCausality._base import Estimator, date_key, period_key
from GeoCausality._plotting import MarketRankingPlotData, PlotBackend, check_backend, get_theme, render
from GeoCausality.augmented_synthetic_control import AugmentedSyntheticControl
from GeoCausality.power import PowerAnalysis


class MarketSelection:
    """Search candidate test-geo sets and rank them by power and pre-period fit.

    When ``dtw_top_k`` is passed to :meth:`search`, candidates are first ranked
    by Dynamic Time Warping similarity between test geos and their nearest
    controls, and only the most promising ones proceed to the expensive power
    simulation -- a pre-filter inspired by the R *MarketMatching* package.
    """

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
            # Every geo outside the test set is a control, so at least one must remain.
            if k < 0 or k > len(free) or n >= len(self.all_geos):
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

    @staticmethod
    def _check_window(window: int | None) -> None:
        """Validate a Sakoe-Chiba band half-width.

        Parameters
        ----------
        window : int, optional
            The band half-width, or ``None`` for unconstrained DTW.

        Raises
        ------
        ValueError
            If ``window`` is not ``None`` or a non-negative integer. ``bool`` is
            rejected even though it subclasses ``int``.
        """
        if window is None:
            return
        if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
            raise ValueError(f"dtw_window must be a non-negative integer or None, got {window!r}")
        if window < 0:
            raise ValueError(f"dtw_window must be non-negative, got {window}")

    @staticmethod
    def _dtw_batch(s: np.ndarray, t: np.ndarray, window: int | None = None) -> np.ndarray:
        """Dynamic Time Warping distances for a batch of series pairs.

        Runs the usual recurrence
        ``D[i, j] = |s[i] - t[j]| + min(D[i-1, j], D[i, j-1], D[i-1, j-1])``
        one anti-diagonal (``i + j = k``) at a time, since every cell on a
        diagonal depends only on the previous two. Each diagonal is updated for
        all pairs at once, so the Python loop runs ``n + m`` times regardless of
        the batch size. The arithmetic is the same as the cell-by-cell loop, so
        the distances match it exactly.

        Parameters
        ----------
        s : 2-D array of shape (n_pairs, n)
            The first series of each pair.
        t : 2-D array of shape (n_pairs, m)
            The second series of each pair.
        window : int, optional
            Sakoe-Chiba band half-width: only cells with ``|i - j| <= window``
            are visited. It is widened to ``|n - m|`` when narrower so the end
            cell stays reachable. ``None`` (the default) is unconstrained DTW.

        Returns
        -------
        A 1-D array of the accumulated DTW cost (L1 element cost) per pair.
        """
        MarketSelection._check_window(window)
        s = np.asarray(s, dtype=float)
        t = np.asarray(t, dtype=float)
        n_pairs, n = s.shape
        m = t.shape[1]
        if n == 0 or m == 0:
            return np.full(n_pairs, 0.0 if n == m else np.inf)
        if window is not None:
            window = max(int(window), abs(n - m))

        # Diagonals are indexed by i (row 0..n) and laid out as (i, pair) so every
        # slice below is contiguous. ``t`` is reversed so that t[j - 1] with
        # j = k - i is also a forward slice in i.
        s_rows = np.ascontiguousarray(s.T)
        t_rev = np.ascontiguousarray(t[:, ::-1].T)
        prev2 = np.full((n + 1, n_pairs), np.inf)  # diagonal k - 2
        prev1 = np.full((n + 1, n_pairs), np.inf)  # diagonal k - 1
        cur = np.full((n + 1, n_pairs), np.inf)
        prev2[0] = 0.0  # D[0, 0]
        cost = np.empty((min(n, m), n_pairs))
        best = np.empty((min(n, m), n_pairs))

        for k in range(2, n + m + 1):
            lo, hi = max(1, k - m), min(n, k - 1)
            if window is not None:
                lo, hi = max(lo, (k - window + 1) // 2), min(hi, (k + window) // 2)
            # lo and hi each advance by at most one per diagonal, so the next two
            # diagonals only read cells lo - 1 .. hi + 1 of this one. The two
            # cells just outside [lo, hi] are off the grid or out of the band.
            cur[lo - 1] = np.inf
            if hi < n:
                cur[hi + 1] = np.inf
            if lo <= hi:
                width = hi - lo + 1
                c, b = cost[:width], best[:width]
                np.subtract(s_rows[lo - 1 : hi], t_rev[m - k + lo : m - k + hi + 1], out=c)
                np.abs(c, out=c)
                # D[i-1, j], D[i, j-1] and D[i-1, j-1] respectively.
                np.minimum(prev1[lo - 1 : hi], prev1[lo : hi + 1], out=b)
                np.minimum(b, prev2[lo - 1 : hi], out=b)
                np.add(c, b, out=cur[lo : hi + 1])
            prev2, prev1, cur = prev1, cur, prev2
        return prev1[n].copy()

    @staticmethod
    def _dtw_distance(s: np.ndarray, t: np.ndarray, window: int | None = None) -> float:
        """Dynamic Time Warping distance between two 1-D series.

        Parameters
        ----------
        s, t : 1-D arrays
            The two time series.
        window : int, optional
            Sakoe-Chiba band half-width (see ``_dtw_batch``). ``None`` (the
            default) is unconstrained DTW.

        Returns
        -------
        The accumulated DTW cost (L1 element cost).
        """
        s_arr = np.asarray(s, dtype=float).reshape(1, -1)
        t_arr = np.asarray(t, dtype=float).reshape(1, -1)
        return float(MarketSelection._dtw_batch(s_arr, t_arr, window)[0])

    def _geo_series(self) -> dict[str, np.ndarray]:
        """Extract the z-scored pre-period time series per geo.

        Z-scoring makes DTW distances comparable across geos with different
        outcome scales: a geo with mean-100 and a geo with mean-10000 are
        judged on shape, not level. Missing (null / NaN) and infinite outcomes
        are dropped before z-scoring -- a single one would otherwise turn the
        whole series, and every DTW distance to it, into NaN -- and DTW aligns
        the shorter series that remain. Rows with a null geo ID are dropped too,
        so a null geo gets an empty series (infinitely far from every other).

        Returns
        -------
        A mapping from geo name to its z-scored pre-period series as a numpy
        array, sorted by date.
        """
        date_str = date_key(self.data, self.date_variable)
        pre_key = period_key(self.pre_period, self.data, self.date_variable)
        pre = self.data.filter(date_str <= pre_key).sort(self.date_variable)
        y = pre[self.y_variable].to_numpy().astype(float)
        keep = np.isfinite(y) & ~pre[self.geo_variable].is_null().to_numpy()
        # Group the date-sorted rows by geo in one pass. A stable sort on the geo
        # code keeps each geo's rows in the same (date) order as filtering would.
        geo_codes, inverse = np.unique(pre[self.geo_variable].to_numpy()[keep], return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        bounds = np.searchsorted(inverse[order], np.arange(len(geo_codes) + 1))
        y_sorted = y[keep][order]
        groups = {geo: y_sorted[bounds[i] : bounds[i + 1]] for i, geo in enumerate(geo_codes)}
        result: dict[str, np.ndarray] = {}
        for geo in self.all_geos:
            vals = groups.get(geo, np.empty(0))
            if vals.size == 0:
                result[geo] = vals
                continue
            std = vals.std()
            if std > 0:
                vals = (vals - vals.mean()) / std
            else:
                vals = vals - vals.mean()
            result[geo] = vals
        return result

    def _dtw_matrix(
        self,
        rows: set[str] | None = None,
        window: int | None = None,
        chunk_size: int = 2048,
    ) -> np.ndarray:
        """Compute the matrix of pairwise DTW distances between geos.

        Pairs are batched by series length and fed through ``_dtw_batch`` in
        chunks of ``chunk_size`` to bound memory.

        Parameters
        ----------
        rows : set of str, optional
            Only compute pairs involving at least one of these geos; every other
            off-diagonal entry is left as NaN. ``None`` computes all pairs.
        window : int, optional
            Sakoe-Chiba band half-width (see ``_dtw_batch``).
        chunk_size : int, default=2048
            The most pairs pushed through the recurrence at once.

        Returns
        -------
        A symmetric ``(n_geos, n_geos)`` array of DTW distances, indexed in
        ``all_geos`` order, with a zero diagonal.
        """
        series = self._geo_series()
        geos = list(series)
        n_geos = len(geos)
        dist = np.full((n_geos, n_geos), np.nan)
        np.fill_diagonal(dist, 0.0)

        a_idx, b_idx = np.triu_indices(n_geos, k=1)
        if rows is not None:
            wanted = np.array([g in rows for g in geos], dtype=bool)
            keep = wanted[a_idx] | wanted[b_idx]
            a_idx, b_idx = a_idx[keep], b_idx[keep]

        # Pairs in one batch must share both series lengths.
        lengths = np.array([len(series[g]) for g in geos])
        by_length = {n: np.array([series[g] for g in geos if len(series[g]) == n]) for n in set(lengths.tolist())}
        position = np.zeros(n_geos, dtype=int)  # row of each geo within its length group
        for n in by_length:
            position[lengths == n] = np.arange(int((lengths == n).sum()))
        pair_lengths = np.stack([lengths[a_idx], lengths[b_idx]], axis=1)
        for n, m in {tuple(p) for p in pair_lengths.tolist()}:
            sel = np.flatnonzero((pair_lengths[:, 0] == n) & (pair_lengths[:, 1] == m))
            for start in range(0, len(sel), chunk_size):
                chunk = sel[start : start + chunk_size]
                a, b = a_idx[chunk], b_idx[chunk]
                d = self._dtw_batch(by_length[n][position[a]], by_length[m][position[b]], window)
                dist[a, b] = d
                dist[b, a] = d
        return dist

    def _pairwise_dtw(self, window: int | None = None) -> dict[tuple[str, str], float]:
        """Compute pairwise DTW distances between all geos.

        Parameters
        ----------
        window : int, optional
            Sakoe-Chiba band half-width (see ``_dtw_batch``).

        Returns
        -------
        A mapping from ``(geo_a, geo_b)`` to DTW distance, with both orderings
        stored for convenient lookup.
        """
        dist = self._dtw_matrix(window=window)
        return {(a, b): float(dist[i, j]) for i, a in enumerate(self.all_geos) for j, b in enumerate(self.all_geos)}

    def _dtw_filter(
        self,
        candidates: list[tuple[str, ...]],
        top_k: int,
        window: int | None = None,
    ) -> list[tuple[str, ...]]:
        """Keep the ``top_k`` candidates whose test geos are closest to controls.

        For each candidate, the score is the average over test geos of each test
        geo's minimum DTW distance to any control geo. Lower is better: it means
        the control pool contains close shape-matches for every test geo.

        Parameters
        ----------
        candidates : list of tuples
            The full candidate pool from ``_candidate_sets``.
        top_k : int
            How many candidates to keep.
        window : int, optional
            Sakoe-Chiba band half-width (see ``_dtw_batch``).

        Returns
        -------
        The ``top_k`` candidates with the smallest DTW scores.
        """
        # Scores only read distances from a test geo, so pairs where neither geo
        # is ever tested (e.g. two excluded geos) are skipped.
        dist = self._dtw_matrix(rows=set().union(*candidates), window=window)
        index = {g: i for i, g in enumerate(self.all_geos)}

        def score(test_geos: tuple[str, ...]) -> float:
            test_idx = [index[g] for g in test_geos]
            controls = np.ones(len(self.all_geos), dtype=bool)
            controls[test_idx] = False
            return float(np.mean(dist[np.ix_(test_idx, controls)].min(axis=1)))

        scored = sorted(candidates, key=score)
        return scored[:top_k]

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
        date_str = date_key(self.data, self.date_variable)
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
        # Only the fit's band is used here, so interval warnings are noise.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The pre-period is too short", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="The conformal confidence set is not", category=RuntimeWarning)
            model.pre_process().generate()
        if model.results is None:
            raise ValueError("model.results must not be None")
        band = model.results.get("conformal_band")
        if band is None:
            return None
        if not np.isfinite(band):
            # A pre-period too short for a finite (1 - alpha) band still has a
            # usable fit; rank it by its largest absolute pre-period residual.
            actual_pre = model.results.get("_actual_pre")
            prediction_pre = model.results.get("_prediction_pre")
            if actual_pre is None or prediction_pre is None:
                return None
            resid = np.asarray(actual_pre, dtype=float) - np.asarray(prediction_pre, dtype=float)
            band = float(np.max(np.abs(resid)))
        # Normalise by the mean of the treated series summed across the test geos
        # per date -- the same scale the estimator's conformal band is computed on.
        # Averaging the individual geo-date rows instead leaves a spurious factor
        # of len(test_geos), which biases search() against larger test sets.
        summed = (
            sub.filter(nw.col(self.geo_variable).is_in(power.test_geos) & (date_str <= pre_boundary))
            .group_by(self.date_variable)
            .agg(nw.col(self.y_variable).sum())
        )
        scale = summed[self.y_variable].mean()
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
        dtw_top_k: int | None = None,
        n_jobs: int = 1,
        dtw_window: int | None = None,
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
        dtw_top_k : int, optional
            When set, candidates are first ranked by Dynamic Time Warping
            similarity and only the top ``dtw_top_k`` proceed to the power
            simulation. This can dramatically speed up the search when the
            candidate pool is large.
        n_jobs : int, default 1
            Number of parallel workers for the candidate scoring loop.
            ``-1`` uses all available cores.  ``1`` (the default) runs
            sequentially with no parallelism overhead.
        dtw_window : int, optional
            Sakoe-Chiba band for the ``dtw_top_k`` pre-filter: series may only
            be warped by up to this many dates. This cuts the DTW cost to
            roughly ``(2 * dtw_window + 1) / T`` of the unconstrained cost for
            ``T`` pre-period dates, but distances (and so the filtered
            candidates) can differ from the unconstrained ones. ``None`` (the
            default) is unconstrained DTW. Ignored unless ``dtw_top_k`` is set.

        Returns
        -------
        MarketSelection
            Itself, so it can be chained with summarize() / plot().
        """
        self._check_window(dtw_window)
        candidates = self._candidate_sets(n_test_geos, include or [], exclude or [])
        if dtw_top_k is not None and dtw_top_k < len(candidates):
            candidates = self._dtw_filter(candidates, dtw_top_k, window=dtw_window)

        native_data = self.data.to_native()
        geo_var = self.geo_variable
        date_var = self.date_variable
        pre_per = self.pre_period
        y_var = self.y_variable
        alpha_val = self.alpha
        est = self.estimator
        est_kw = self.estimator_kwargs
        inj = self.injection
        min_pre = self.min_pre_periods
        seed_val = self.seed
        all_geos = self.all_geos

        def _score_candidate(test_geos: tuple[str, ...]) -> dict[str, Any]:
            control = [g for g in all_geos if g not in set(test_geos)]
            pa = PowerAnalysis(
                native_data,
                geo_variable=geo_var,
                test_geos=list(test_geos),
                control_geos=control,
                date_variable=date_var,
                pre_period=pre_per,
                y_variable=y_var,
                alpha=alpha_val,
                estimator=est,
                estimator_kwargs=est_kw,
                injection=inj,
                min_pre_periods=min_pre,
                seed=seed_val,
            ).simulate(effect_sizes=[effect_size], durations=[duration], n_sims=n_sims)
            curve = pa.power_curve
            if curve is None:
                raise ValueError("power_curve must not be None")
            return {
                "test_geos": list(test_geos),
                "n_test": len(test_geos),
                "n_control": len(control),
                "power": curve[0]["power"],
                "pre_fit": self._pre_fit(pa, duration),
            }

        if n_jobs == 1:
            records: list[dict[str, Any]] = [_score_candidate(c) for c in candidates]
        else:
            from joblib import Parallel, delayed

            records = Parallel(n_jobs=n_jobs)(delayed(_score_candidate)(c) for c in candidates)

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

    def plot(self, top: int = 10, dark_mode: bool = False, backend: PlotBackend = "plotly") -> Any:
        """Plot the score of the top-ranked candidate test sets.

        Parameters
        ----------
        top : int, default=10
            The number of top-ranked candidates to display.
        dark_mode : bool, default=False
            Render with a dark background and lighter series colours.
        backend : {"plotly", "altair"}, default="plotly"
            Library to draw the figure with. ``"altair"`` needs the optional
            dependency (``pip install geocausality[altair]``).

        Returns
        -------
        plotly Figure or altair LayerChart
            The figure. Notebooks render it when it is the cell's last
            expression; elsewhere call its ``.show()`` (Plotly) or ``.save()``
            (Altair).

        Raises
        ------
        ValueError
            If ``search()`` has not been run yet.
        """
        check_backend(backend)
        if self.rankings is None:
            raise ValueError("Call search() before plot()")
        rows = self.rankings[:top]
        data = MarketRankingPlotData(
            labels=[", ".join(r["test_geos"]) for r in rows],
            scores=[r["score"] for r in rows],
            powers=[r["power"] for r in rows],
        )
        return render(data, get_theme(dark_mode), backend)
