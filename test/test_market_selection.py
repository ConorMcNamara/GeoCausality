"""Tests for the pre-experiment market-selection module.

``MarketSelection`` enumerates candidate test-geo sets, scores each via the
``PowerAnalysis`` engine plus a pre-period fit metric, and returns a ranked
recommendation. These tests assert candidate generation (sizes, include/exclude,
sampling+warning when the enumeration is large), the ranking contract, and the
reporting surface. ``SyntheticControl`` is the fast, deterministic estimator.
"""

import io
from contextlib import redirect_stdout
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from GeoCausality.market_selection import MarketSelection
from GeoCausality.power import PowerAnalysis
from GeoCausality.synthetic_control import SyntheticControl

N_DATES = 40
PRE_PERIOD = "2021-02-09"  # the whole series is clean history
N_GEOS = 8


def _make_history(seed: int = 0, n_geos: int = N_GEOS) -> pl.DataFrame:
    """Builds a clean geo panel (no treatment effect) with shared latent factors.

    Parameters
    ----------
    seed : int
        Seed for reproducibility.
    n_geos : int
        Number of geos in the panel.
    """
    rng = np.random.default_rng(seed)
    d0 = date(2021, 1, 1)
    dates = [d0 + timedelta(days=i) for i in range(N_DATES)]
    geos = [f"g{i}" for i in range(n_geos)]
    factors = rng.normal(0, 1, size=(2, N_DATES)).cumsum(axis=1)
    loadings = {g: rng.uniform(0.5, 2.0, size=2) for g in geos}

    rows = []
    for g in geos:
        for di, d in enumerate(dates):
            y = 100.0 + loadings[g] @ factors[:, di] + rng.normal(0, 2)
            rows.append({"geo": g, "date": d, "y": float(y)})
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def history() -> pl.DataFrame:
    return _make_history()


def _selection(history: pl.DataFrame, **overrides: object) -> MarketSelection:
    kwargs = {
        "geo_variable": "geo",
        "date_variable": "date",
        "pre_period": PRE_PERIOD,
        "y_variable": "y",
        "alpha": 0.1,
        "estimator": SyntheticControl,
        "seed": 0,
    }
    kwargs.update(overrides)
    return MarketSelection(history, **kwargs)


@pytest.fixture(scope="module")
def searched_size2(history: pl.DataFrame) -> MarketSelection:
    return _selection(history).search(n_test_geos=[2], effect_size=0.3, duration=10, n_sims=8)


@pytest.fixture(scope="module")
def searched_sizes_1_2(history: pl.DataFrame) -> MarketSelection:
    return _selection(history).search(n_test_geos=[1, 2], effect_size=0.3, duration=10, n_sims=8)


class TestCandidateGeneration:
    @staticmethod
    def test_enumerates_requested_sizes(history: pl.DataFrame) -> None:
        ms = _selection(history)
        cands = ms._candidate_sets(n_test_geos=[1, 2], include=[], exclude=[])
        # C(8,1) + C(8,2) = 8 + 28 = 36
        assert len(cands) == 36
        assert all(len(c) in (1, 2) for c in cands)
        assert len({c for c in cands}) == len(cands)  # all unique

    @staticmethod
    def test_include_forced_into_every_set(history: pl.DataFrame) -> None:
        ms = _selection(history)
        cands = ms._candidate_sets(n_test_geos=[2], include=["g0"], exclude=[])
        assert all("g0" in c for c in cands)
        assert len(cands) == 7  # g0 + one of the other 7

    @staticmethod
    def test_exclude_never_a_test_geo(history: pl.DataFrame) -> None:
        ms = _selection(history)
        cands = ms._candidate_sets(n_test_geos=[2], include=[], exclude=["g7"])
        assert all("g7" not in c for c in cands)

    @staticmethod
    def test_unknown_geo_raises(history: pl.DataFrame) -> None:
        ms = _selection(history)
        with pytest.raises(ValueError):
            ms._candidate_sets(n_test_geos=[1], include=["g99"], exclude=[])

    @staticmethod
    def test_include_exclude_overlap_raises(history: pl.DataFrame) -> None:
        ms = _selection(history)
        with pytest.raises(ValueError):
            ms._candidate_sets(n_test_geos=[1], include=["g0"], exclude=["g0"])

    @staticmethod
    def test_large_enumeration_samples_and_warns(history: pl.DataFrame) -> None:
        ms = _selection(history, max_candidates=5)
        with pytest.warns(UserWarning, match="exceed max_candidates"):
            cands = ms._candidate_sets(n_test_geos=[1, 2, 3], include=[], exclude=[])
        assert len(cands) <= 5


class TestSearch:
    @pytest.mark.slow
    @staticmethod
    def test_rankings_structure(searched_size2: MarketSelection) -> None:
        assert searched_size2.rankings is not None
        assert len(searched_size2.rankings) == 28  # C(8,2)
        for r in searched_size2.rankings:
            assert {"test_geos", "n_test", "n_control", "power", "pre_fit", "score"} <= r.keys()
            assert r["n_test"] + r["n_control"] == N_GEOS
            assert 0.0 <= r["power"] <= 1.0

    @pytest.mark.slow
    @staticmethod
    def test_rankings_sorted_by_score_desc(searched_sizes_1_2: MarketSelection) -> None:
        scores = [r["score"] for r in searched_sizes_1_2.rankings]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.slow
    @staticmethod
    def test_pre_fit_computed_for_synthetic_control(searched_size2: MarketSelection) -> None:
        assert all(r["pre_fit"] is not None and r["pre_fit"] >= 0.0 for r in searched_size2.rankings)

    @pytest.mark.slow
    @staticmethod
    def test_fit_weight_zero_ranks_on_power(history: pl.DataFrame) -> None:
        ms = _selection(history, fit_weight=0.0).search(n_test_geos=[2], effect_size=0.3, duration=10, n_sims=8)
        assert all(r["score"] == r["power"] for r in ms.rankings)

    @staticmethod
    def test_invalid_fit_weight_raises(history: pl.DataFrame) -> None:
        with pytest.raises(ValueError):
            _selection(history, fit_weight=1.5)


class TestReporting:
    @pytest.mark.slow
    @staticmethod
    def test_summarize_runs(searched_sizes_1_2: MarketSelection) -> None:
        with redirect_stdout(io.StringIO()) as buffer:
            searched_sizes_1_2.summarize(top=5)
        out = buffer.getvalue()
        assert "Score" in out
        assert "Test Geos" in out

    @staticmethod
    def test_summarize_requires_search(history: pl.DataFrame) -> None:
        with pytest.raises(ValueError):
            _selection(history).summarize()


class TestDTW:
    @staticmethod
    def test_dtw_identical_series_is_zero() -> None:
        s = np.array([1.0, 2.0, 3.0])
        assert MarketSelection._dtw_distance(s, s) == 0.0

    @staticmethod
    def test_dtw_symmetric() -> None:
        s = np.array([1.0, 2.0, 3.0])
        t = np.array([1.0, 2.0, 5.0])
        assert MarketSelection._dtw_distance(s, t) == MarketSelection._dtw_distance(t, s)

    @staticmethod
    def test_dtw_known_value() -> None:
        s = np.array([0.0, 0.0, 0.0])
        t = np.array([1.0, 1.0, 1.0])
        assert MarketSelection._dtw_distance(s, t) == pytest.approx(3.0)

    @staticmethod
    def test_dtw_different_lengths() -> None:
        s = np.array([0.0, 1.0])
        t = np.array([0.0, 1.0, 2.0])
        d = MarketSelection._dtw_distance(s, t)
        assert d > 0.0
        assert np.isfinite(d)

    @staticmethod
    def test_geo_series_z_scored(history: pl.DataFrame) -> None:
        ms = _selection(history)
        series = ms._geo_series()
        assert len(series) == N_GEOS
        for vals in series.values():
            assert abs(vals.mean()) < 1e-10
            if vals.std() > 0:
                assert vals.std() == pytest.approx(1.0, abs=0.1)

    @staticmethod
    def test_pairwise_dtw_structure(history: pl.DataFrame) -> None:
        ms = _selection(history)
        dist = ms._pairwise_dtw()
        for g in ms.all_geos:
            assert dist[(g, g)] == 0.0
        for a in ms.all_geos:
            for b in ms.all_geos:
                assert dist[(a, b)] == dist[(b, a)]

    @staticmethod
    def test_dtw_filter_reduces_candidates(history: pl.DataFrame) -> None:
        ms = _selection(history)
        candidates = ms._candidate_sets(n_test_geos=[1, 2], include=[], exclude=[])
        assert len(candidates) == 36  # C(8,1) + C(8,2)
        filtered = ms._dtw_filter(candidates, top_k=10)
        assert len(filtered) == 10

    @staticmethod
    def test_search_with_dtw_top_k(history: pl.DataFrame) -> None:
        ms = _selection(history).search(
            n_test_geos=[1, 2],
            effect_size=0.3,
            duration=10,
            n_sims=6,
            dtw_top_k=5,
        )
        assert ms.rankings is not None
        assert len(ms.rankings) == 5

    @staticmethod
    def test_dtw_top_k_larger_than_pool_is_noop(history: pl.DataFrame) -> None:
        ms = _selection(history).search(
            n_test_geos=[1],
            effect_size=0.3,
            duration=10,
            n_sims=6,
            dtw_top_k=100,
        )
        assert ms.rankings is not None
        assert len(ms.rankings) == 8  # C(8,1) = 8, all kept


def _reference_dtw(s: np.ndarray, t: np.ndarray, window: int | None = None) -> float:
    """Straightforward cell-by-cell DTW, the original implementation, kept as an oracle.

    Parameters
    ----------
    s, t : 1-D arrays
        The two time series.
    window : int, optional
        Sakoe-Chiba band half-width, widened to ``|n - m|``.
    """
    n, m = len(s), len(t)
    band = None if window is None else max(window, abs(n - m))
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if band is not None and abs(i - j) > band:
                continue
            cost = abs(s[i - 1] - t[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


def _reference_dtw_filter(ms: MarketSelection, candidates: list[tuple[str, ...]], top_k: int) -> list[tuple[str, ...]]:
    """The original dict-based ``_dtw_filter``, built on ``_reference_dtw``."""
    series = ms._geo_series()
    dist = {(a, b): _reference_dtw(series[a], series[b]) for a in ms.all_geos for b in ms.all_geos if a != b}

    def score(test_geos: tuple[str, ...]) -> float:
        controls = [g for g in ms.all_geos if g not in set(test_geos)]
        return float(np.mean([min(dist[(tg, cg)] for cg in controls) for tg in test_geos]))

    return sorted(candidates, key=score)[:top_k]


class TestDTWMatchesReference:
    @staticmethod
    @pytest.mark.parametrize("equal_lengths", [True, False])
    def test_random_series(equal_lengths: bool) -> None:
        rng = np.random.default_rng(42)
        for _ in range(50):
            n = int(rng.integers(1, 30))
            m = n if equal_lengths else int(rng.integers(1, 30))
            s, t = rng.normal(size=n), rng.normal(size=m)
            np.testing.assert_allclose(MarketSelection._dtw_distance(s, t), _reference_dtw(s, t), rtol=1e-12)

    @staticmethod
    def test_batch_matches_pairwise_reference() -> None:
        rng = np.random.default_rng(1)
        s, t = rng.normal(size=(25, 17)), rng.normal(size=(25, 23))
        expected = [_reference_dtw(a, b) for a, b in zip(s, t, strict=True)]
        np.testing.assert_allclose(MarketSelection._dtw_batch(s, t), expected, rtol=1e-12)

    @staticmethod
    @pytest.mark.parametrize(
        ("s", "t"),
        [
            (np.array([3.0]), np.array([1.0])),
            (np.array([3.0]), np.array([1.0, 2.0, 5.0])),
            (np.array([1.0, 2.0, 5.0]), np.array([3.0])),
            (np.zeros(6), np.zeros(6)),
            (np.full(5, 2.0), np.full(8, -1.0)),
            (np.zeros(0), np.zeros(0)),
            (np.zeros(0), np.ones(3)),
        ],
    )
    def test_edge_cases(s: np.ndarray, t: np.ndarray) -> None:
        assert MarketSelection._dtw_distance(s, t) == _reference_dtw(s, t)

    @staticmethod
    @pytest.mark.parametrize("window", [0, 1, 3, 40])
    def test_sakoe_chiba_window(window: int) -> None:
        rng = np.random.default_rng(7)
        for _ in range(20):
            n, m = (int(x) for x in rng.integers(1, 25, size=2))
            s, t = rng.normal(size=n), rng.normal(size=m)
            np.testing.assert_allclose(
                MarketSelection._dtw_distance(s, t, window=window), _reference_dtw(s, t, window=window), rtol=1e-12
            )

    @staticmethod
    def test_wide_window_is_unconstrained() -> None:
        rng = np.random.default_rng(3)
        s, t = rng.normal(size=20), rng.normal(size=15)
        assert MarketSelection._dtw_distance(s, t, window=20) == MarketSelection._dtw_distance(s, t)

    @staticmethod
    def test_pairwise_dtw_matches_reference(history: pl.DataFrame) -> None:
        ms = _selection(history)
        series = ms._geo_series()
        dist = ms._pairwise_dtw()
        for a in ms.all_geos:
            for b in ms.all_geos:
                np.testing.assert_allclose(dist[(a, b)], _reference_dtw(series[a], series[b]), rtol=1e-12)

    @staticmethod
    def test_pairwise_dtw_unequal_lengths(history: pl.DataFrame) -> None:
        # Drop some dates from a few geos so the series come in several lengths.
        ragged = history.filter(~((pl.col("geo") == "g1") & (pl.col("date") < date(2021, 1, 6))))
        ragged = ragged.filter(~((pl.col("geo").is_in(["g4", "g6"])) & (pl.col("date") > date(2021, 2, 6))))
        ms = _selection(ragged)
        series = ms._geo_series()
        assert len({len(v) for v in series.values()}) == 3
        dist = ms._pairwise_dtw()
        for a in ms.all_geos:
            for b in ms.all_geos:
                np.testing.assert_allclose(dist[(a, b)], _reference_dtw(series[a], series[b]), rtol=1e-12)

    @staticmethod
    def test_geo_series_matches_per_geo_filter(history: pl.DataFrame) -> None:
        # Shuffle rows (and use pandas) so the single-pass grouping has to re-sort.
        shuffled = pd.DataFrame(history.sample(fraction=1.0, shuffle=True, seed=0).to_dict(as_series=False))
        ms = _selection(shuffled)
        series = ms._geo_series()
        for geo in ms.all_geos:
            vals = history.filter(pl.col("geo") == geo).sort("date")["y"].to_numpy()
            np.testing.assert_array_equal(series[geo], (vals - vals.mean()) / vals.std())

    @staticmethod
    def test_dtw_filter_matches_reference(history: pl.DataFrame) -> None:
        ms = _selection(history)
        candidates = ms._candidate_sets(n_test_geos=[1, 2, 3], include=[], exclude=[])
        for top_k in (1, 5, 20):
            assert ms._dtw_filter(candidates, top_k=top_k) == _reference_dtw_filter(ms, candidates, top_k)

    @staticmethod
    def test_dtw_filter_with_excluded_geos_matches_reference(history: pl.DataFrame) -> None:
        # Excluded geos are never tested, so pairs between them are skipped.
        ms = _selection(history)
        candidates = ms._candidate_sets(n_test_geos=[2], include=[], exclude=["g0", "g3", "g5"])
        assert ms._dtw_filter(candidates, top_k=4) == _reference_dtw_filter(ms, candidates, 4)

    @staticmethod
    def test_search_dtw_top_k_matches_reference(history: pl.DataFrame) -> None:
        ms = _selection(history)
        candidates = ms._candidate_sets(n_test_geos=[1, 2], include=[], exclude=[])
        expected = _reference_dtw_filter(ms, candidates, 5)
        ms.search(n_test_geos=[1, 2], effect_size=0.3, duration=10, n_sims=6, dtw_top_k=5)
        assert ms.rankings is not None
        assert sorted(tuple(r["test_geos"]) for r in ms.rankings) == sorted(expected)

    @staticmethod
    def test_search_dtw_window(history: pl.DataFrame) -> None:
        ms = _selection(history).search(
            n_test_geos=[1], effect_size=0.3, duration=10, n_sims=6, dtw_top_k=3, dtw_window=2
        )
        assert ms.rankings is not None
        assert len(ms.rankings) == 3
        with pytest.raises(ValueError, match="dtw_window"):
            _selection(history).search(n_test_geos=[1], dtw_top_k=3, dtw_window=-1)

    @staticmethod
    @pytest.mark.parametrize("window", [2.0, 1.5, "2", True])
    def test_dtw_window_must_be_integer(history: pl.DataFrame, window: object) -> None:
        with pytest.raises(ValueError, match="dtw_window must be a non-negative integer"):
            _selection(history).search(n_test_geos=[1], dtw_top_k=3, dtw_window=window)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="dtw_window must be a non-negative integer"):
            MarketSelection._dtw_distance(np.ones(5), np.zeros(5), window=window)  # type: ignore[arg-type]

    @staticmethod
    def test_dtw_window_accepts_numpy_integer() -> None:
        s, t = np.arange(8.0), np.arange(8.0)[::-1]
        assert MarketSelection._dtw_distance(s, t, window=np.int64(2)) == MarketSelection._dtw_distance(s, t, window=2)

    @staticmethod
    def test_dtw_filter_skips_missing_outcomes(history: pl.DataFrame) -> None:
        # One null outcome used to turn a geo's z-scored series (and every
        # distance to it) into NaN, degrading the ranking to enumeration order.
        gappy = history.with_columns(
            pl.when((pl.col("geo") == "g5") & (pl.col("date") == date(2021, 1, 10)))
            .then(None)
            .otherwise(pl.col("y"))
            .alias("y")
        )
        ms = _selection(gappy)
        series = ms._geo_series()
        assert all(np.isfinite(v).all() for v in series.values())
        assert len(series["g5"]) == len(series["g0"]) - 1
        assert not np.isnan(ms._pairwise_dtw()[("g0", "g5")])
        # The ranking matches the one computed with g5's missing date dropped.
        dropped = history.filter(~((pl.col("geo") == "g5") & (pl.col("date") == date(2021, 1, 10))))
        candidates = ms._candidate_sets(n_test_geos=[1, 2], include=[], exclude=[])
        assert ms._dtw_filter(candidates, top_k=5) == _selection(dropped)._dtw_filter(candidates, top_k=5)
        assert ms._dtw_filter(candidates, top_k=5) == _reference_dtw_filter(ms, candidates, 5)

    @staticmethod
    def test_dtw_filter_null_geo_id(history: pl.DataFrame) -> None:
        # Null geo IDs must not break the single-pass grouping; the null geo's
        # rows are dropped, leaving it an empty series as on a per-geo filter.
        nulled = history.with_columns(pl.when(pl.col("geo") == "g0").then(None).otherwise(pl.col("geo")).alias("geo"))
        ms = _selection(nulled)
        series = ms._geo_series()
        assert len(series[None]) == 0  # type: ignore[index]
        candidates = [c for c in ms._candidate_sets(n_test_geos=[1], include=[], exclude=[]) if c != (None,)]
        kept = ms._dtw_filter(candidates, top_k=3)
        assert kept == _reference_dtw_filter(ms, candidates, 3)


class TestParallel:
    @staticmethod
    def test_parallel_matches_sequential(history: pl.DataFrame) -> None:
        seq = _selection(history).search(n_test_geos=[1], effect_size=0.3, duration=10, n_sims=6, n_jobs=1)
        par = _selection(history).search(n_test_geos=[1], effect_size=0.3, duration=10, n_sims=6, n_jobs=2)
        assert len(seq.rankings) == len(par.rankings)
        seq_scores = [r["score"] for r in seq.rankings]
        par_scores = [r["score"] for r in par.rankings]
        assert seq_scores == pytest.approx(par_scores)


def test_pre_fit_normalizes_by_summed_series(history: pl.DataFrame) -> None:
    duration = 5
    test_geos = ["g0", "g1"]
    control_geos = [f"g{i}" for i in range(2, N_GEOS)]
    ms = _selection(history)
    pa = PowerAnalysis(
        history,
        geo_variable="geo",
        test_geos=test_geos,
        control_geos=control_geos,
        date_variable="date",
        pre_period=PRE_PERIOD,
        y_variable="y",
        alpha=0.1,
        estimator=SyntheticControl,
        seed=0,
    )
    pre_fit = ms._pre_fit(pa, duration)
    assert pre_fit is not None

    hist = pa.history
    pre_boundary, post_boundary = hist[-duration - 1], hist[-duration]
    model = SyntheticControl(
        history,
        geo_variable="geo",
        test_geos=test_geos,
        control_geos=control_geos,
        date_variable="date",
        pre_period=pre_boundary,
        post_period=post_boundary,
        y_variable="y",
        alpha=0.1,
    )
    model.pre_process().generate()
    band = float(model.results["conformal_band"])
    pre = history.filter(pl.col("geo").is_in(test_geos) & (pl.col("date").cast(pl.Utf8) <= pre_boundary))
    per_geo_mean = abs(float(pre["y"].mean()))
    summed_mean = abs(float(pre.group_by("date").agg(pl.col("y").sum())["y"].mean()))

    assert pre_fit == pytest.approx(band / summed_mean, rel=1e-9)
    assert pre_fit == pytest.approx((band / per_geo_mean) / len(test_geos), rel=1e-9)
    assert pre_fit != pytest.approx(band / per_geo_mean, rel=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_candidate_sets_leave_a_control_geo(history: pl.DataFrame) -> None:
    ms = _selection(history)
    with pytest.raises(ValueError, match="No valid candidate sizes"):
        ms._candidate_sets(n_test_geos=[len(ms.all_geos)], include=[], exclude=[])
    # Excluded geos are still controls, so every non-excluded geo may be tested.
    cands = ms._candidate_sets(n_test_geos=[len(ms.all_geos) - 1], include=[], exclude=["g0"])
    assert cands == [tuple(g for g in ms.all_geos if g != "g0")]
