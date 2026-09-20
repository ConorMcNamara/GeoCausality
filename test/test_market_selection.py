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
    @staticmethod
    def test_rankings_structure(searched_size2: MarketSelection) -> None:
        assert searched_size2.rankings is not None
        assert len(searched_size2.rankings) == 28  # C(8,2)
        for r in searched_size2.rankings:
            assert {"test_geos", "n_test", "n_control", "power", "pre_fit", "score"} <= r.keys()
            assert r["n_test"] + r["n_control"] == N_GEOS
            assert 0.0 <= r["power"] <= 1.0

    @staticmethod
    def test_rankings_sorted_by_score_desc(searched_sizes_1_2: MarketSelection) -> None:
        scores = [r["score"] for r in searched_sizes_1_2.rankings]
        assert scores == sorted(scores, reverse=True)

    @staticmethod
    def test_pre_fit_computed_for_synthetic_control(searched_size2: MarketSelection) -> None:
        assert all(r["pre_fit"] is not None and r["pre_fit"] >= 0.0 for r in searched_size2.rankings)

    @staticmethod
    def test_fit_weight_zero_ranks_on_power(history: pl.DataFrame) -> None:
        ms = _selection(history, fit_weight=0.0).search(n_test_geos=[2], effect_size=0.3, duration=10, n_sims=8)
        assert all(r["score"] == r["power"] for r in ms.rankings)

    @staticmethod
    def test_invalid_fit_weight_raises(history: pl.DataFrame) -> None:
        with pytest.raises(ValueError):
            _selection(history, fit_weight=1.5)


class TestReporting:
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
