"""Tests for the Generalized Synthetic Control (interactive fixed effects) estimator.

``GeneralizedSyntheticControl`` learns latent time factors from the control geos,
recovers the treated unit's loadings from its pre-period, and projects the
counterfactual forward. These tests assert the conformal-inference contract
shared by every synthetic-control estimator (matching ``test_conformal_inference``),
the factor-count behaviour (auto-selected via eigenvalue-ratio or cross-validation,
or fixed), and that the method recovers a known effect under a parallel-trends
(two-way fixed effects) regime.
"""

import io
from contextlib import redirect_stdout
from datetime import date, timedelta

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest

from GeoCausality.generalized_synthetic_control import GeneralizedSyntheticControl

N_DATES = 40
PRE_PERIOD = "2021-01-30"  # first 30 days are pre-period
POST_PERIOD = "2021-01-31"  # final 10 days are post-period
TEST_GEOS = ("g0", "g1")
CONFORMAL_KEYS = (
    "p_value",
    "lift_ci_lower",
    "lift_ci_upper",
    "incrementality_ci_lower",
    "incrementality_ci_upper",
    "conformal_band",
)
LIFT_TYPES = ("incremental", "absolute", "relative", "revenue", "roas")


def _make_data(effect: float, seed: int = 0, n_factors: int = 2) -> pl.DataFrame:
    """Builds a geo panel whose geos share ``n_factors`` latent time factors.

    Parameters
    ----------
    effect : float
        Per-geo, per-day additive treatment effect on the test geos post-period.
    seed : int
        Seed for reproducibility.
    n_factors : int
        Number of shared latent factors driving the panel.
    """
    rng = np.random.default_rng(seed)
    d0 = date(2021, 1, 1)
    dates = [d0 + timedelta(days=i) for i in range(N_DATES)]
    geos = [f"g{i}" for i in range(10)]
    factors = rng.normal(0, 1, size=(n_factors, N_DATES)).cumsum(axis=1)
    loadings = {g: rng.uniform(0.5, 2.0, size=n_factors) for g in geos}

    rows = []
    for g in geos:
        is_test = g in TEST_GEOS
        for di, d in enumerate(dates):
            y = 100.0 + loadings[g] @ factors[:, di] + rng.normal(0, 2)
            if is_test and d >= date.fromisoformat(POST_PERIOD):
                y += effect
            rows.append({"geo": g, "date": d, "y": float(y), "is_treatment": int(is_test)})
    return pl.DataFrame(rows)


def _make_parallel_trends(effect: float, seed: int = 0) -> pl.DataFrame:
    """Builds a single-common-trend panel: the two-way fixed effects regime.

    Every geo shares one common time trend plus a geo-specific level, so a single
    factor (plus intercept) suffices and the generalized method should behave like
    classic parallel-trends difference-in-differences.

    Parameters
    ----------
    effect : float
        Per-geo, per-day additive treatment effect on the test geos post-period.
    seed : int
        Seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    d0 = date(2021, 1, 1)
    dates = [d0 + timedelta(days=i) for i in range(N_DATES)]
    geos = [f"g{i}" for i in range(10)]
    common_trend = rng.normal(0, 1, N_DATES).cumsum()

    rows = []
    for gi, g in enumerate(geos):
        is_test = g in TEST_GEOS
        level = 50.0 + 5.0 * gi
        for di, d in enumerate(dates):
            y = level + common_trend[di] + rng.normal(0, 0.5)
            if is_test and d >= date.fromisoformat(POST_PERIOD):
                y += effect
            rows.append({"geo": g, "date": d, "y": float(y), "is_treatment": int(is_test)})
    return pl.DataFrame(rows)


def _model(df: pl.DataFrame, **overrides: object) -> GeneralizedSyntheticControl:
    kwargs = {
        "geo_variable": "geo",
        "treatment_variable": "is_treatment",
        "date_variable": "date",
        "pre_period": PRE_PERIOD,
        "post_period": POST_PERIOD,
        "y_variable": "y",
        "msrp": 7.0,
        "spend": 10_000,
    }
    kwargs.update(overrides)
    return GeneralizedSyntheticControl(df, **kwargs)


@pytest.fixture(scope="module")
def effect_data() -> pl.DataFrame:
    return _make_data(effect=50.0)


@pytest.fixture(scope="module")
def null_data() -> pl.DataFrame:
    return _make_data(effect=0.0)


class TestConformalInference:
    @staticmethod
    def test_results_contain_conformal_keys(effect_data: pl.DataFrame) -> None:
        results = _model(effect_data).pre_process().generate().results
        for key in CONFORMAL_KEYS:
            assert key in results, f"missing conformal key {key!r}"
        assert 0.0 <= results["p_value"] <= 1.0
        assert results["conformal_band"] >= 0.0
        assert results["incrementality_ci_lower"] <= results["incrementality_ci_upper"]

    @staticmethod
    def test_point_estimate_within_interval(effect_data: pl.DataFrame) -> None:
        results = _model(effect_data).pre_process().generate().results
        assert results["incrementality_ci_lower"] <= results["incrementality"] <= results["incrementality_ci_upper"]

    @staticmethod
    def test_strong_effect_is_significant(effect_data: pl.DataFrame) -> None:
        results = _model(effect_data).pre_process().generate().results
        assert results["p_value"] <= 0.1, "failed to detect a strong effect"
        assert results["incrementality_ci_lower"] > 0.0

    @staticmethod
    def test_effect_more_significant_than_null(effect_data: pl.DataFrame, null_data: pl.DataFrame) -> None:
        p_effect = _model(effect_data).pre_process().generate().results["p_value"]
        p_null = _model(null_data).pre_process().generate().results["p_value"]
        assert p_effect <= p_null

    @staticmethod
    @pytest.mark.parametrize("lift", LIFT_TYPES)
    def test_summarize_runs_for_all_lift_types(lift: str, effect_data: pl.DataFrame) -> None:
        model = _model(effect_data).pre_process().generate()
        with redirect_stdout(io.StringIO()) as buffer:
            model.summarize(lift)
        out = buffer.getvalue()
        assert "p_value" in out
        assert "CI" in out

    @staticmethod
    def test_summarize_rejects_invalid_lift(effect_data: pl.DataFrame) -> None:
        model = _model(effect_data).pre_process().generate()
        with pytest.raises(ValueError):
            model.summarize("nonsense")


class TestFactorSelection:
    @staticmethod
    def test_default_selects_factor_count(effect_data: pl.DataFrame) -> None:
        # The default ("er") selects a factor count within the configured range.
        model = _model(effect_data, max_factors=5).pre_process().generate()
        assert model.factor_selection == "er"
        assert model.n_factors_selected is not None
        assert 0 <= model.n_factors_selected <= 5
        assert model.results["n_factors"] == model.n_factors_selected

    @staticmethod
    @pytest.mark.parametrize("selection", ["er", "cv"])
    def test_both_selectors_run_and_recover_effect(selection: str, effect_data: pl.DataFrame) -> None:
        # Both factor-selection strategies should run and recover the strong
        # injected effect on the well-specified synthetic panel.
        model = _model(effect_data, factor_selection=selection, max_factors=5).pre_process().generate()
        assert 0 <= model.n_factors_selected <= 5
        assert model.results["p_value"] <= 0.1
        assert model.results["incrementality_ci_lower"] > 0.0

    @staticmethod
    def test_invalid_factor_selection_raises(effect_data: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="factor_selection"):
            _model(effect_data, factor_selection="bogus")

    @staticmethod
    def test_fixed_n_factors_is_respected(effect_data: pl.DataFrame) -> None:
        model = _model(effect_data, n_factors=2).pre_process().generate()
        assert model.n_factors_selected == 2

    @staticmethod
    def test_negative_n_factors_raises(effect_data: pl.DataFrame) -> None:
        with pytest.raises(ValueError):
            _model(effect_data, n_factors=-1)

    @staticmethod
    def test_zero_factors_is_intercept_only(effect_data: pl.DataFrame) -> None:
        # With no factors the counterfactual is a flat pre-period mean; it should
        # still run and produce a finite incrementality.
        results = _model(effect_data, n_factors=0).pre_process().generate().results
        assert results["n_factors"] == 0
        assert np.isfinite(results["incrementality"])


class TestParallelTrendsReduction:
    @staticmethod
    def test_recovers_known_effect_under_parallel_trends() -> None:
        # One common trend => a single factor suffices and the method reduces to
        # the two-way fixed effects / parallel-trends case. Total injected effect
        # is 2 test geos x 10 post days x 8.0 = 160.
        df = _make_parallel_trends(effect=8.0, seed=1)
        results = _model(df).pre_process().generate().results
        expected = 2 * 10 * 8.0
        assert results["incrementality"] == pytest.approx(expected, rel=0.15)
        assert results["p_value"] <= 0.1


class TestPlot:
    @staticmethod
    def test_plot_builds_figure(effect_data: pl.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
        shown = {}
        monkeypatch.setattr(go.Figure, "show", lambda self: shown.setdefault("ok", True))
        _model(effect_data).pre_process().generate().plot()
        assert shown.get("ok") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
