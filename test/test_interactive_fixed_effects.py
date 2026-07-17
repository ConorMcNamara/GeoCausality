"""Tests for the Interactive Fixed Effects (Bai 2009) estimator.

``InteractiveFixedEffects`` fits a treatment coefficient and a latent factor
structure jointly from the full panel by alternating least squares, then reports
the treated counterfactual (outcome with the treatment effect removed). These
tests assert the conformal-inference contract shared by the counterfactual
estimators, the factor-count behaviour (auto-selected via eigenvalue-ratio or
fixed), and that the method recovers a known effect both when geos share
heterogeneous latent factors and under a plain two-way fixed effects regime.
"""

import io
from contextlib import redirect_stdout
from datetime import date, timedelta

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest

from GeoCausality.interactive_fixed_effects import InteractiveFixedEffects

N_DATES = 40
PRE_PERIOD = "2021-01-30"  # first 30 days are pre-period
POST_PERIOD = "2021-01-31"  # final 10 days are post-period
TEST_GEOS = ("g0", "g1")
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
    factor (plus additive effects) suffices and the interactive method should
    behave like classic parallel-trends difference-in-differences.

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


def _model(df: pl.DataFrame, **overrides: object) -> InteractiveFixedEffects:
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
    return InteractiveFixedEffects(df, **kwargs)


@pytest.fixture(scope="module")
def effect_data() -> pl.DataFrame:
    return _make_data(effect=50.0)


@pytest.fixture(scope="module")
def null_data() -> pl.DataFrame:
    return _make_data(effect=0.0)


class TestConformalInference:
    @staticmethod
    def test_effect_detected_and_beats_null(effect_data: pl.DataFrame, null_data: pl.DataFrame) -> None:
        # Strong effect is significant (p<=0.1, CI>0) and at least as significant as the null panel.
        results = _model(effect_data).pre_process().generate().results
        assert results["p_value"] <= 0.1, "failed to detect a strong effect"
        assert results["incrementality_ci_lower"] > 0.0
        p_null = _model(null_data).pre_process().generate().results["p_value"]
        assert results["p_value"] <= p_null

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
        model = _model(effect_data, max_factors=5).pre_process().generate()
        assert model.n_factors_selected is not None
        assert 0 <= model.n_factors_selected <= 5
        assert model.results["n_factors"] == model.n_factors_selected

    @staticmethod
    def test_fixed_n_factors_is_respected(effect_data: pl.DataFrame) -> None:
        model = _model(effect_data, n_factors=2).pre_process().generate()
        assert model.n_factors_selected == 2

    @staticmethod
    def test_negative_n_factors_raises(effect_data: pl.DataFrame) -> None:
        with pytest.raises(ValueError):
            _model(effect_data, n_factors=-1)

    @staticmethod
    def test_zero_factors_is_two_way_fixed_effects(effect_data: pl.DataFrame) -> None:
        # With no factors the model reduces to additive two-way fixed effects; it
        # should still run and produce a finite incrementality.
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

    @staticmethod
    def test_recovers_known_effect_with_latent_factors() -> None:
        # Heterogeneous factor loadings: two-way fixed effects would be biased,
        # but the interactive model recovers the injected effect.
        df = _make_data(effect=50.0, seed=3, n_factors=2)
        results = _model(df).pre_process().generate().results
        expected = 2 * 10 * 50.0
        assert results["incrementality"] == pytest.approx(expected, rel=0.2)


class TestMethods:
    @staticmethod
    def test_method_param_contract(effect_data: pl.DataFrame) -> None:
        # Default mode is projection; an unknown method raises.
        assert _model(effect_data).method == "projection"
        with pytest.raises(ValueError, match="method"):
            _model(effect_data, method="bogus")

    @staticmethod
    @pytest.mark.parametrize("method", ["projection", "coefficient"])
    def test_both_methods_recover_effect(method: str, effect_data: pl.DataFrame) -> None:
        # On this well-identified panel (2 of 10 geos treated, 2 true factors) both
        # the control-only projection and the full-panel Bai coefficient recover
        # the injected effect (2 geos x 10 days x 50.0 = 1000).
        results = _model(effect_data, method=method).pre_process().generate().results
        assert results["incrementality"] == pytest.approx(1000.0, rel=0.2)
        assert results["incrementality_ci_lower"] <= results["incrementality"] <= results["incrementality_ci_upper"]

    @staticmethod
    def test_coefficient_mode_uses_treated_in_factors(effect_data: pl.DataFrame) -> None:
        # The two modes estimate the counterfactual differently, so their fitted
        # counterfactuals differ even when both recover a similar total effect.
        proj = _model(effect_data, method="projection").pre_process().generate()
        coef = _model(effect_data, method="coefficient").pre_process().generate()
        assert not np.allclose(
            proj.prediction_post[proj.y_variable].to_numpy(),
            coef.prediction_post[coef.y_variable].to_numpy(),
        )


class TestAtt:
    @staticmethod
    def test_att_times_treated_cells_equals_incrementality(effect_data: pl.DataFrame) -> None:
        # The aggregated incrementality equals the per-cell ATT times the number
        # of treated post-period cells (2 test geos x 10 post days).
        model = _model(effect_data).pre_process().generate()
        assert model.att is not None
        assert model.att * 2 * 10 == pytest.approx(model.results["incrementality"], rel=1e-6)


class TestPlot:
    @staticmethod
    def test_plot_builds_figure(effect_data: pl.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = {}
        monkeypatch.setattr(go.Figure, "show", lambda self: captured.update(fig=self))
        _model(effect_data).pre_process().generate().plot()
        assert "fig" in captured

    @staticmethod
    def test_plot_renders_confidence_bands(effect_data: pl.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
        # Regression: IFE now inherits the base counterfactual plot, which shades a
        # confidence band on each of the three panels (its own plot dropped them).
        captured = {}
        monkeypatch.setattr(go.Figure, "show", lambda self: captured.update(fig=self))
        _model(effect_data).pre_process().generate().plot()
        band_traces = [trace for trace in captured["fig"].data if trace.fill == "tonexty"]
        assert len(band_traces) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
