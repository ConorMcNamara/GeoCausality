"""Tests for the Synthetic Difference-in-Differences (Arkhangelsky et al. 2021) estimator.

``SyntheticDiffInDiff`` fits non-negative, L2-penalized unit weights against the
treated *trend* (a free intercept absorbs the level gap, so donors need only move
parallel to the treated unit) plus non-negative time weights that focus the
pre-period comparison, and reports the scalar average treatment effect on the
treated with placebo-variance inference.

Like every synthetic-control estimator in the package, the treated geos are summed
into a single series compared against the individual donor geos, so these tests use
a single treated unit -- the setting the method (and the Prop 99 / reunification
benchmarks) targets -- where the treated series and the donor pool share a scale.
The suite asserts the results contract shared with the conformal estimators, that
the method recovers a known effect under a common-trend regime, and that the
placebo p-value separates a real effect from a null.
"""

import io
from contextlib import redirect_stdout
from datetime import date, timedelta

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest

from GeoCausality.synthetic_diff_in_diff import SyntheticDiffInDiff

N_DATES = 40
PRE_PERIOD = "2021-01-30"  # first 30 days are pre-period
POST_PERIOD = "2021-01-31"  # final 10 days are post-period
N_POST = 10
RESULT_KEYS = (
    "p_value",
    "lift_ci_lower",
    "lift_ci_upper",
    "incrementality_ci_lower",
    "incrementality_ci_upper",
    "conformal_band",
    "standard_error",
    "att",
)
LIFT_TYPES = ("incremental", "absolute", "relative", "revenue", "roas")


def _single_treated(effect: float, seed: int = 0, n_control: int = 6) -> pl.DataFrame:
    """Builds a common-trend panel with one treated geo and a comparable donor pool.

    Every geo shares one common time trend plus a geo-specific level, so a
    convex combination of donors (shifted by the unit fixed effect) reconstructs
    the treated trajectory -- the two-way fixed-effects regime SDID targets.

    Parameters
    ----------
    effect : float
        Per-day additive treatment effect on the treated geo in the post-period.
    seed : int
        Seed for reproducibility.
    n_control : int
        Number of donor geos.
    """
    rng = np.random.default_rng(seed)
    d0 = date(2021, 1, 1)
    dates = [d0 + timedelta(days=i) for i in range(N_DATES)]
    geos = ["t0"] + [f"c{i}" for i in range(n_control)]
    common_trend = rng.normal(0, 1, N_DATES).cumsum()
    post_start = date.fromisoformat(POST_PERIOD)

    rows = []
    for gi, g in enumerate(geos):
        is_test = g == "t0"
        level = 50.0 + 5.0 * gi
        for di, d in enumerate(dates):
            y = level + common_trend[di] + rng.normal(0, 0.5)
            if is_test and d >= post_start:
                y += effect
            rows.append({"geo": g, "date": d, "y": float(y), "is_treatment": int(is_test)})
    return pl.DataFrame(rows)


def _model(df: pl.DataFrame, **overrides: object) -> SyntheticDiffInDiff:
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
    return SyntheticDiffInDiff(df, **kwargs)


@pytest.fixture(scope="module")
def effect_data() -> pl.DataFrame:
    return _single_treated(effect=8.0)


@pytest.fixture(scope="module")
def null_data() -> pl.DataFrame:
    return _single_treated(effect=0.0)


class TestPlaceboInference:
    @staticmethod
    def test_results_contain_expected_keys(effect_data: pl.DataFrame) -> None:
        results = _model(effect_data).pre_process().generate().results
        for key in RESULT_KEYS:
            assert key in results, f"missing key {key!r}"
        assert results["method"] == "placebo"
        assert 0.0 <= results["p_value"] <= 1.0
        assert results["conformal_band"] >= 0.0
        assert results["incrementality_ci_lower"] <= results["incrementality_ci_upper"]

    @staticmethod
    def test_incrementality_matches_summed_lift(effect_data: pl.DataFrame) -> None:
        # The per-period counterfactual gap sums to the reported incrementality.
        results = _model(effect_data).pre_process().generate().results
        assert float(np.sum(results["lift"])) == pytest.approx(results["incrementality"])
        assert results["att"] * N_POST == pytest.approx(results["incrementality"])

    @staticmethod
    def test_effect_significant_and_beats_null(effect_data: pl.DataFrame, null_data: pl.DataFrame) -> None:
        # Strong effect is significant (p<=0.1, CI>0) and at least as significant as the null panel.
        results = _model(effect_data).pre_process().generate().results
        assert results["p_value"] <= 0.1, "failed to detect a strong effect"
        assert results["incrementality_ci_lower"] > 0.0
        p_null = _model(null_data).pre_process().generate().results["p_value"]
        assert results["p_value"] <= p_null


class TestEffectRecovery:
    @staticmethod
    def test_recovers_effect_with_default_zeta(effect_data: pl.DataFrame) -> None:
        # With zeta unset the estimator applies the Arkhangelsky et al. default and still
        # recovers the effect: one treated geo, 8.0/day over 10 post days => incrementality ~ 80.
        model = _model(effect_data).pre_process().generate()
        results = model.results
        assert model.zeta is None
        assert results["incrementality"] == pytest.approx(N_POST * 8.0, rel=0.2)
        assert results["att"] == pytest.approx(8.0, rel=0.2)

    @staticmethod
    def test_null_effect_is_near_zero(null_data: pl.DataFrame) -> None:
        results = _model(null_data).pre_process().generate().results
        assert abs(results["att"]) < 2.0


class TestZeta:
    @staticmethod
    def test_explicit_zeta_shrinks_weights_toward_uniform(effect_data: pl.DataFrame) -> None:
        # A large L2 penalty drives the unit weights toward the uniform simplex
        # point (minimum ``||w||^2``), so their spread collapses relative to the
        # default fit. (On a clean common-trend panel every donor carries the same
        # trend, so this barely moves the counterfactual -- the weights are what
        # respond to the penalty.)
        default = _model(effect_data).pre_process().generate()
        heavy = _model(effect_data, zeta=1_000.0).pre_process().generate()
        assert heavy.zeta == 1_000.0
        n = default.unit_weights.shape[0]
        assert float(np.var(heavy.unit_weights)) < float(np.var(default.unit_weights))
        assert np.allclose(heavy.unit_weights, 1.0 / n, atol=1e-3)

    @staticmethod
    def test_unit_and_time_weights_are_simplex(effect_data: pl.DataFrame) -> None:
        model = _model(effect_data).pre_process().generate()
        for weights in (model.unit_weights, model.time_weights):
            assert weights is not None
            assert np.all(weights >= -1e-8)
            assert float(np.sum(weights)) == pytest.approx(1.0, abs=1e-6)


class TestSummarize:
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


class TestPlot:
    @staticmethod
    def test_plot_renders_confidence_bands(effect_data: pl.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = {}
        monkeypatch.setattr(go.Figure, "show", lambda self: captured.update(fig=self))
        model = _model(effect_data).pre_process().generate()
        model.plot()
        fig = captured["fig"]
        # One shaded band per panel (top, middle, cumulative) via `fill="tonexty"`.
        band_traces = [t for t in fig.data if t.fill == "tonexty"]
        assert len(band_traces) == 3
        # The cumulative band's final point matches the reported incrementality CI.
        cumulative_lower = band_traces[-1]
        assert float(cumulative_lower.y[-1]) == pytest.approx(model.results["incrementality_ci_lower"], rel=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
