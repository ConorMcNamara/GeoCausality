"""Tests for the FixedEffects estimator.

FixedEffects fits a two-way (entity + time) fixed-effects panel regression on the
full geo-level data, so the ``campaign_treatment`` coefficient is the per-geo,
per-day treatment effect and ``lift`` should recover the effect directly. Unlike
Diff-in-Diff, FixedEffects does not support a ``"relative"`` lift.
"""

import io
from contextlib import redirect_stdout

import pytest

from GeoCausality.fixed_effects import FixedEffects

# Every lift type FixedEffects accepts (note: "relative" is intentionally absent).
LIFT_TYPES = ("incremental", "absolute", "cost-per", "revenue", "roas")


def _fit(panel) -> FixedEffects:
    return FixedEffects(panel.df, **panel.kwargs()).pre_process().generate()


class TestFixedEffects:
    @staticmethod
    def test_pre_process_counts(effect_panel) -> None:
        model = FixedEffects(effect_panel.df, **effect_panel.kwargs()).pre_process()
        assert model.n_geos == effect_panel.n_test
        assert model.n_dates == effect_panel.n_post_dates

    @staticmethod
    def test_recovers_per_geo_effect(effect_panel) -> None:
        # The coefficient is the per-geo effect, so lift ~ effect (not summed).
        results = _fit(effect_panel).results
        assert results["lift"] == pytest.approx(effect_panel.effect, rel=0.2)

    @staticmethod
    def test_incrementality_identity(effect_panel) -> None:
        model = _fit(effect_panel)
        scale = model.n_dates * model.n_geos
        assert model.results["incrementality"] == pytest.approx(model.results["lift"] * scale)
        assert model.results["incrementality_ci_lower"] == pytest.approx(model.results["lift_ci_lower"] * scale)
        assert model.results["incrementality_ci_upper"] == pytest.approx(model.results["lift_ci_upper"] * scale)

    @staticmethod
    def test_effect_vs_null_significance(effect_panel, null_panel) -> None:
        # A clear effect is significant with a positive interval; the null is not.
        effect_results = _fit(effect_panel).results
        p_null = _fit(null_panel).results["p_value"]
        assert effect_results["p_value"] < 0.05
        assert effect_results["incrementality_ci_lower"] > 0.0
        assert effect_results["p_value"] < p_null
        assert p_null > 0.1

    @staticmethod
    @pytest.mark.parametrize("lift", LIFT_TYPES)
    def test_summarize_runs_for_all_lift_types(lift: str, effect_panel) -> None:
        model = _fit(effect_panel)
        with redirect_stdout(io.StringIO()) as buffer:
            model.summarize(lift)
        out = buffer.getvalue()
        assert "p_value" in out
        assert "CI" in out

    @staticmethod
    def test_summarize_rejects_relative(effect_panel) -> None:
        # FixedEffects does not implement a relative lift.
        model = _fit(effect_panel)
        with pytest.raises(ValueError, match="Cannot measure relative"):
            model.summarize("relative")

    @staticmethod
    def test_summarize_rejects_unknown_lift(effect_panel) -> None:
        model = _fit(effect_panel)
        with pytest.raises(ValueError, match="Cannot measure blarg"):
            model.summarize("blarg")

    @staticmethod
    @pytest.mark.parametrize("inference", ("conformal", "jackknife", "auto"))
    def test_conformal_inference_keeps_ols_point_estimate(inference: str, effect_panel) -> None:
        # The point estimate is always the fixed-effects coefficient; only the
        # inference (p-value / CIs) changes, so lift must match the "ols" path.
        ols = _fit(effect_panel)
        conf = FixedEffects(effect_panel.df, **effect_panel.kwargs(), inference=inference).pre_process().generate()
        assert conf.results["lift"] == pytest.approx(ols.results["lift"], rel=1e-9)
        scale = conf.n_dates * conf.n_geos
        assert conf.results["incrementality"] == pytest.approx(conf.results["lift"] * scale)
        assert conf.results["method"] in ("conformal", "jackknife+", "jackknife+ (residual)")
        # Incrementality CI stays on the coefficient scale (n_dates * n_geos), not
        # the engine's default per-period (* t1) scaling.
        assert conf.results["incrementality_ci_lower"] == pytest.approx(conf.results["lift_ci_lower"] * scale)
        assert conf.results["incrementality_ci_upper"] == pytest.approx(conf.results["lift_ci_upper"] * scale)

    @staticmethod
    def test_conformal_effect_vs_null_significance(effect_panel, null_panel) -> None:
        effect = FixedEffects(effect_panel.df, **effect_panel.kwargs(), inference="conformal").pre_process().generate()
        null = FixedEffects(null_panel.df, **null_panel.kwargs(), inference="conformal").pre_process().generate()
        assert effect.results["p_value"] < null.results["p_value"]
        assert effect.results["incrementality_ci_lower"] > 0.0

    @staticmethod
    def test_rejects_unknown_inference(effect_panel) -> None:
        with pytest.raises(ValueError, match="inference must be one of"):
            FixedEffects(effect_panel.df, **effect_panel.kwargs(), inference="bogus")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
