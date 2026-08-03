"""Tests for the Diff-in-Diff estimator.

Diff-in-Diff groups by ``(treatment, treatment_period, date)`` and sums ``y``
across geos before fitting ``y ~ treatment * treatment_period``. The interaction
coefficient is therefore the treatment effect on the geo-*summed* series, so a
recovered ``lift`` is expected to be roughly ``n_test * effect``.
"""

import io
from contextlib import redirect_stdout

import pytest

from GeoCausality.diff_in_diff import DiffinDiff

# Every lift type Diff-in-Diff accepts (note: includes "relative").
LIFT_TYPES = ("incremental", "absolute", "relative", "cost-per", "revenue", "roas")


def _fit(panel) -> DiffinDiff:
    return DiffinDiff(panel.df, **panel.kwargs()).pre_process().generate()


class TestDiffinDiff:
    @staticmethod
    def test_pre_process_counts_post_dates(effect_panel) -> None:
        model = DiffinDiff(effect_panel.df, **effect_panel.kwargs()).pre_process()
        assert model.n_dates == effect_panel.n_post_dates

    @staticmethod
    def test_recovers_summed_effect(effect_panel) -> None:
        # lift is the effect on the summed series ~ n_test * per-geo effect.
        results = _fit(effect_panel).results
        expected = effect_panel.n_test * effect_panel.effect
        assert results["lift"] == pytest.approx(expected, rel=0.2)

    @staticmethod
    def test_incrementality_identity(effect_panel) -> None:
        model = _fit(effect_panel)
        assert model.results["incrementality"] == pytest.approx(model.results["lift"] * model.n_dates)
        assert model.results["incrementality_ci_lower"] == pytest.approx(model.results["lift_ci_lower"] * model.n_dates)
        assert model.results["incrementality_ci_upper"] == pytest.approx(model.results["lift_ci_upper"] * model.n_dates)

    @staticmethod
    def test_effect_vs_null_significance(effect_panel, null_panel) -> None:
        # A clear effect is significant with a positive interval; the null is not.
        effect_results = _fit(effect_panel).results
        p_null = _fit(null_panel).results["p_value"]
        assert effect_results["p_value"] < 0.05
        assert effect_results["incrementality_ci_lower"] > 0.0
        assert effect_results["p_value"] < p_null
        assert p_null > 0.1  # no effect should not look significant

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
    def test_summarize_rejects_unknown_lift(effect_panel) -> None:
        model = _fit(effect_panel)
        with pytest.raises(ValueError, match="Cannot measure blarg"):
            model.summarize("blarg")

    @staticmethod
    def test_test_geos_argument_matches_treatment_variable(panel_factory) -> None:
        # Specifying test_geos rebuilds the treatment column and should agree with
        # the pre-baked is_treatment column.
        panel = panel_factory(effect=8.0)
        via_column = DiffinDiff(panel.df, **panel.kwargs()).pre_process().generate()
        via_geos = (
            DiffinDiff(
                panel.df,
                test_geos=[f"g{i}" for i in range(panel.n_test)],
                **panel.kwargs(),
            )
            .pre_process()
            .generate()
        )
        assert via_geos.results["lift"] == pytest.approx(via_column.results["lift"], rel=1e-6)

    @staticmethod
    @pytest.mark.parametrize("inference", ("conformal", "jackknife", "auto"))
    def test_conformal_inference_keeps_ols_point_estimate(inference: str, effect_panel) -> None:
        # The point estimate is always the OLS coefficient; only the inference
        # (p-value / CIs) changes, so lift must match the default "ols" path.
        ols = _fit(effect_panel)
        conf = DiffinDiff(effect_panel.df, **effect_panel.kwargs(), inference=inference).pre_process().generate()
        assert conf.results["lift"] == pytest.approx(ols.results["lift"], rel=1e-9)
        assert conf.results["incrementality"] == pytest.approx(conf.results["lift"] * conf.n_dates)
        assert conf.results["method"] in ("conformal", "jackknife+", "jackknife+ (residual)")
        # The conformal CI is centred on the same effect and brackets it.
        assert conf.results["lift_ci_lower"] <= conf.results["lift"] <= conf.results["lift_ci_upper"]
        assert conf.results["incrementality_ci_lower"] == pytest.approx(conf.results["lift_ci_lower"] * conf.n_dates)

    @staticmethod
    def test_conformal_effect_vs_null_significance(effect_panel, null_panel) -> None:
        effect = DiffinDiff(effect_panel.df, **effect_panel.kwargs(), inference="conformal").pre_process().generate()
        null = DiffinDiff(null_panel.df, **null_panel.kwargs(), inference="conformal").pre_process().generate()
        assert effect.results["p_value"] < null.results["p_value"]
        assert effect.results["incrementality_ci_lower"] > 0.0

    @staticmethod
    def test_rejects_unknown_inference(effect_panel) -> None:
        with pytest.raises(ValueError, match="inference must be one of"):
            DiffinDiff(effect_panel.df, **effect_panel.kwargs(), inference="bogus")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
