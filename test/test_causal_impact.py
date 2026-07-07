"""Validation of the CausalImpact estimator against its canonical simulated example.

Brodersen et al. validate CausalImpact on a simulated series built from an AR(1)
control: ``y = 1.2 * x1 + noise`` over 100 days, with a known additive effect
injected into the final 30 days (the post-period). The estimator should recover the
injected effect, bracket it with its confidence interval, and -- crucially for a
causal method -- report *no* significant effect when none was injected.

These tests reproduce that setup in the geo-panel format the estimator consumes: the
treated series is one geo, the AR(1) control is a donor geo.
"""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from GeoCausality.causal_impact import CausalImpact

N_DATES = 100
POST_START = 70  # first post-period index; pre = [0, 69], post = [70, 99]
TRUE_EFFECT = 10.0  # per-period additive effect injected post-period
CUM_EFFECT = TRUE_EFFECT * (N_DATES - POST_START)  # 300 over 30 post-period days


def _dates() -> list[date]:
    d0 = date(2021, 1, 1)
    return [d0 + timedelta(days=i) for i in range(N_DATES)]


def _make_data(effect: float, seed: int = 1) -> pl.DataFrame:
    """Build CausalImpact's canonical simulated panel with a known post-period effect.

    Parameters
    ----------
    effect : float
        Per-day additive effect added to the treated series over the post-period.
    seed : int
        Seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    # AR(1) control with coefficient near 1 (a near-random walk), as in the
    # CausalImpact documentation's ``arima.sim(model = list(ar = 0.999))``.
    noise = rng.normal(0, 1, N_DATES)
    x1 = np.empty(N_DATES)
    x1[0] = noise[0]
    for t in range(1, N_DATES):
        x1[t] = 0.999 * x1[t - 1] + noise[t]
    x1 = 100 + x1
    y = 1.2 * x1 + rng.normal(0, 1, N_DATES)
    y[POST_START:] += effect

    dates = _dates()
    rows = []
    for i, dt in enumerate(dates):
        rows.append({"geo": "t0", "date": dt, "y": float(y[i]), "is_treatment": 1})
        rows.append({"geo": "c0", "date": dt, "y": float(x1[i]), "is_treatment": 0})
    return pl.DataFrame(rows)


def _build(df: pl.DataFrame, **overrides: object) -> CausalImpact:
    dates = _dates()
    model = CausalImpact(
        df,
        treatment_variable="is_treatment",
        pre_period=dates[POST_START - 1].isoformat(),
        post_period=dates[POST_START].isoformat(),
        y_variable="y",
    )
    for key, value in overrides.items():
        setattr(model, key, value)
    return model


@pytest.fixture(scope="module")
def effect_results() -> dict:
    return _build(_make_data(TRUE_EFFECT)).pre_process().generate().results


@pytest.fixture(scope="module")
def null_results() -> dict:
    return _build(_make_data(0.0)).pre_process().generate().results


class TestRecoversKnownEffect:
    """The injected effect is recovered, bracketed, and flagged significant."""

    @staticmethod
    def test_point_estimate_near_truth(effect_results: dict) -> None:
        # Within 15% of the injected cumulative effect of 300.
        assert effect_results["incrementality"] == pytest.approx(CUM_EFFECT, rel=0.15)

    @staticmethod
    def test_average_lift_near_truth(effect_results: dict) -> None:
        assert float(np.mean(effect_results["lift"])) == pytest.approx(TRUE_EFFECT, abs=2.0)

    @staticmethod
    def test_ci_brackets_truth(effect_results: dict) -> None:
        assert effect_results["incrementality_ci_lower"] <= CUM_EFFECT <= effect_results["incrementality_ci_upper"]

    @staticmethod
    def test_ci_brackets_estimate(effect_results: dict) -> None:
        assert (
            effect_results["incrementality_ci_lower"]
            <= effect_results["incrementality"]
            <= effect_results["incrementality_ci_upper"]
        )

    @staticmethod
    def test_effect_is_significant(effect_results: dict) -> None:
        assert effect_results["p_value"] < 0.05

    @staticmethod
    def test_method_label(effect_results: dict) -> None:
        assert effect_results["method"] == "structural-ts"


class TestNullProducesNoFalsePositive:
    """With no injected effect the method must not manufacture one."""

    @staticmethod
    def test_effect_not_significant(null_results: dict) -> None:
        assert null_results["p_value"] > 0.05

    @staticmethod
    def test_ci_brackets_zero(null_results: dict) -> None:
        assert null_results["incrementality_ci_lower"] <= 0 <= null_results["incrementality_ci_upper"]

    @staticmethod
    def test_estimate_near_zero(null_results: dict) -> None:
        # No effect, so the cumulative estimate should be small relative to the
        # signal it would carry under the alternative (300).
        assert abs(null_results["incrementality"]) < 0.5 * CUM_EFFECT


class TestPosteriorSimulation:
    """The simulate()-based cumulative-effect distribution is reproducible."""

    @staticmethod
    def test_reproducible_with_seed() -> None:
        df = _make_data(TRUE_EFFECT)
        a = _build(df, sim_seed=7).pre_process().generate().results
        b = _build(df, sim_seed=7).pre_process().generate().results
        assert a["incrementality_ci_lower"] == b["incrementality_ci_lower"]
        assert a["incrementality_ci_upper"] == b["incrementality_ci_upper"]
        assert a["p_value"] == b["p_value"]

    @staticmethod
    def test_more_draws_still_brackets_truth() -> None:
        results = _build(_make_data(TRUE_EFFECT), n_sim=2000).pre_process().generate().results
        assert results["incrementality_ci_lower"] <= CUM_EFFECT <= results["incrementality_ci_upper"]


class TestInferenceRouting:
    """The conformal fallback stays available for parity with the SC family."""

    @staticmethod
    def test_conformal_route(effect_results: dict) -> None:
        results = _build(_make_data(TRUE_EFFECT), inference_method="conformal").pre_process().generate().results
        assert results["method"] == "conformal"
        assert results["incrementality_ci_lower"] <= results["incrementality"] <= results["incrementality_ci_upper"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
