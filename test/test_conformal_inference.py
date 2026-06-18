"""Tests for the conformal inference added to the Synthetic Control estimators.

Every Synthetic Control method routes through ``EconometricEstimator`` and, after
``generate()``, exposes a conformal ``p_value``, confidence intervals for the
incrementality / per-period lift, and a split-conformal prediction band. These
tests assert the structural contract of those keys and their qualitative
behaviour (a strong treatment effect should be more significant than no effect).
"""

import io
from contextlib import redirect_stdout
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from GeoCausality.augmented_synthetic_control import AugmentedSyntheticControl
from GeoCausality.penalized_synthetic_control import PenalizedSyntheticControl
from GeoCausality.robust_synthetic_control import RobustSyntheticControl
from GeoCausality.synthetic_control import SyntheticControl, SyntheticControlV

N_DATES = 40
N_POST = 10
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


def _make_data(effect: float, seed: int = 0) -> pl.DataFrame:
    """Builds a geo panel where controls share latent structure with the test geos.

    Parameters
    ----------
    effect : float
        The per-geo, per-day additive treatment effect applied to the test geos
        during the post-period.
    seed : int
        Seed for reproducibility; the estimators and the conformal routines are
        otherwise deterministic, so a fixed seed makes the tests non-flaky.
    """
    rng = np.random.default_rng(seed)
    d0 = date(2021, 1, 1)
    dates = [d0 + timedelta(days=i) for i in range(N_DATES)]
    geos = [f"g{i}" for i in range(10)]

    # Two shared latent factors so a synthetic control can reconstruct the test
    # geos from the controls.
    factors = rng.normal(0, 1, size=(2, N_DATES)).cumsum(axis=1)
    loadings = {g: rng.uniform(0.5, 2.0, size=2) for g in geos}

    rows = []
    for g in geos:
        is_test = g in TEST_GEOS
        for di, d in enumerate(dates):
            y = 100.0 + loadings[g] @ factors[:, di] + rng.normal(0, 2)
            if is_test and d >= date.fromisoformat(POST_PERIOD):
                y += effect
            rows.append({"geo": g, "date": d, "y": float(y), "is_treatment": int(is_test)})
    return pl.DataFrame(rows)


# Factories keyed by name so failures point at the offending estimator. Each takes
# the panel and returns a configured (but not yet run) estimator.
MODEL_FACTORIES = {
    "SyntheticControl": lambda df: SyntheticControl(
        df,
        geo_variable="geo",
        treatment_variable="is_treatment",
        date_variable="date",
        pre_period=PRE_PERIOD,
        post_period=POST_PERIOD,
        y_variable="y",
        msrp=7.0,
        spend=10_000,
    ),
    "SyntheticControlV": lambda df: SyntheticControlV(
        df,
        geo_variable="geo",
        treatment_variable="is_treatment",
        date_variable="date",
        pre_period=PRE_PERIOD,
        post_period=POST_PERIOD,
        y_variable="y",
        msrp=7.0,
        spend=10_000,
    ),
    "PenalizedSyntheticControl": lambda df: PenalizedSyntheticControl(
        df,
        geo_variable="geo",
        treatment_variable="is_treatment",
        date_variable="date",
        pre_period=PRE_PERIOD,
        post_period=POST_PERIOD,
        y_variable="y",
        msrp=7.0,
        spend=10_000,
    ),
    "AugmentedSyntheticControl": lambda df: AugmentedSyntheticControl(
        df,
        geo_variable="geo",
        treatment_variable="is_treatment",
        date_variable="date",
        pre_period=PRE_PERIOD,
        post_period=POST_PERIOD,
        y_variable="y",
        msrp=7.0,
        spend=10_000,
    ),
    "RobustSyntheticControl": lambda df: RobustSyntheticControl(
        df,
        geo_variable="geo",
        treatment_variable="is_treatment",
        date_variable="date",
        pre_period=PRE_PERIOD,
        post_period=POST_PERIOD,
        y_variable="y",
        msrp=7.0,
        spend=10_000,
        sv_count=3,
    ),
}
MODEL_NAMES = list(MODEL_FACTORIES)


@pytest.fixture(scope="module")
def effect_data() -> pl.DataFrame:
    return _make_data(effect=50.0)


@pytest.fixture(scope="module")
def null_data() -> pl.DataFrame:
    return _make_data(effect=0.0)


def _run(name: str, df: pl.DataFrame):
    return MODEL_FACTORIES[name](df).pre_process().generate()


class TestConformalInference:
    @staticmethod
    @pytest.mark.parametrize("name", MODEL_NAMES)
    def test_results_contain_conformal_keys(name: str, effect_data: pl.DataFrame) -> None:
        results = _run(name, effect_data).results
        for key in CONFORMAL_KEYS:
            assert key in results, f"{name} missing conformal key {key!r}"
        assert 0.0 <= results["p_value"] <= 1.0
        assert results["conformal_band"] >= 0.0
        assert results["incrementality_ci_lower"] <= results["incrementality_ci_upper"]

    @staticmethod
    @pytest.mark.parametrize("name", MODEL_NAMES)
    def test_point_estimate_within_interval(name: str, effect_data: pl.DataFrame) -> None:
        # The interval is centred on the observed mean lift, so the point estimate
        # is inside it by construction.
        results = _run(name, effect_data).results
        assert results["incrementality_ci_lower"] <= results["incrementality"] <= results["incrementality_ci_upper"]

    @staticmethod
    @pytest.mark.parametrize("name", MODEL_NAMES)
    def test_strong_effect_is_significant(name: str, effect_data: pl.DataFrame) -> None:
        # A large positive effect should reject the no-effect null, so zero falls
        # below the confidence interval and the p-value is small.
        results = _run(name, effect_data).results
        assert results["p_value"] <= 0.1, f"{name} failed to detect a strong effect"
        assert results["incrementality_ci_lower"] > 0.0

    @staticmethod
    @pytest.mark.parametrize("name", MODEL_NAMES)
    def test_effect_more_significant_than_null(name: str, effect_data: pl.DataFrame, null_data: pl.DataFrame) -> None:
        p_effect = _run(name, effect_data).results["p_value"]
        p_null = _run(name, null_data).results["p_value"]
        assert p_effect <= p_null, f"{name}: effect p={p_effect} should be <= null p={p_null}"

    @staticmethod
    @pytest.mark.parametrize("name", MODEL_NAMES)
    @pytest.mark.parametrize("lift", LIFT_TYPES)
    def test_summarize_runs_for_all_lift_types(name: str, lift: str, effect_data: pl.DataFrame) -> None:
        model = _run(name, effect_data)
        with redirect_stdout(io.StringIO()) as buffer:
            model.summarize(lift)
        out = buffer.getvalue()
        assert "p_value" in out
        assert "CI" in out  # confidence-interval columns are rendered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
