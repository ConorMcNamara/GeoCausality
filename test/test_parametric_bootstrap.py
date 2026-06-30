"""Tests for the parametric-bootstrap inference path.

Setting ``inference_method = "bootstrap"`` makes an estimator perform inference by
parametric bootstrap (GeoLift's GSC-style approach): hold the counterfactual
fixed, draw parametric noise at the pre-period residual scale, refit via
``_bootstrap_refit``, and form an incrementality distribution. These tests cover
the path on ``GeneralizedSyntheticControl`` (which implements the hook), the
fallback for estimators without it, reproducibility, and that the default
``"auto"`` behaviour is unchanged.
"""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from GeoCausality.generalized_synthetic_control import GeneralizedSyntheticControl
from GeoCausality.synthetic_control import SyntheticControl

N_DATES = 40
PRE_PERIOD = "2021-01-30"
POST_PERIOD = "2021-01-31"
TEST_GEOS = ("g0", "g1")


def _make_data(effect: float, seed: int = 0) -> pl.DataFrame:
    """Builds a factor-model geo panel with a known post-period effect.

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


def _gsc(df: pl.DataFrame, **overrides: object) -> GeneralizedSyntheticControl:
    model = GeneralizedSyntheticControl(
        df,
        treatment_variable="is_treatment",
        pre_period=PRE_PERIOD,
        post_period=POST_PERIOD,
        spend=10_000,
        msrp=7.0,
    )
    model.inference_method = "bootstrap"
    for key, value in overrides.items():
        setattr(model, key, value)
    return model


@pytest.fixture(scope="module")
def effect_data() -> pl.DataFrame:
    return _make_data(effect=50.0)


@pytest.fixture(scope="module")
def null_data() -> pl.DataFrame:
    return _make_data(effect=0.0)


class TestBootstrapPath:
    @staticmethod
    def test_method_is_bootstrap(effect_data: pl.DataFrame) -> None:
        results = _gsc(effect_data, n_boot=200).pre_process().generate().results
        assert results["method"] == "bootstrap"

    @staticmethod
    def test_ci_brackets_point_estimate(effect_data: pl.DataFrame) -> None:
        results = _gsc(effect_data, n_boot=400).pre_process().generate().results
        assert results["incrementality_ci_lower"] <= results["incrementality"] <= results["incrementality_ci_upper"]
        assert results["incrementality_ci_lower"] <= results["incrementality_ci_upper"]

    @staticmethod
    def test_lift_and_incrementality_consistent(effect_data: pl.DataFrame) -> None:
        results = _gsc(effect_data, n_boot=200).pre_process().generate().results
        t1 = 10
        assert results["incrementality_ci_lower"] == pytest.approx(results["lift_ci_lower"] * t1)
        assert results["incrementality_ci_upper"] == pytest.approx(results["lift_ci_upper"] * t1)

    @staticmethod
    def test_strong_effect_is_significant(effect_data: pl.DataFrame) -> None:
        results = _gsc(effect_data, n_boot=400).pre_process().generate().results
        assert results["p_value"] <= 0.1

    @staticmethod
    def test_effect_more_significant_than_null(effect_data: pl.DataFrame, null_data: pl.DataFrame) -> None:
        p_effect = _gsc(effect_data, n_boot=400).pre_process().generate().results["p_value"]
        p_null = _gsc(null_data, n_boot=400).pre_process().generate().results["p_value"]
        assert p_effect <= p_null


class TestReproducibility:
    @staticmethod
    def test_same_seed_same_result(effect_data: pl.DataFrame) -> None:
        a = _gsc(effect_data, n_boot=200, bootstrap_seed=7).pre_process().generate().results
        b = _gsc(effect_data, n_boot=200, bootstrap_seed=7).pre_process().generate().results
        assert a["p_value"] == b["p_value"]
        assert a["incrementality_ci_lower"] == b["incrementality_ci_lower"]
        assert a["incrementality_ci_upper"] == b["incrementality_ci_upper"]


class TestFallback:
    @staticmethod
    def test_estimator_without_hook_falls_back(effect_data: pl.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
        # Synthetic-control estimators now support the bootstrap via the shared
        # weight hook; when that hook is unavailable, "bootstrap" must not crash —
        # it falls through to the conformal / jackknife path instead.
        monkeypatch.setattr(SyntheticControl, "_fit_predict_weights", lambda self, x_train, y_train, x_eval: None)
        model = SyntheticControl(
            effect_data, treatment_variable="is_treatment", pre_period=PRE_PERIOD, post_period=POST_PERIOD
        )
        model.inference_method = "bootstrap"
        results = model.pre_process().generate().results
        assert results["method"] != "bootstrap"
        assert "p_value" in results


class TestDefaultUnaffected:
    @staticmethod
    def test_auto_still_conformal_on_long_pre(effect_data: pl.DataFrame) -> None:
        model = GeneralizedSyntheticControl(
            effect_data, treatment_variable="is_treatment", pre_period=PRE_PERIOD, post_period=POST_PERIOD
        )
        results = model.pre_process().generate().results
        assert results["method"] == "conformal"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
