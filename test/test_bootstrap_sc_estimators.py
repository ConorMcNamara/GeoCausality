"""Parametric-bootstrap inference across the synthetic-control estimator family.

Every synthetic-control estimator caches its donor matrices and implements
``_fit_predict_weights`` (from the faithful jackknife+ work), so the shared
``_bootstrap_refit`` delegates to it and they all support
``inference_method = "bootstrap"`` -- not just ``GeneralizedSyntheticControl``.
These tests assert the bootstrap path activates, brackets the estimate, is
reproducible, and that the default ``"auto"`` behaviour is unchanged.
"""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from GeoCausality.augmented_synthetic_control import AugmentedSyntheticControl
from GeoCausality.elastic_net_synthetic_control import ElasticNetSyntheticControl
from GeoCausality.penalized_synthetic_control import PenalizedSyntheticControl
from GeoCausality.robust_synthetic_control import RobustSyntheticControl
from GeoCausality.synthetic_control import SyntheticControl, SyntheticControlV

N_DATES = 40
PRE_PERIOD = "2021-01-30"
POST_PERIOD = "2021-01-31"
TEST_GEOS = ("g0", "g1")

MODELS = [
    SyntheticControl,
    SyntheticControlV,
    PenalizedSyntheticControl,
    RobustSyntheticControl,
    AugmentedSyntheticControl,
    ElasticNetSyntheticControl,
]
MODEL_IDS = [c.__name__ for c in MODELS]


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


def _build(cls, df: pl.DataFrame, **overrides: object):
    kwargs = {
        "treatment_variable": "is_treatment",
        "pre_period": PRE_PERIOD,
        "post_period": POST_PERIOD,
        "y_variable": "y",
    }
    if cls is RobustSyntheticControl:
        kwargs["sv_count"] = 3
    model = cls(df, **kwargs)
    model.inference_method = "bootstrap"
    model.n_boot = 200
    for key, value in overrides.items():
        setattr(model, key, value)
    return model


@pytest.fixture(scope="module")
def effect_data() -> pl.DataFrame:
    return _make_data(effect=50.0)


class TestBootstrapAcrossEstimators:
    @staticmethod
    @pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
    def test_bootstrap_path_and_ci(cls, effect_data: pl.DataFrame) -> None:
        # Each estimator's weight hook drives the parametric bootstrap: the method
        # activates and its interval brackets the estimate. Reproducibility and the
        # default "auto" behaviour are covered by test_parametric_bootstrap.py.
        results = _build(cls, effect_data, n_boot=300).pre_process().generate().results
        assert results["method"] == "bootstrap", f"{cls.__name__} did not use the bootstrap"
        assert results["incrementality_ci_lower"] <= results["incrementality"] <= results["incrementality_ci_upper"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
