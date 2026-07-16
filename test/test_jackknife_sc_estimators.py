"""Faithful jackknife+ across the synthetic-control estimator family.

Every synthetic-control estimator now caches its donor matrices and implements
``_fit_predict_weights``, so on a short pre-period they use the faithful
refit-based jackknife+ (``results["method"] == "jackknife+"``) via the shared
``_block_loo`` loop -- not the residual-only approximation. These tests assert
that contract, the leave-one-out array shapes, that the interval brackets the
estimate, and that long pre-periods are unchanged (still ``"conformal"``).
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

TEST_GEOS = ("g0", "g1")
SHORT_PRE, SHORT_POST = 8, 6  # 8 pre-period days triggers the jackknife+ fallback
LONG_PRE, LONG_POST = 30, 10


def _panel(n_pre: int, n_post: int, effect: float, seed: int = 0, n_geos: int = 10) -> tuple[pl.DataFrame, str, str]:
    """Builds a factor-model panel and returns it with its pre/post boundaries.

    Parameters
    ----------
    n_pre, n_post : int
        Number of pre- and post-period dates.
    effect : float
        Per-geo, per-day additive treatment effect on the test geos post-period.
    seed : int
        Seed for reproducibility.
    n_geos : int
        Number of geos in the panel.
    """
    rng = np.random.default_rng(seed)
    total = n_pre + n_post
    d0 = date(2021, 1, 1)
    dates = [d0 + timedelta(days=i) for i in range(total)]
    geos = [f"g{i}" for i in range(n_geos)]
    factors = rng.normal(0, 1, size=(2, total)).cumsum(axis=1)
    loadings = {g: rng.uniform(0.5, 2.0, size=2) for g in geos}

    rows = []
    for g in geos:
        is_test = g in TEST_GEOS
        for di, d in enumerate(dates):
            y = 100.0 + loadings[g] @ factors[:, di] + rng.normal(0, 2)
            if is_test and di >= n_pre:
                y += effect
            rows.append({"geo": g, "date": d, "y": float(y), "is_treatment": int(is_test)})
    return pl.DataFrame(rows), dates[n_pre - 1].isoformat(), dates[n_pre].isoformat()


def _build(cls, df: pl.DataFrame, pre: str, post: str):
    kwargs = {"treatment_variable": "is_treatment", "pre_period": pre, "post_period": post, "y_variable": "y"}
    if cls is RobustSyntheticControl:
        kwargs["sv_count"] = 3
    return cls(df, **kwargs)


MODELS = [
    SyntheticControl,
    SyntheticControlV,
    PenalizedSyntheticControl,
    RobustSyntheticControl,
    AugmentedSyntheticControl,
    ElasticNetSyntheticControl,
]
MODEL_IDS = [c.__name__ for c in MODELS]


class TestFaithfulJackknife:
    @staticmethod
    @pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
    def test_short_pre_faithful_jackknife(cls) -> None:
        # On a short pre-period each estimator's weight hook drives the faithful
        # refit-based jackknife+: correct method, well-shaped leave-one-out arrays,
        # and an interval that brackets the estimate. (The mechanism itself is
        # covered in depth by test_jackknife_fallback.py.)
        df, pre, post = _panel(SHORT_PRE, SHORT_POST, effect=40.0)
        model = _build(cls, df, pre, post).pre_process().generate()
        results = model.results
        assert results["method"] == "jackknife+", f"{cls.__name__} did not use faithful jackknife+"
        loo = model._loo_counterfactuals()
        assert loo is not None
        loo_resid, loo_post = loo
        assert loo_resid.shape == (SHORT_PRE,)
        assert loo_post.shape == (SHORT_PRE, SHORT_POST)
        assert np.all(np.isfinite(loo_resid)) and np.all(np.isfinite(loo_post))
        assert results["incrementality_ci_lower"] <= results["incrementality"] <= results["incrementality_ci_upper"]

    @staticmethod
    @pytest.mark.parametrize("cls", MODELS, ids=MODEL_IDS)
    def test_long_pre_uses_conformal(cls) -> None:
        df, pre, post = _panel(LONG_PRE, LONG_POST, effect=40.0)
        results = _build(cls, df, pre, post).pre_process().generate().results
        assert results["method"] == "conformal"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
