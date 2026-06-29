"""Tests for the jackknife+ fallback used when the pre-period is short.

When the pre-treatment period is too short for the moving-block permutation test
and split-conformal band to be reliable, ``EconometricEstimator`` falls back to
jackknife+ (Barber et al., 2021). Estimators that override ``_loo_counterfactuals``
(e.g. ``GeneralizedSyntheticControl``) get the faithful refit-based version;
others get a residual-only approximation. These tests cover the trigger logic,
the quantile helper, and both fallback paths, plus that long pre-periods are
unchanged (still ``"conformal"``).
"""

import io
from contextlib import redirect_stdout
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from GeoCausality.generalized_synthetic_control import GeneralizedSyntheticControl
from GeoCausality.synthetic_control import SyntheticControl

TEST_GEOS = ("g0", "g1")


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
    pre_period = dates[n_pre - 1].isoformat()
    post_period = dates[n_pre].isoformat()
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
    return pl.DataFrame(rows), pre_period, post_period


def _fit(estimator_cls, df: pl.DataFrame, pre: str, post: str, **overrides: object):
    kwargs = {
        "geo_variable": "geo",
        "treatment_variable": "is_treatment",
        "date_variable": "date",
        "pre_period": pre,
        "post_period": post,
        "y_variable": "y",
        "msrp": 7.0,
        "spend": 10_000,
    }
    kwargs.update(overrides)
    return estimator_cls(df, **kwargs).pre_process().generate()


class TestTrigger:
    @staticmethod
    def test_short_pre_period_flagged() -> None:
        df, pre, post = _panel(n_pre=6, n_post=5, effect=0.0)
        model = SyntheticControl(df, treatment_variable="is_treatment", pre_period=pre, post_period=post)
        assert model._pre_period_too_short(6) is True
        assert model._pre_period_too_short(30) is False

    @staticmethod
    def test_inference_method_override() -> None:
        df, pre, post = _panel(n_pre=30, n_post=5, effect=0.0)
        model = SyntheticControl(df, treatment_variable="is_treatment", pre_period=pre, post_period=post)
        model.inference_method = "jackknife"
        assert model._pre_period_too_short(100) is True
        model.inference_method = "conformal"
        assert model._pre_period_too_short(2) is False


class TestJackknifeQuantile:
    @staticmethod
    def test_widens_to_max_when_rank_exceeds_n() -> None:
        # alpha=0.1, n=5 -> k=ceil(0.9*6)=6 > 5 -> max residual.
        resids = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        assert SyntheticControl._jackknife_quantile(resids, 0.1) == 5.0

    @staticmethod
    def test_selects_rank_within_n() -> None:
        # alpha=0.5, n=5 -> k=ceil(0.5*6)=3 -> 3rd smallest = 3.0.
        resids = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        assert SyntheticControl._jackknife_quantile(resids, 0.5) == 3.0


class TestResidualFallback:
    @staticmethod
    def test_residual_fallback_when_hook_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
        # When an estimator's leave-one-out hook yields nothing, the short-pre-period
        # path falls back to the residual-only jackknife+ approximation.
        df, pre, post = _panel(n_pre=6, n_post=5, effect=30.0)
        monkeypatch.setattr(SyntheticControl, "_loo_counterfactuals", lambda self: None)
        results = _fit(SyntheticControl, df, pre, post).results
        assert results["method"] == "jackknife+ (residual)"
        assert results["conformal_band"] >= 0.0
        assert results["incrementality_ci_lower"] <= results["incrementality_ci_upper"]
        assert results["incrementality_ci_lower"] <= results["incrementality"] <= results["incrementality_ci_upper"]

    @staticmethod
    def test_long_pre_uses_conformal() -> None:
        df, pre, post = _panel(n_pre=30, n_post=10, effect=30.0)
        results = _fit(SyntheticControl, df, pre, post).results
        assert results["method"] == "conformal"


class TestFullJackknife:
    @staticmethod
    def test_short_pre_uses_refit_jackknife() -> None:
        df, pre, post = _panel(n_pre=6, n_post=5, effect=30.0)
        results = _fit(GeneralizedSyntheticControl, df, pre, post).results
        assert results["method"] == "jackknife+"
        assert results["incrementality_ci_lower"] <= results["incrementality"] <= results["incrementality_ci_upper"]
        # lift CI and incrementality CI stay consistent (incrementality = lift * t1).
        t1 = 5
        assert results["incrementality_ci_lower"] == pytest.approx(results["lift_ci_lower"] * t1)

    @staticmethod
    def test_loo_counterfactuals_shapes() -> None:
        df, pre, post = _panel(n_pre=6, n_post=5, effect=30.0)
        model = _fit(GeneralizedSyntheticControl, df, pre, post, n_factors=1)
        loo = model._loo_counterfactuals()
        assert loo is not None
        loo_resid, loo_post = loo
        assert loo_resid.ndim == 1
        assert loo_post.shape[0] == loo_resid.shape[0]
        assert loo_post.shape[1] == 5  # one column per post-period date

    @staticmethod
    def test_long_pre_uses_conformal() -> None:
        df, pre, post = _panel(n_pre=30, n_post=10, effect=30.0)
        results = _fit(GeneralizedSyntheticControl, df, pre, post).results
        assert results["method"] == "conformal"

    @staticmethod
    def test_forced_jackknife_on_long_pre() -> None:
        df, pre, post = _panel(n_pre=30, n_post=10, effect=30.0)
        model = GeneralizedSyntheticControl(df, treatment_variable="is_treatment", pre_period=pre, post_period=post)
        model.inference_method = "jackknife"
        results = model.pre_process().generate().results
        assert results["method"] == "jackknife+"


class TestReportingUnaffected:
    @staticmethod
    def test_summarize_runs_with_fallback() -> None:
        df, pre, post = _panel(n_pre=6, n_post=5, effect=30.0)
        model = _fit(SyntheticControl, df, pre, post)
        with redirect_stdout(io.StringIO()) as buffer:
            model.summarize("incremental")
        assert "p_value" in buffer.getvalue()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
