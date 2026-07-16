"""Unit tests for the elastic-net synthetic control estimator (Doudchenko-Imbens synthesis, MVP).

ElasticNetSyntheticControl fits ``mu + donor_matrix @ w`` by elastic net over the
pre-period, relaxing classic synthetic control's intercept / sum-to-one /
non-negativity restrictions. This MVP covers the unconstrained regime (sklearn
elastic net); the sum-to-one constrained path is not implemented.

These tests focus on DI-*specific* behavior: effect recovery, the null, the
configuration switches (intercept, non-negativity, ridge, OLS), and input
validation. Its inference is exercised as part of the shared weight-based family
sweeps (``test_jackknife_sc_estimators`` / ``test_bootstrap_sc_estimators``) --
DI has a donor-weight vector, so it gets the faithful jackknife+ and the
parametric bootstrap there -- rather than re-checked here.
"""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from GeoCausality.elastic_net_synthetic_control import ElasticNetSyntheticControl

TEST_GEOS = ("g0", "g1")
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


def _build(df: pl.DataFrame, pre: str, post: str, **overrides: object) -> ElasticNetSyntheticControl:
    kwargs: dict = {"treatment_variable": "is_treatment", "pre_period": pre, "post_period": post, "y_variable": "y"}
    kwargs.update(overrides)
    return ElasticNetSyntheticControl(df, **kwargs)


class TestEffectRecovery:
    @staticmethod
    def test_recovers_effect_and_contract() -> None:
        # Default (elastic net + intercept, CV): recovers the known effect, is
        # positive and significant, and populates the shared results contract.
        df, pre, post = _panel(LONG_PRE, LONG_POST, effect=40.0)
        results = _build(df, pre, post).pre_process().generate().results
        # True total effect: 2 test geos x 40/day x 10 post-days = 800.
        assert results["incrementality"] == pytest.approx(800.0, rel=0.15)
        assert results["p_value"] <= 0.1
        assert {"test", "counterfactual", "lift", "incrementality", "p_value", "method"} <= set(results.keys())
        assert results["incrementality_ci_lower"] <= results["incrementality"] <= results["incrementality_ci_upper"]

    @staticmethod
    def test_null_is_near_zero() -> None:
        df, pre, post = _panel(LONG_PRE, LONG_POST, effect=0.0)
        results = _build(df, pre, post).pre_process().generate().results
        baseline = float(np.sum(results["counterfactual"]))
        assert abs(results["incrementality"]) < 0.05 * baseline


class TestConfiguration:
    @staticmethod
    def test_intercept_toggle() -> None:
        # The distinguishing DI feature: a fitted level-shift intercept by default,
        # which is zeroed out when intercept=False.
        df, pre, post = _panel(LONG_PRE, LONG_POST, effect=40.0)
        assert abs(_build(df, pre, post).pre_process().generate().intercept_) > 0.0
        assert _build(df, pre, post, intercept=False).pre_process().generate().intercept_ == 0.0

    @staticmethod
    def test_non_negative_constraint() -> None:
        df, pre, post = _panel(LONG_PRE, LONG_POST, effect=40.0)
        model = _build(df, pre, post, non_negative=True).pre_process().generate()
        assert float(np.min(model.model)) >= -1e-9

    @staticmethod
    @pytest.mark.parametrize("overrides", [{"l1_ratio": 0.0}, {"lambda_": 0.0}], ids=["ridge", "ols"])
    def test_penalty_variants_run(overrides: dict) -> None:
        # Pure ridge (l1_ratio=0, the CV-alpha-grid regression guard) and the
        # unpenalised OLS path (lambda_=0 -> LinearRegression) both fit and recover.
        df, pre, post = _panel(LONG_PRE, LONG_POST, effect=40.0)
        results = _build(df, pre, post, **overrides).pre_process().generate().results
        assert results["incrementality"] > 0.0


class TestValidation:
    @staticmethod
    @pytest.mark.parametrize(
        ("overrides", "exc", "match"),
        [
            ({"l1_ratio": -0.1}, ValueError, "l1_ratio"),
            ({"l1_ratio": 1.5}, ValueError, "l1_ratio"),
            ({"lambda_": -1.0}, ValueError, "lambda_"),
            ({"sum_to_one": True}, NotImplementedError, "sum-to-one"),
        ],
    )
    def test_invalid_config_raises(overrides: dict, exc: type[Exception], match: str) -> None:
        df, pre, post = _panel(LONG_PRE, LONG_POST, effect=40.0)
        with pytest.raises(exc, match=match):
            _build(df, pre, post, **overrides)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
