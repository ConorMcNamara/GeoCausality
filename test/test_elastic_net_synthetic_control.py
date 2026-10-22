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

import warnings
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


class TestRefitSolver:
    @staticmethod
    def test_penalty_selected_once() -> None:
        # The penalty is cross-validated once on the observed pre-period; the
        # ~50 conformal / cross-fit refits reuse it rather than re-tuning.
        df, pre, post = _panel(LONG_PRE, LONG_POST, effect=40.0)
        model = _build(df, pre, post).pre_process()
        fitted: list[str] = []
        silent_fit = ElasticNetSyntheticControl._silent_fit

        def counting_fit(estimator: object, x: np.ndarray, y: np.ndarray) -> object:
            fitted.append(type(estimator).__name__)
            return silent_fit(estimator, x, y)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(model, "_silent_fit", counting_fit)
            model.generate()
        assert fitted.count("ElasticNetCV") == 1
        assert fitted.count("ElasticNet") > 1

    @staticmethod
    @pytest.mark.parametrize("overrides", [{}, {"intercept": False}, {"non_negative": True}, {"l1_ratio": 0.0}])
    def test_gram_solver_matches_plain(overrides: dict) -> None:
        # The Gram (precomputed) coordinate descent reaches the same solution
        # as the plain solver, so the refit weights agree to rounding error.
        df, pre, post = _panel(LONG_PRE, LONG_POST, effect=40.0)
        model = _build(df, pre, post, **overrides).pre_process().generate()
        assert model.lambda_ is not None
        x, y = model._jk_x_pre, model._jk_y_pre
        plain = model._silent_fit(model._make_estimator(model.lambda_, precompute=False), x, y)
        gram = model._silent_fit(model._make_estimator(model.lambda_, precompute=True), x, y)
        np.testing.assert_allclose(gram.coef_, plain.coef_, rtol=1e-8, atol=1e-10)
        assert gram.intercept_ == pytest.approx(plain.intercept_, rel=1e-8, abs=1e-10)

    @staticmethod
    @pytest.mark.parametrize(("n_pre", "n_post", "n_geos"), [(16, 4, 32), (12, 3, 30)])
    def test_gram_solver_matches_plain_end_to_end(n_pre: int, n_post: int, n_geos: int) -> None:
        # More donors than training dates, so the Gram matrix is singular. The
        # (12, 3) panel is too short for the test to reject, so the interval
        # search runs out to its cap on both sides, refitting at the largest
        # candidate effects it tries; the reported results must not depend on
        # which solver ran those refits.
        df, pre, post = _panel(n_pre, n_post, effect=40.0, n_geos=n_geos)
        runs = []
        for use_gram in (False, True):
            with pytest.MonkeyPatch.context() as mp, warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mp.setattr(ElasticNetSyntheticControl, "_use_gram", staticmethod(lambda x, flag=use_gram: flag))
                runs.append(_build(df, pre, post).pre_process().generate().results)
        plain, gram = runs
        if n_pre == 12:
            assert plain["incrementality_ci_lower"] == -np.inf
            assert plain["incrementality_ci_upper"] == np.inf
        for key in ("incrementality", "p_value", "incrementality_ci_lower", "incrementality_ci_upper"):
            assert gram[key] == pytest.approx(plain[key], rel=1e-9), key

    @staticmethod
    @pytest.mark.parametrize(("shape", "expected"), [((120, 20), True), ((120, 300), True), ((120, 400), False)])
    def test_use_gram_threshold(shape: tuple[int, int], expected: bool) -> None:
        # Gram is used unless donors exceed 2.5x the training dates.
        assert ElasticNetSyntheticControl._use_gram(np.zeros(shape)) is expected


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
