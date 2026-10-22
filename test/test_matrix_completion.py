"""Unit tests for the matrix-completion (MC-NNM) estimator.

MatrixCompletion stacks every unit into one panel, masks the treated unit's
post-period, and completes the matrix under a nuclear-norm penalty (two-way
fixed effects plus a low-rank term, solved by soft-impute with a
cross-validated penalty). Unlike the linear synthetic-control family it has no
donor-weight vector, so it does not cache the donor matrices: on a short
pre-period it uses the residual-only jackknife+ fallback (``"jackknife+
(residual)"``), not the faithful refit-based ``"jackknife+"``. These tests cover
effect recovery, the null, the results contract, reproducibility, the fixed vs
cross-validated penalty, and the inference-method routing.
"""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from GeoCausality.matrix_completion import MatrixCompletion, _svt

TEST_GEOS = ("g0", "g1")
LONG_PRE, LONG_POST = 30, 10
SHORT_PRE, SHORT_POST = 8, 6


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


def _build(df: pl.DataFrame, pre: str, post: str, **overrides: object) -> MatrixCompletion:
    kwargs: dict = {"treatment_variable": "is_treatment", "pre_period": pre, "post_period": post, "y_variable": "y"}
    kwargs.update(overrides)
    return MatrixCompletion(df, **kwargs)


@pytest.fixture(scope="module")
def long_effect_model() -> MatrixCompletion:
    df, pre, post = _panel(LONG_PRE, LONG_POST, effect=40.0)
    return _build(df, pre, post).pre_process().generate()


@pytest.fixture(scope="module")
def null_model() -> MatrixCompletion:
    df, pre, post = _panel(LONG_PRE, LONG_POST, effect=0.0)
    return _build(df, pre, post).pre_process().generate()


@pytest.fixture(scope="module")
def short_pre_model() -> MatrixCompletion:
    df, pre, post = _panel(SHORT_PRE, SHORT_POST, effect=40.0)
    return _build(df, pre, post).pre_process().generate()


class TestEffectRecovery:
    @staticmethod
    def test_recovers_known_effect(long_effect_model: MatrixCompletion) -> None:
        # True total effect: 2 test geos x 40/day x 10 post-days = 800.
        assert long_effect_model.results["incrementality"] == pytest.approx(800.0, rel=0.15)

    @staticmethod
    def test_effect_is_positive_and_significant(long_effect_model: MatrixCompletion) -> None:
        assert long_effect_model.results["incrementality"] > 0.0
        assert long_effect_model.results["p_value"] <= 0.1

    @staticmethod
    def test_null_is_near_zero(null_model: MatrixCompletion) -> None:
        baseline = float(np.sum(null_model.results["counterfactual"]))
        assert abs(null_model.results["incrementality"]) < 0.05 * baseline


class TestResultsContract:
    @staticmethod
    def test_results_keys_present(long_effect_model: MatrixCompletion) -> None:
        expected = {
            "test",
            "counterfactual",
            "lift",
            "incrementality",
            "p_value",
            "incrementality_ci_lower",
            "incrementality_ci_upper",
            "lift_ci_lower",
            "lift_ci_upper",
            "conformal_band",
            "method",
        }
        assert expected <= set(long_effect_model.results.keys())

    @staticmethod
    def test_counterfactual_length_matches_post_period(long_effect_model: MatrixCompletion) -> None:
        assert long_effect_model.results["counterfactual"].shape == (LONG_POST,)
        assert long_effect_model.results["test"].shape == (LONG_POST,)

    @staticmethod
    def test_ci_brackets_estimate(long_effect_model: MatrixCompletion) -> None:
        results = long_effect_model.results
        assert results["incrementality_ci_lower"] <= results["incrementality_ci_upper"]
        assert results["incrementality_ci_lower"] <= results["incrementality"] <= results["incrementality_ci_upper"]

    @staticmethod
    def test_summarize_runs(long_effect_model: MatrixCompletion, capsys: pytest.CaptureFixture) -> None:
        long_effect_model.summarize("absolute")
        assert "Incremental" in capsys.readouterr().out


class TestInferenceRouting:
    @staticmethod
    def test_long_pre_uses_conformal(long_effect_model: MatrixCompletion) -> None:
        assert long_effect_model.results["method"] == "conformal"

    @staticmethod
    def test_short_pre_uses_residual_jackknife(short_pre_model: MatrixCompletion) -> None:
        assert short_pre_model.results["method"] == "jackknife+ (residual)"

    @staticmethod
    def test_no_faithful_loo(short_pre_model: MatrixCompletion) -> None:
        assert short_pre_model._loo_counterfactuals() is None


class TestPenaltyAndReproducibility:
    @staticmethod
    def test_reproducible(long_effect_model: MatrixCompletion) -> None:
        df, pre, post = _panel(LONG_PRE, LONG_POST, effect=40.0)
        fresh = _build(df, pre, post).pre_process().generate().results
        assert long_effect_model.results["incrementality"] == fresh["incrementality"]
        assert long_effect_model.results["p_value"] == fresh["p_value"]

    @staticmethod
    def test_fixed_lambda_runs() -> None:
        df, pre, post = _panel(LONG_PRE, LONG_POST, effect=40.0)
        results = _build(df, pre, post, lambda_=0.0).pre_process().generate().results
        assert results["incrementality"] > 0.0

    @staticmethod
    def test_no_time_effect_runs() -> None:
        df, pre, post = _panel(LONG_PRE, LONG_POST, effect=40.0)
        results = _build(df, pre, post, use_time_effect=False).pre_process().generate().results
        assert results["incrementality"] > 0.0


def _full_svd_svt(x: np.ndarray, lambda_: float) -> np.ndarray:
    """Reference soft-threshold via a full SVD."""
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return (u * np.maximum(s - lambda_, 0.0)) @ vt


def _masked_panel(seed: int = 0, shape: tuple[int, int] = (12, 40)) -> tuple[np.ndarray, np.ndarray]:
    """A rank-2-plus-noise panel with fixed effects and ~20% of cells masked."""
    rng = np.random.default_rng(seed)
    n_units, n_periods = shape
    panel = (
        rng.normal(0, 5, size=(n_units, 1))
        + rng.normal(0, 1, size=(1, n_periods)).cumsum(axis=1)
        + rng.normal(size=(n_units, 2)) @ rng.normal(size=(2, n_periods)) * 3.0
        + rng.normal(0, 0.5, size=shape)
    )
    mask = rng.uniform(size=shape) > 0.2
    return panel, mask


class TestSoftThreshold:
    @staticmethod
    @pytest.mark.parametrize("shape", [(5, 9), (9, 5), (30, 30), (21, 120), (120, 21)])
    @pytest.mark.parametrize("rank", [1, 3, None])
    def test_matches_full_svd(shape: tuple[int, int], rank: int | None) -> None:
        rng = np.random.default_rng(sum(shape))
        k = min(shape) if rank is None else rank
        x = rng.normal(size=(shape[0], k)) @ rng.normal(size=(k, shape[1])) * 10.0 + rng.normal(size=shape)
        s = np.linalg.svd(x, compute_uv=False)
        # Above the top singular value, between singular values, near the
        # bottom of the spectrum, and the tiny-penalty full-SVD fallback.
        for lambda_ in [1.01 * s[0], 0.5 * (s[0] + s[-1]), s[min(2, s.size - 1)], 0.5 * s[-1], 1e-3 * s[0], 1e-9, 0.0]:
            expected = _full_svd_svt(x, lambda_)
            np.testing.assert_allclose(_svt(x, lambda_), expected, rtol=0.0, atol=1e-10 * s[0])

    @staticmethod
    def test_above_top_singular_value_is_zero() -> None:
        x = np.random.default_rng(0).normal(size=(6, 8))
        lambda_ = 1.5 * np.linalg.svd(x, compute_uv=False)[0]
        assert not np.any(_svt(x, lambda_))


class TestSolverInternals:
    @staticmethod
    def test_two_way_fe_full_mask_is_two_way_demeaning() -> None:
        df, pre, post = _panel(SHORT_PRE, SHORT_POST, effect=0.0)
        model = _build(df, pre, post, tol=1e-12, max_iter=1000)
        panel, _ = _masked_panel()
        gamma, delta = model._two_way_fe(panel, np.ones_like(panel, dtype=bool))
        expected = panel.mean(axis=1)[:, None] + panel.mean(axis=0)[None, :] - panel.mean()
        np.testing.assert_allclose(gamma[:, None] + delta[None, :], expected, atol=1e-9)

    @staticmethod
    def test_warm_start_reaches_cold_start_solution() -> None:
        # Warm-starting a penalty from the previous one's low-rank term (as the
        # cross-validation path does) converges to the same completion.
        df, pre, post = _panel(SHORT_PRE, SHORT_POST, effect=0.0)
        model = _build(df, pre, post, tol=1e-11, max_iter=5000)
        panel, mask = _masked_panel()
        path = model._lambda_path(panel, mask)
        _, warm = model._complete(panel, mask, path[5])
        cold_fit, _ = model._complete(panel, mask, path[6])
        warm_fit, _ = model._complete(panel, mask, path[6], warm)
        np.testing.assert_allclose(warm_fit, cold_fit, rtol=0.0, atol=1e-6 * np.abs(cold_fit).max())

    @staticmethod
    def test_continuation_solve_is_near_converged() -> None:
        # At a small penalty the final fit (reached by continuation down the
        # path) at the default tolerance sits close to a tightly converged fit;
        # a cold solve from zero stops ~30x farther away on this panel.
        df, pre, post = _panel(SHORT_PRE, SHORT_POST, effect=0.0)
        panel, mask = _masked_panel(seed=1)
        loose = _build(df, pre, post)
        lambda_ = float(loose._lambda_path(panel, mask)[30])
        loose.lambda_ = lambda_
        tight = _build(df, pre, post, lambda_=lambda_, tol=1e-11, max_iter=20000)
        expected = tight._solve(panel, mask)
        np.testing.assert_allclose(loose._solve(panel, mask), expected, rtol=0.0, atol=1e-4 * np.abs(expected).max())

    @staticmethod
    def test_fixed_lambda_ignores_cv_path_settings() -> None:
        # A supplied penalty is reached by a fixed continuation path, so the CV
        # path's length and depth do not change the fit.
        df, pre, post = _panel(SHORT_PRE, SHORT_POST, effect=0.0)
        panel, mask = _masked_panel(seed=1)
        lambda_ = 0.002 * float(_build(df, pre, post)._lambda_path(panel, mask)[0])
        expected = _build(df, pre, post, lambda_=lambda_)._solve(panel, mask)
        for n_lambda, lambda_ratio in [(2, 1e-3), (200, 1e-3), (50, 0.5)]:
            model = _build(df, pre, post, lambda_=lambda_, n_lambda=n_lambda, lambda_ratio=lambda_ratio)
            np.testing.assert_array_equal(model._solve(panel, mask), expected)

    @staticmethod
    def test_cv_selected_lambda_is_pinned() -> None:
        # Regression pin on the warm-started CV path: the held-out error has a
        # clear minimum at step 25 of 50 on this panel (the neighbours are ~0.1%
        # worse), so any drift in the path or the warm starts shows up here.
        df, pre, post = _panel(SHORT_PRE, SHORT_POST, effect=0.0)
        panel, mask = _masked_panel(seed=0)
        assert _build(df, pre, post)._cv_lambda(panel, mask) == pytest.approx(1.1571268454788184, rel=1e-6)


class TestValidation:
    @staticmethod
    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_cv_fraction_out_of_range_raises(bad: float) -> None:
        df, pre, post = _panel(LONG_PRE, LONG_POST, effect=40.0)
        with pytest.raises(ValueError, match="cv_fraction"):
            _build(df, pre, post, cv_fraction=bad)

    @staticmethod
    def test_negative_lambda_raises() -> None:
        df, pre, post = _panel(LONG_PRE, LONG_POST, effect=40.0)
        with pytest.raises(ValueError, match="lambda_"):
            _build(df, pre, post, lambda_=-1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
