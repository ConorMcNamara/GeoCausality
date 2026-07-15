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

from GeoCausality.matrix_completion import MatrixCompletion

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


class TestEffectRecovery:
    @staticmethod
    def test_recovers_known_effect() -> None:
        df, pre, post = _panel(LONG_PRE, LONG_POST, effect=40.0)
        results = _build(df, pre, post).pre_process().generate().results
        # True total effect: 2 test geos x 40/day x 10 post-days = 800.
        assert results["incrementality"] == pytest.approx(800.0, rel=0.15)

    @staticmethod
    def test_effect_is_positive_and_significant() -> None:
        df, pre, post = _panel(LONG_PRE, LONG_POST, effect=40.0)
        results = _build(df, pre, post).pre_process().generate().results
        assert results["incrementality"] > 0.0
        assert results["p_value"] <= 0.1

    @staticmethod
    def test_null_is_near_zero() -> None:
        # Under the null the recovered incrementality is a tiny fraction of the
        # baseline. We assert magnitude, not (non-)significance: the conformal
        # p-value on a near-perfect completion is noisy on its discrete grid, so
        # it is not a reliable null signal -- the same reason the family parity
        # tests only assert significance for real effects.
        df, pre, post = _panel(LONG_PRE, LONG_POST, effect=0.0)
        results = _build(df, pre, post).pre_process().generate().results
        baseline = float(np.sum(results["counterfactual"]))
        assert abs(results["incrementality"]) < 0.05 * baseline


class TestResultsContract:
    @staticmethod
    def test_results_keys_present() -> None:
        df, pre, post = _panel(LONG_PRE, LONG_POST, effect=40.0)
        results = _build(df, pre, post).pre_process().generate().results
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
        assert expected <= set(results.keys())

    @staticmethod
    def test_counterfactual_length_matches_post_period() -> None:
        df, pre, post = _panel(LONG_PRE, LONG_POST, effect=40.0)
        model = _build(df, pre, post).pre_process().generate()
        assert model.results["counterfactual"].shape == (LONG_POST,)
        assert model.results["test"].shape == (LONG_POST,)

    @staticmethod
    def test_ci_brackets_estimate() -> None:
        df, pre, post = _panel(LONG_PRE, LONG_POST, effect=40.0)
        results = _build(df, pre, post).pre_process().generate().results
        assert results["incrementality_ci_lower"] <= results["incrementality_ci_upper"]
        assert results["incrementality_ci_lower"] <= results["incrementality"] <= results["incrementality_ci_upper"]

    @staticmethod
    def test_summarize_runs(capsys: pytest.CaptureFixture) -> None:
        df, pre, post = _panel(LONG_PRE, LONG_POST, effect=40.0)
        model = _build(df, pre, post).pre_process().generate()
        model.summarize("absolute")
        assert "Incremental" in capsys.readouterr().out


class TestInferenceRouting:
    @staticmethod
    def test_long_pre_uses_conformal() -> None:
        df, pre, post = _panel(LONG_PRE, LONG_POST, effect=40.0)
        results = _build(df, pre, post).pre_process().generate().results
        assert results["method"] == "conformal"

    @staticmethod
    def test_short_pre_uses_residual_jackknife() -> None:
        # No donor-weight refit hook, so the short pre-period falls back to the
        # residual-only jackknife+ (not the faithful refit-based variant).
        df, pre, post = _panel(SHORT_PRE, SHORT_POST, effect=40.0)
        results = _build(df, pre, post).pre_process().generate().results
        assert results["method"] == "jackknife+ (residual)"

    @staticmethod
    def test_no_faithful_loo() -> None:
        df, pre, post = _panel(SHORT_PRE, SHORT_POST, effect=40.0)
        model = _build(df, pre, post).pre_process().generate()
        assert model._loo_counterfactuals() is None


class TestPenaltyAndReproducibility:
    @staticmethod
    def test_reproducible() -> None:
        df, pre, post = _panel(LONG_PRE, LONG_POST, effect=40.0)
        a = _build(df, pre, post).pre_process().generate().results
        b = _build(df, pre, post).pre_process().generate().results
        assert a["incrementality"] == b["incrementality"]
        assert a["p_value"] == b["p_value"]

    @staticmethod
    def test_fixed_lambda_runs() -> None:
        df, pre, post = _panel(LONG_PRE, LONG_POST, effect=40.0)
        results = _build(df, pre, post, lambda_=0.0).pre_process().generate().results
        # lambda_=0 is pure interpolation of the observed cells plus two-way FE;
        # it still recovers a positive, sizeable effect.
        assert results["incrementality"] > 0.0

    @staticmethod
    def test_no_time_effect_runs() -> None:
        df, pre, post = _panel(LONG_PRE, LONG_POST, effect=40.0)
        results = _build(df, pre, post, use_time_effect=False).pre_process().generate().results
        assert results["incrementality"] > 0.0


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
