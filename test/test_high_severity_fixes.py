"""Regression tests for the remaining high-severity review fixes.

* ``DiffinDiff`` compares group means, so unequal group sizes no longer bias it.
* Simplex weight fits solve to optimality instead of stopping near the start.
* ``NonlinearSyntheticControl`` solves its QP at unit magnitude, so positive L1
  penalties converge on level data.
* The conformal test refits the counterfactual on all periods under each null
  (Chernozhukov, Wuthrich & Zhu, 2021), and the prediction band is calibrated on
  out-of-sample cross-fitted residuals.
"""

import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from GeoCausality.diff_in_diff import DiffinDiff
from GeoCausality.fixed_effects import FixedEffects
from GeoCausality.nonlinear_synthetic_control import NonlinearSyntheticControl
from GeoCausality.penalized_synthetic_control import PenalizedSyntheticControl
from GeoCausality.synthetic_control import SyntheticControl

N_PRE, N_POST = 30, 10
D0 = date(2021, 1, 1)
DATES = [D0 + timedelta(days=i) for i in range(N_PRE + N_POST)]
PRE_PERIOD = str(DATES[N_PRE - 1])
POST_PERIOD = str(DATES[N_PRE])


def _factor_panel(n_treated: int = 1, n_control: int = 12, effect: float = 0.0, level: float = 1000.0, seed: int = 0):
    """A two-factor panel at a realistic outcome level."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(0, 1, size=(len(DATES), 2)).cumsum(axis=0)
    rows = []
    for g in range(n_treated + n_control):
        treated = g < n_treated
        y = level + factors @ rng.uniform(5.0, 15.0, size=2) + rng.normal(0, 3.0, size=len(DATES))
        if treated:
            y[N_PRE:] += effect
        rows += [
            {"geo": f"g{g}", "date": d, "y": float(y[t]), "is_treatment": int(treated)} for t, d in enumerate(DATES)
        ]
    return pl.DataFrame(rows)


def _kwargs(**overrides):
    return {"pre_period": PRE_PERIOD, "post_period": POST_PERIOD, "alpha": 0.1, **overrides}


class TestDiffinDiffGroupMeans:
    @staticmethod
    def _unequal_panel() -> pd.DataFrame:
        # 3 treated vs 10 control geos, a common trend of 2/day and a per-geo
        # effect of 10/day: the true total effect over 20 post days is 600.
        dates = pd.date_range("2021-01-01", periods=60)
        rows = [
            {
                "geo": f"g{g}",
                "date": d,
                "y": 50.0 + 2.0 * t + (5.0 if g < 3 else 0.0) + (10.0 if g < 3 and t >= 40 else 0.0),
                "is_treatment": int(g < 3),
            }
            for g in range(13)
            for t, d in enumerate(dates)
        ]
        return pd.DataFrame(rows)

    @pytest.mark.parametrize("inference", ["ols", "conformal"])
    def test_unequal_groups_recover_total_effect(self, inference: str) -> None:
        df = self._unequal_panel()
        kwargs = {"pre_period": "2021-02-09", "post_period": "2021-02-10", "inference": inference}
        did = DiffinDiff(df, **kwargs).pre_process().generate()
        fe = FixedEffects(df, pre_period="2021-02-09", post_period="2021-02-10").pre_process().generate()
        assert did.results["att"] == pytest.approx(10.0)
        assert did.results["lift"] == pytest.approx(30.0)
        assert did.results["incrementality"] == pytest.approx(600.0)
        assert did.results["incrementality"] == pytest.approx(fe.results["incrementality"])


def _assert_simplex_optimal(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, linear: np.ndarray | None = None
) -> None:
    """KKT check for ``min ||y - x w||^2 + l'w`` over the simplex.

    At the optimum every donor with positive weight has the same (smallest)
    gradient component; donors at zero have a gradient at least that large.
    """
    grad = 2.0 * x.T @ (x @ weights - y)
    if linear is not None:
        grad = grad + linear
    active = weights > 1e-6
    scale = np.max(np.abs(grad)) + 1e-12
    assert np.ptp(grad[active]) / scale < 1e-4
    assert np.all(grad[~active] >= grad[active].min() - 1e-4 * scale)


class TestSimplexSolverOptimality:
    @pytest.mark.parametrize("n_treated", [1, 3])
    def test_synthetic_control_weights_are_optimal(self, n_treated: int) -> None:
        model = SyntheticControl(_factor_panel(n_treated=n_treated), **_kwargs()).pre_process().generate()
        x = model._jk_x_pre / np.max(np.abs(model._jk_x_pre))
        y = model._jk_y_pre / n_treated / np.max(np.abs(model._jk_x_pre))
        w = model.model / n_treated
        assert w.sum() == pytest.approx(1.0)
        _assert_simplex_optimal(x, y, w)

    def test_penalized_weights_are_optimal(self) -> None:
        model = PenalizedSyntheticControl(_factor_panel(), **_kwargs()).pre_process().generate()
        scale = np.max(np.abs(model._jk_x_pre))
        x, y = model._jk_x_pre / scale, model._jk_y_pre / scale
        n = x.shape[0]
        discrepancy = np.mean((y[:, None] - x) ** 2, axis=0)
        w = model._create_model(model._jk_x_pre, model._jk_y_pre)
        _assert_simplex_optimal(x / np.sqrt(n), y / np.sqrt(n), w, linear=model.lambda_ * discrepancy)

    def test_penalized_without_penalty_matches_synthetic_control(self) -> None:
        df = _factor_panel()
        sc = SyntheticControl(df, **_kwargs()).pre_process().generate()
        pen = PenalizedSyntheticControl(df, **_kwargs(), lambda_=0.0).pre_process().generate()
        np.testing.assert_allclose(pen.results["counterfactual"], sc.results["counterfactual"], rtol=1e-4)


class TestNonlinearSyntheticControlSolver:
    @pytest.mark.parametrize("penalty", [1e3, 1e5])
    def test_positive_penalties_converge_on_level_data(self, penalty: float) -> None:
        model = NonlinearSyntheticControl(_factor_panel(), **_kwargs()).pre_process()
        model.generate()
        x, y = model._jk_x_pre, model._jk_y_pre
        weights, converged = model._solve_nsc(y, x, penalty, penalty)
        assert converged
        assert weights.sum() == pytest.approx(1.0)


class TestConformalNullRefit:
    def test_p_value_uses_full_period_refit(self) -> None:
        model = SyntheticControl(_factor_panel(effect=20.0), **_kwargs()).pre_process().generate()
        assert model.results["conformal_refit"]
        x_full = np.vstack([model._jk_x_pre, model._jk_x_post])
        y_full = np.concatenate([model._jk_y_pre, model.actual_post])
        resid = y_full - x_full @ model._create_model(x_full, y_full)
        expected = model._conformal_p_value(resid[:N_PRE], resid[N_PRE:], model.conformal_q)
        assert model.results["p_value"] == pytest.approx(expected)

    def test_effect_is_detected_and_null_is_not(self) -> None:
        effect = SyntheticControl(_factor_panel(effect=40.0), **_kwargs()).pre_process().generate()
        null = SyntheticControl(_factor_panel(effect=0.0), **_kwargs()).pre_process().generate()
        assert effect.results["p_value"] <= 0.1
        assert effect.results["lift_ci_lower"] > 0.0
        assert null.results["lift_ci_lower"] <= 0.0 <= null.results["lift_ci_upper"]

    def test_ci_brackets_point_estimate(self) -> None:
        model = SyntheticControl(_factor_panel(effect=20.0), **_kwargs()).pre_process().generate()
        mean_lift = float(np.mean(model.results["lift"]))
        assert model.results["lift_ci_lower"] <= mean_lift <= model.results["lift_ci_upper"]

    def test_estimator_without_refit_falls_back(self) -> None:
        model = SyntheticControl(_factor_panel(), **_kwargs()).pre_process().generate()
        model._jk_x_pre = None
        assert model._null_residual_fn(model.actual_pre, model.actual_post) is None


class TestCrossFittedBand:
    def test_residuals_are_out_of_sample(self) -> None:
        model = SyntheticControl(_factor_panel(), **_kwargs()).pre_process().generate()
        resid = model._cross_fit_residuals(model.actual_pre, N_POST, n_folds=5)
        # Recompute the first fold by hand: fit without its rows, predict them.
        fold = np.arange(N_PRE // 5)
        keep = np.setdiff1d(np.arange(N_PRE), fold)
        weights = model._create_model(model._jk_x_pre[keep], model._jk_y_pre[keep])
        np.testing.assert_allclose(resid[fold], model._jk_y_pre[fold] - model._jk_x_pre[fold] @ weights)

    def test_band_is_wider_than_in_sample_quantile(self) -> None:
        model = SyntheticControl(_factor_panel(n_control=20), **_kwargs()).pre_process().generate()
        in_sample = np.asarray(model.actual_pre) - np.asarray(model.prediction_pre)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert model.results["conformal_band"] > model._split_conformal_band(in_sample)
