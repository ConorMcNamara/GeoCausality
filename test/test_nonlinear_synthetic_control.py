"""Simulation-based validation for the two nonlinear synthetic-control estimators.

No real panel has a *known* counterfactual, so correctness of a causal estimator
is validated on synthetic panels where the ground-truth effect is injected by
construction. These tests cover both nonlinear estimators:

* ``NonlinearSyntheticControl`` -- Tian (2023): a linear-in-weights counterfactual
  with affine (signed, sum-to-one) weights and a distance-weighted L1 + ridge L2
  penalty. Faithful when the treated unit's level is reachable from the donors, so
  the panels here use a **single treated geo** (as in the Prop 99 / West Germany
  benchmarks), where the treated and donor series live on the same scale.
* ``KernelSyntheticControl`` -- kernel ridge with a composite linear+RBF kernel: a
  genuinely nonlinear map of the donor outcomes.

The suite asserts (1) the conformal-inference contract shared by the whole
estimator family, (2) recovery of a known effect and a calibrated null, and (3)
the head-to-head that motivates keeping both: when the treated unit relates to the
donors *nonlinearly*, the kernel estimator reconstructs the counterfactual that
the linear-in-weights NSC cannot. The published-benchmark checks (that NSC
reproduces the canonical Prop 99 / West Germany ATTs) live in the parity tests.
"""

import io
from contextlib import redirect_stdout
from datetime import date, timedelta

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest

from GeoCausality.kernel_synthetic_control import KernelSyntheticControl
from GeoCausality.nonlinear_synthetic_control import NonlinearSyntheticControl

N_DATES = 40
PRE_PERIOD = "2021-01-30"  # first 30 days are pre-period
POST_PERIOD = "2021-01-31"  # final 10 days are post-period
N_POST = 10
TREATED = "g0"
N_CONTROL = 9
CONFORMAL_KEYS = (
    "p_value",
    "lift_ci_lower",
    "lift_ci_upper",
    "incrementality_ci_lower",
    "incrementality_ci_upper",
    "conformal_band",
)
LIFT_TYPES = ("incremental", "absolute", "relative", "revenue", "roas")


def _make_data(effect: float, seed: int = 0, nonlinear: bool = False, n_factors: int = 2) -> pl.DataFrame:
    """Build a single-treated-geo panel with a known post-period effect.

    Every geo's outcome is driven by ``n_factors`` shared latent time factors.
    Control geos are linear in the factors. When ``nonlinear`` is False the treated
    geo is too (so a linear-in-weights method is well specified); when True the
    treated geo is a *quadratic* function of the shared factors, so its untreated
    outcome is a nonlinear function of the (linear) donor outcomes -- recoverable by
    the kernel map but not by a linear combination of donors.

    Parameters
    ----------
    effect : float
        Per-day additive treatment effect applied to the treated geo post-period.
    seed : int
        Seed for reproducibility.
    nonlinear : bool
        Whether the treated geo relates nonlinearly to the shared factors.
    n_factors : int
        Number of shared latent factors driving the panel.
    """
    rng = np.random.default_rng(seed)
    d0 = date(2021, 1, 1)
    dates = [d0 + timedelta(days=i) for i in range(N_DATES)]
    geos = [f"g{i}" for i in range(1 + N_CONTROL)]
    factors = rng.normal(0, 1, size=(n_factors, N_DATES)).cumsum(axis=1)
    loadings = {g: rng.uniform(0.5, 2.0, size=n_factors) for g in geos}
    post_start = date.fromisoformat(POST_PERIOD)

    rows = []
    for gi, g in enumerate(geos):
        is_test = gi == 0
        index = loadings[g] @ factors
        for di, d in enumerate(dates):
            if is_test and nonlinear:
                y = 100.0 + 0.5 * index[di] ** 2 + rng.normal(0, 2)
            else:
                y = 100.0 + index[di] + rng.normal(0, 2)
            if is_test and d >= post_start:
                y += effect
            rows.append({"geo": g, "date": d, "y": float(y), "is_treatment": int(is_test)})
    return pl.DataFrame(rows)


def _nsc(df: pl.DataFrame, **overrides: object) -> NonlinearSyntheticControl:
    kwargs = {
        "geo_variable": "geo",
        "treatment_variable": "is_treatment",
        "date_variable": "date",
        "pre_period": PRE_PERIOD,
        "post_period": POST_PERIOD,
        "y_variable": "y",
        "msrp": 7.0,
        "spend": 10_000,
        "cv_grid_step": 0.25,  # coarse grid keeps the well-specified tests fast
    }
    kwargs.update(overrides)
    return NonlinearSyntheticControl(df, **kwargs)


def _kernel(df: pl.DataFrame, **overrides: object) -> KernelSyntheticControl:
    kwargs = {
        "geo_variable": "geo",
        "treatment_variable": "is_treatment",
        "date_variable": "date",
        "pre_period": PRE_PERIOD,
        "post_period": POST_PERIOD,
        "y_variable": "y",
        "msrp": 7.0,
        "spend": 10_000,
    }
    kwargs.update(overrides)
    return KernelSyntheticControl(df, **kwargs)


@pytest.fixture(scope="module")
def effect_data() -> pl.DataFrame:
    return _make_data(effect=8.0)


@pytest.fixture(scope="module")
def null_data() -> pl.DataFrame:
    return _make_data(effect=0.0)


@pytest.fixture(scope="module")
def nonlinear_null_data() -> pl.DataFrame:
    return _make_data(effect=0.0, nonlinear=True)


class TestNonlinearSyntheticControl:
    """Tian (2023) NSC on the well-specified linear, single-treated panel."""

    @staticmethod
    def test_results_contain_conformal_keys(effect_data: pl.DataFrame) -> None:
        results = _nsc(effect_data).pre_process().generate().results
        for key in CONFORMAL_KEYS:
            assert key in results, f"missing conformal key {key!r}"
        assert 0.0 <= results["p_value"] <= 1.0
        assert results["conformal_band"] >= 0.0
        assert results["incrementality_ci_lower"] <= results["incrementality_ci_upper"]

    @staticmethod
    def test_point_estimate_within_interval(effect_data: pl.DataFrame) -> None:
        results = _nsc(effect_data).pre_process().generate().results
        assert results["incrementality_ci_lower"] <= results["incrementality"] <= results["incrementality_ci_upper"]

    @staticmethod
    def test_recovers_known_effect(effect_data: pl.DataFrame) -> None:
        results = _nsc(effect_data).pre_process().generate().results
        expected = 8.0 * N_POST
        assert results["incrementality"] == pytest.approx(expected, rel=0.4)
        assert results["p_value"] <= 0.1
        assert results["incrementality_ci_lower"] > 0.0

    @staticmethod
    def test_effect_more_significant_than_null(effect_data: pl.DataFrame, null_data: pl.DataFrame) -> None:
        p_effect = _nsc(effect_data).pre_process().generate().results["p_value"]
        p_null = _nsc(null_data).pre_process().generate().results["p_value"]
        assert p_effect <= p_null

    @staticmethod
    def test_null_effect_not_significant(null_data: pl.DataFrame) -> None:
        # Calibration / in-place placebo: no injected effect => not significant.
        results = _nsc(null_data).pre_process().generate().results
        assert results["p_value"] > 0.1

    @staticmethod
    def test_weights_are_affine(effect_data: pl.DataFrame) -> None:
        # Tian NSC drops non-negativity but keeps the adding-up (sum-to-one) constraint.
        model = _nsc(effect_data).pre_process().generate()
        assert float(np.sum(model.model)) == pytest.approx(1.0, abs=1e-4)

    @staticmethod
    def test_fixed_penalties_skip_cv(effect_data: pl.DataFrame) -> None:
        model = _nsc(effect_data, a=3.0, b=35.0).pre_process().generate()
        assert model._fit_a == 3.0
        assert model._fit_b == 35.0

    @staticmethod
    def test_is_deterministic(effect_data: pl.DataFrame) -> None:
        first = _nsc(effect_data).pre_process().generate().results["incrementality"]
        second = _nsc(effect_data).pre_process().generate().results["incrementality"]
        assert first == pytest.approx(second)

    @staticmethod
    def test_jackknife_fallback_contract(effect_data: pl.DataFrame) -> None:
        model = _nsc(effect_data)
        model.inference_method = "jackknife"
        results = model.pre_process().generate().results
        assert results["method"].startswith("jackknife+")
        for key in CONFORMAL_KEYS:
            assert key in results

    @staticmethod
    @pytest.mark.parametrize("lift", LIFT_TYPES)
    def test_summarize_runs_for_all_lift_types(lift: str, effect_data: pl.DataFrame) -> None:
        model = _nsc(effect_data).pre_process().generate()
        with redirect_stdout(io.StringIO()) as buffer:
            model.summarize(lift)
        assert "p_value" in buffer.getvalue()

    @staticmethod
    def test_summarize_rejects_invalid_lift(effect_data: pl.DataFrame) -> None:
        model = _nsc(effect_data).pre_process().generate()
        with pytest.raises(ValueError):
            model.summarize("nonsense")


class TestKernelSyntheticControl:
    """Kernel-ridge (nonlinear-map) SC on the well-specified linear panel."""

    @staticmethod
    def test_results_contain_conformal_keys(effect_data: pl.DataFrame) -> None:
        results = _kernel(effect_data).pre_process().generate().results
        for key in CONFORMAL_KEYS:
            assert key in results, f"missing conformal key {key!r}"
        assert 0.0 <= results["p_value"] <= 1.0
        assert results["incrementality_ci_lower"] <= results["incrementality_ci_upper"]

    @staticmethod
    def test_recovers_known_effect(effect_data: pl.DataFrame) -> None:
        results = _kernel(effect_data).pre_process().generate().results
        expected = 8.0 * N_POST
        assert results["incrementality"] == pytest.approx(expected, rel=0.5)
        assert results["p_value"] <= 0.1

    @staticmethod
    def test_model_is_callable_predictor(effect_data: pl.DataFrame) -> None:
        # Unlike the linear family, the fitted model is a predictor closure, not weights.
        model = _kernel(effect_data).pre_process().generate()
        assert callable(model.model)

    @staticmethod
    def test_fixed_hyperparameters_honored(effect_data: pl.DataFrame) -> None:
        model = _kernel(effect_data, bandwidth=2.0, lambda_=0.5).pre_process().generate()
        assert model._fit_bandwidth == 2.0
        assert model._fit_lambda == 0.5

    @staticmethod
    def test_pure_rbf_kernel_runs(effect_data: pl.DataFrame) -> None:
        # linear_weight=0 drops the linear backbone (pure RBF); it should still fit.
        results = _kernel(effect_data, linear_weight=0.0).pre_process().generate().results
        assert np.isfinite(results["incrementality"])

    @staticmethod
    def test_jackknife_fallback_contract(effect_data: pl.DataFrame) -> None:
        model = _kernel(effect_data)
        model.inference_method = "jackknife"
        results = model.pre_process().generate().results
        assert results["method"].startswith("jackknife+")
        for key in CONFORMAL_KEYS:
            assert key in results


class TestNonlinearVsKernel:
    """The head-to-head that justifies keeping both estimators."""

    @staticmethod
    def test_kernel_beats_nsc_on_nonlinear_dgp(nonlinear_null_data: pl.DataFrame) -> None:
        # Treated is quadratic in the shared factors and there is no real effect, so
        # any measured incrementality is pure counterfactual error. The kernel map
        # reconstructs the nonlinear relationship; the linear-in-weights NSC cannot,
        # so it carries a much larger spurious "effect".
        nsc_incr = abs(_nsc(nonlinear_null_data).pre_process().generate().results["incrementality"])
        kernel_incr = abs(_kernel(nonlinear_null_data).pre_process().generate().results["incrementality"])
        assert kernel_incr < nsc_incr

    @staticmethod
    def test_both_recover_effect_on_linear_dgp(effect_data: pl.DataFrame) -> None:
        # On a linear DGP both are well specified and detect the same strong effect.
        nsc = _nsc(effect_data).pre_process().generate().results
        kernel = _kernel(effect_data).pre_process().generate().results
        assert nsc["p_value"] <= 0.1
        assert kernel["p_value"] <= 0.1


class TestPlot:
    @staticmethod
    @pytest.mark.parametrize("builder", [_nsc, _kernel])
    def test_plot_builds_figure(builder, effect_data: pl.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
        shown = {}
        monkeypatch.setattr(go.Figure, "show", lambda self: shown.setdefault("ok", True))
        builder(effect_data).pre_process().generate().plot()
        assert shown.get("ok") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
