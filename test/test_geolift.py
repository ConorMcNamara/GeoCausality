"""Tests for the unified GeoLift entry point.

``GeoLift`` reproduces Meta GeoLift's two-step pipeline: an Augmented Synthetic
Control for the de-biased point estimate plus a Generalized Synthetic Control
with parametric-bootstrap inference for the uncertainty. These tests check that
the point estimate matches a standalone ASC, the inference matches a standalone
GSC bootstrap, the reported interval brackets the estimate, and that the
chainable interface (summarize / plot) delegates to the ASC counterfactual.
"""

import io
from contextlib import redirect_stdout
from datetime import date, timedelta

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest

from GeoCausality.augmented_synthetic_control import AugmentedSyntheticControl
from GeoCausality.generalized_synthetic_control import GeneralizedSyntheticControl
from GeoCausality.geolift import GeoLift

N_DATES = 40
PRE_PERIOD = "2021-01-30"
POST_PERIOD = "2021-01-31"
TEST_GEOS = ("g0", "g1")
LIFT_TYPES = ("incremental", "absolute", "relative", "revenue", "roas")


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


def _geolift(df: pl.DataFrame, **overrides: object) -> GeoLift:
    kwargs = {
        "treatment_variable": "is_treatment",
        "pre_period": PRE_PERIOD,
        "post_period": POST_PERIOD,
        "msrp": 7.0,
        "spend": 10_000,
        "n_boot": 300,
        "bootstrap_seed": 0,
    }
    kwargs.update(overrides)
    return GeoLift(df, **kwargs)


@pytest.fixture(scope="module")
def effect_data() -> pl.DataFrame:
    return _make_data(effect=50.0)


class TestPipeline:
    @staticmethod
    def test_method_label(effect_data: pl.DataFrame) -> None:
        results = _geolift(effect_data).pre_process().generate().results
        assert results["method"] == "asc + gsc-bootstrap"

    @staticmethod
    def test_point_estimate_matches_standalone_asc(effect_data: pl.DataFrame) -> None:
        gl = _geolift(effect_data).pre_process().generate()
        asc = (
            AugmentedSyntheticControl(
                effect_data, treatment_variable="is_treatment", pre_period=PRE_PERIOD, post_period=POST_PERIOD
            )
            .pre_process()
            .generate()
        )
        assert gl.results["incrementality"] == pytest.approx(asc.results["incrementality"])

    @staticmethod
    def test_inference_matches_standalone_gsc_bootstrap(effect_data: pl.DataFrame) -> None:
        gl = _geolift(effect_data, n_boot=300, bootstrap_seed=0).pre_process().generate()
        gsc = GeneralizedSyntheticControl(
            effect_data, treatment_variable="is_treatment", pre_period=PRE_PERIOD, post_period=POST_PERIOD
        )
        gsc.inference_method = "bootstrap"
        gsc.n_boot = 300
        gsc.bootstrap_seed = 0
        gsc.pre_process().generate()
        assert gl.results["p_value"] == pytest.approx(gsc.results["p_value"])

    @staticmethod
    def test_ci_brackets_point_estimate(effect_data: pl.DataFrame) -> None:
        results = _geolift(effect_data).pre_process().generate().results
        assert results["incrementality_ci_lower"] <= results["incrementality"] <= results["incrementality_ci_upper"]
        assert results["lift_ci_lower"] <= results["lift_ci_upper"]

    @staticmethod
    def test_raw_gsc_inference_exposed(effect_data: pl.DataFrame) -> None:
        results = _geolift(effect_data).pre_process().generate().results
        assert results["gsc_inference"]["method"] == "bootstrap"
        assert "incrementality_ci_lower" in results["gsc_inference"]

    @staticmethod
    def test_strong_effect_is_significant(effect_data: pl.DataFrame) -> None:
        results = _geolift(effect_data, n_boot=400).pre_process().generate().results
        assert results["p_value"] <= 0.1


class TestOptionsForwarding:
    @staticmethod
    def test_gsc_kwargs_forwarded(effect_data: pl.DataFrame) -> None:
        gl = _geolift(effect_data, gsc_kwargs={"n_factors": 1}).pre_process().generate()
        assert gl.inference.n_factors == 1
        assert gl.inference.n_factors_selected == 1

    @staticmethod
    def test_asc_kwargs_forwarded(effect_data: pl.DataFrame) -> None:
        gl = _geolift(effect_data, asc_kwargs={"lambda_": 0.5}).pre_process().generate()
        assert gl.estimator.lambda_ == 0.5


class TestReporting:
    @staticmethod
    @pytest.mark.parametrize("lift", LIFT_TYPES)
    def test_summarize_runs_for_all_lift_types(lift: str, effect_data: pl.DataFrame) -> None:
        gl = _geolift(effect_data).pre_process().generate()
        with redirect_stdout(io.StringIO()) as buffer:
            gl.summarize(lift)
        out = buffer.getvalue()
        assert "p_value" in out
        assert "CI" in out

    @staticmethod
    def test_plot_builds_figure(effect_data: pl.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
        shown = {}
        monkeypatch.setattr(go.Figure, "show", lambda self: shown.setdefault("ok", True))
        _geolift(effect_data).pre_process().generate().plot()
        assert shown.get("ok") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
