"""Tests for the pre-experiment power-analysis / MDE module.

``PowerAnalysis`` carves placebo experiments out of clean pre-period history,
injects a known effect into the test geos, and runs a pluggable estimator to see
how often the effect is detected. These tests assert the structural contract of
the power curve, the qualitative behaviour that detection rises with effect size,
that the no-effect cell calibrates against ``alpha``, and that the MDE falls in a
sensible range.

The panel reuses the two-latent-factor structure from the conformal-inference
tests so a synthetic control can reconstruct the test geos from the controls.
``SyntheticControl`` is used as the (fast, deterministic) estimator throughout.
"""

import io
from contextlib import redirect_stdout
from datetime import date, timedelta

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest

from GeoCausality.power import PowerAnalysis
from GeoCausality.synthetic_control import SyntheticControl

N_DATES = 40
PRE_PERIOD = "2021-02-09"  # the whole series is clean history
TEST_GEOS = ["g0", "g1"]


def _make_history(seed: int = 0) -> pl.DataFrame:
    """Builds a clean geo panel (no treatment effect) with shared latent factors.

    Parameters
    ----------
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
        for di, d in enumerate(dates):
            y = 100.0 + loadings[g] @ factors[:, di] + rng.normal(0, 2)
            rows.append({"geo": g, "date": d, "y": float(y), "is_treatment": int(g in TEST_GEOS)})
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def history() -> pl.DataFrame:
    return _make_history()


def _power(history: pl.DataFrame, **overrides: object) -> PowerAnalysis:
    kwargs = {
        "geo_variable": "geo",
        "treatment_variable": "is_treatment",
        "date_variable": "date",
        "pre_period": PRE_PERIOD,
        "y_variable": "y",
        "alpha": 0.1,
        "estimator": SyntheticControl,
        "seed": 0,
    }
    kwargs.update(overrides)
    return PowerAnalysis(history, **kwargs)


class TestConstruction:
    @staticmethod
    def test_resolves_geos_from_treatment_variable(history: pl.DataFrame) -> None:
        pa = _power(history)
        assert set(pa.test_geos) == set(TEST_GEOS)
        assert set(pa.test_geos).isdisjoint(pa.control_geos)
        assert len(pa.control_geos) == 8

    @staticmethod
    def test_explicit_geo_lists_are_respected(history: pl.DataFrame) -> None:
        pa = _power(history, test_geos=["g0"], control_geos=["g5", "g6"])
        assert pa.test_geos == ["g0"]
        assert pa.control_geos == ["g5", "g6"]

    @staticmethod
    def test_history_stops_at_pre_period(history: pl.DataFrame) -> None:
        pa = _power(history, pre_period="2021-01-20")
        assert max(pa.history) <= "2021-01-20"
        assert len(pa.history) == 20

    @staticmethod
    def test_invalid_injection_raises(history: pl.DataFrame) -> None:
        with pytest.raises(ValueError):
            _power(history, injection="exponential")


class TestSimulate:
    @staticmethod
    def test_power_curve_structure(history: pl.DataFrame) -> None:
        pa = _power(history).simulate(effect_sizes=[0.0, 0.5], durations=[10], n_sims=10)
        assert pa.power_curve is not None
        assert len(pa.power_curve) == 2  # 2 effects x 1 duration
        for row in pa.power_curve:
            assert {"effect", "duration", "n_sims", "n_detected", "power"} <= row.keys()
            assert 0.0 <= row["power"] <= 1.0
            assert row["n_detected"] <= row["n_sims"]

    @staticmethod
    def test_detection_rises_with_effect(history: pl.DataFrame) -> None:
        pa = _power(history).simulate(effect_sizes=[0.0, 0.6], durations=[10], n_sims=15)
        power_by_effect = {r["effect"]: r["power"] for r in pa.power_curve}
        assert power_by_effect[0.6] > power_by_effect[0.0]

    @staticmethod
    def test_null_effect_is_calibrated(history: pl.DataFrame) -> None:
        # With no injected effect, detection should hover near the false-positive
        # rate alpha, not blow up. Allow generous slack for a small sim count.
        pa = _power(history, alpha=0.1).simulate(effect_sizes=[0.0], durations=[10], n_sims=20)
        assert pa.power_curve[0]["power"] <= 0.35

    @staticmethod
    def test_strong_effect_is_well_powered(history: pl.DataFrame) -> None:
        pa = _power(history).simulate(effect_sizes=[0.8], durations=[14], n_sims=15)
        assert pa.power_curve[0]["power"] >= 0.7

    @staticmethod
    def test_duration_too_long_raises(history: pl.DataFrame) -> None:
        with pytest.raises(ValueError):
            _power(history).simulate(effect_sizes=[0.5], durations=[N_DATES], n_sims=5)


class TestMDE:
    @staticmethod
    def test_mde_within_tested_range(history: pl.DataFrame) -> None:
        effects = [0.0, 0.2, 0.4, 0.6, 0.8]
        pa = _power(history).simulate(effect_sizes=effects, durations=[14], n_sims=15).mde(target_power=0.8)
        assert pa.mde_table is not None
        mde = pa.mde_table[0]["mde"]
        assert mde is None or (min(effects) <= mde <= max(effects))

    @staticmethod
    def test_mde_requires_simulate(history: pl.DataFrame) -> None:
        with pytest.raises(ValueError):
            _power(history).mde()

    @staticmethod
    def test_interpolate_mde_brackets_target() -> None:
        # power crosses 0.8 between effect 0.2 (0.6) and 0.4 (1.0): expect ~0.3.
        points = [(0.0, 0.05), (0.2, 0.6), (0.4, 1.0)]
        mde = PowerAnalysis._interpolate_mde(points, 0.8)
        assert mde == pytest.approx(0.3, abs=1e-9)

    @staticmethod
    def test_interpolate_mde_returns_none_when_unreached() -> None:
        points = [(0.0, 0.05), (0.2, 0.3), (0.4, 0.5)]
        assert PowerAnalysis._interpolate_mde(points, 0.8) is None


class TestReporting:
    @staticmethod
    def test_summarize_runs(history: pl.DataFrame) -> None:
        pa = _power(history).simulate(effect_sizes=[0.0, 0.5], durations=[10], n_sims=10).mde()
        with redirect_stdout(io.StringIO()) as buffer:
            pa.summarize()
        out = buffer.getvalue()
        assert "Power" in out
        assert "MDE" in out

    @staticmethod
    def test_summarize_requires_simulate(history: pl.DataFrame) -> None:
        with pytest.raises(ValueError):
            _power(history).summarize()

    @staticmethod
    def test_plot_builds_figure(history: pl.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
        shown = {}
        monkeypatch.setattr(go.Figure, "show", lambda self: shown.setdefault("ok", True))
        pa = _power(history).simulate(effect_sizes=[0.0, 0.5], durations=[10], n_sims=8).mde()
        pa.plot()
        assert shown.get("ok") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
