"""Tests for the EconometricEstimator.diagnose() diagnostic method."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from GeoCausality.synthetic_control import SyntheticControl

N_GEOS = 8
N_PRE = 20
N_POST = 10
EFFECT = 30.0


def _make_panel(effect: float = EFFECT, seed: int = 0, n_pre: int = N_PRE) -> tuple[pl.DataFrame, str, str]:
    rng = np.random.default_rng(seed)
    d0 = date(2021, 1, 1)
    dates = [d0 + timedelta(days=i) for i in range(n_pre + N_POST)]
    geos = [f"g{i}" for i in range(N_GEOS)]
    factor = rng.normal(0, 1, n_pre + N_POST).cumsum()
    rows = []
    for g in geos:
        level = 50.0 + rng.normal(0, 5)
        for di, d in enumerate(dates):
            y = level + factor[di] + rng.normal(0, 2)
            if g == "g0" and di >= n_pre:
                y += effect
            rows.append({"geo": g, "date": d, "y": float(y)})
    pre = dates[n_pre - 1].isoformat()
    post = dates[n_pre].isoformat()
    return pl.DataFrame(rows), pre, post


def _fit(df: pl.DataFrame, pre: str, post: str) -> SyntheticControl:
    return (
        SyntheticControl(
            df,
            geo_variable="geo",
            test_geos=["g0"],
            date_variable="date",
            pre_period=pre,
            post_period=post,
            y_variable="y",
        )
        .pre_process()
        .generate()
    )


@pytest.fixture(scope="module")
def fitted() -> SyntheticControl:
    df, pre, post = _make_panel()
    return _fit(df, pre, post)


class TestDiagnoseBasic:
    @staticmethod
    def test_requires_generate() -> None:
        df, pre, post = _make_panel()
        model = SyntheticControl(
            df,
            geo_variable="geo",
            test_geos=["g0"],
            date_variable="date",
            pre_period=pre,
            post_period=post,
            y_variable="y",
        )
        with pytest.raises(ValueError, match="generate"):
            model.diagnose()

    @staticmethod
    def test_returns_expected_keys(fitted: SyntheticControl) -> None:
        diag = fitted.diagnose()
        expected = {"pre_mape", "pre_rmse", "durbin_watson", "residual_stationarity", "placebo_in_time"}
        assert expected == set(diag.keys())

    @staticmethod
    def test_stored_in_results(fitted: SyntheticControl) -> None:
        fitted.diagnose()
        assert "diagnostics" in fitted.results


class TestPrePeriodFit:
    @staticmethod
    def test_mape_is_finite_and_nonnegative(fitted: SyntheticControl) -> None:
        diag = fitted.diagnose()
        assert np.isfinite(diag["pre_mape"])
        assert diag["pre_mape"] >= 0.0

    @staticmethod
    def test_rmse_is_finite_and_nonnegative(fitted: SyntheticControl) -> None:
        diag = fitted.diagnose()
        assert np.isfinite(diag["pre_rmse"])
        assert diag["pre_rmse"] >= 0.0

    @staticmethod
    def test_good_fit_has_low_mape(fitted: SyntheticControl) -> None:
        diag = fitted.diagnose()
        assert diag["pre_mape"] < 0.20


class TestDurbinWatson:
    @staticmethod
    def test_dw_in_valid_range(fitted: SyntheticControl) -> None:
        diag = fitted.diagnose()
        assert 0.0 <= diag["durbin_watson"] <= 4.0


class TestStationarity:
    @staticmethod
    def test_stationarity_keys(fitted: SyntheticControl) -> None:
        diag = fitted.diagnose()
        stat = diag["residual_stationarity"]
        assert {"adf_statistic", "adf_p_value", "is_stationary"} == set(stat.keys())

    @staticmethod
    def test_adf_values_are_finite(fitted: SyntheticControl) -> None:
        diag = fitted.diagnose()
        stat = diag["residual_stationarity"]
        assert np.isfinite(stat["adf_statistic"])
        assert np.isfinite(stat["adf_p_value"])

    @staticmethod
    def test_short_pre_returns_nan() -> None:
        df, pre, post = _make_panel(n_pre=5)
        model = _fit(df, pre, post)
        diag = model.diagnose()
        assert np.isnan(diag["residual_stationarity"]["adf_statistic"])
        assert diag["residual_stationarity"]["is_stationary"] is None


class TestPlaceboInTime:
    @staticmethod
    def test_placebo_returns_dict(fitted: SyntheticControl) -> None:
        diag = fitted.diagnose()
        pit = diag["placebo_in_time"]
        assert pit is not None
        assert {"lift", "p_value", "fake_pre", "fake_post"} == set(pit.keys())

    @staticmethod
    def test_placebo_lift_is_near_zero(fitted: SyntheticControl) -> None:
        diag = fitted.diagnose()
        pit = diag["placebo_in_time"]
        assert abs(pit["lift"]) < 20.0

    @staticmethod
    def test_placebo_p_value_not_extreme(fitted: SyntheticControl) -> None:
        diag = fitted.diagnose()
        pit = diag["placebo_in_time"]
        assert pit["p_value"] > 0.01

    @staticmethod
    def test_short_pre_returns_none() -> None:
        df, pre, post = _make_panel(n_pre=8)
        model = _fit(df, pre, post)
        diag = model.diagnose()
        assert diag["placebo_in_time"] is None
