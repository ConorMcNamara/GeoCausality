"""Tests for the Abadie-style randomization inference on EconometricEstimator."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from GeoCausality.synthetic_control import SyntheticControl

TREATED = "g0"
N_GEOS = 8
N_PRE = 20
N_POST = 10
EFFECT = 30.0


def _make_panel(effect: float = EFFECT, seed: int = 0) -> tuple[pl.DataFrame, str, str]:
    rng = np.random.default_rng(seed)
    d0 = date(2021, 1, 1)
    dates = [d0 + timedelta(days=i) for i in range(N_PRE + N_POST)]
    geos = [f"g{i}" for i in range(N_GEOS)]
    factor = rng.normal(0, 1, N_PRE + N_POST).cumsum()

    rows = []
    for g in geos:
        level = 50.0 + rng.normal(0, 5)
        for di, d in enumerate(dates):
            y = level + factor[di] + rng.normal(0, 2)
            if g == TREATED and di >= N_PRE:
                y += effect
            rows.append({"geo": g, "date": d, "y": float(y)})
    pre = dates[N_PRE - 1].isoformat()
    post = dates[N_PRE].isoformat()
    return pl.DataFrame(rows), pre, post


def _fit(df: pl.DataFrame, pre: str, post: str) -> SyntheticControl:
    return (
        SyntheticControl(
            df,
            geo_variable="geo",
            test_geos=[TREATED],
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


class TestRandomizationBasic:
    @staticmethod
    def test_requires_generate() -> None:
        df, pre, post = _make_panel()
        model = SyntheticControl(
            df,
            geo_variable="geo",
            test_geos=[TREATED],
            date_variable="date",
            pre_period=pre,
            post_period=post,
            y_variable="y",
        )
        with pytest.raises(ValueError, match="generate"):
            model.randomization_test()

    @staticmethod
    def test_returns_expected_keys(fitted: SyntheticControl) -> None:
        result = fitted.randomization_test()
        expected = {
            "observed_statistic",
            "placebo_statistics",
            "placebo_geos",
            "p_value",
            "n_placebos",
            "n_failed",
            "statistic",
        }
        assert expected == set(result.keys())

    @staticmethod
    def test_n_placebos_equals_donor_count(fitted: SyntheticControl) -> None:
        result = fitted.randomization_test()
        assert result["n_placebos"] == N_GEOS - 1
        assert len(result["placebo_statistics"]) == N_GEOS - 1
        assert len(result["placebo_geos"]) == N_GEOS - 1

    @staticmethod
    def test_p_value_in_range(fitted: SyntheticControl) -> None:
        result = fitted.randomization_test()
        assert 0.0 < result["p_value"] <= 1.0

    @staticmethod
    def test_stored_in_results(fitted: SyntheticControl) -> None:
        fitted.randomization_test()
        assert "randomization_test" in fitted.results

    @staticmethod
    def test_treated_geo_not_in_placebos(fitted: SyntheticControl) -> None:
        result = fitted.randomization_test()
        assert TREATED not in result["placebo_geos"]


class TestStatistics:
    @staticmethod
    def test_avg_lift_default(fitted: SyntheticControl) -> None:
        result = fitted.randomization_test(statistic="avg_lift")
        assert result["statistic"] == "avg_lift"
        assert result["observed_statistic"] > 0

    @staticmethod
    def test_sum_lift(fitted: SyntheticControl) -> None:
        result = fitted.randomization_test(statistic="sum_lift")
        assert result["statistic"] == "sum_lift"
        assert result["observed_statistic"] > 0

    @staticmethod
    def test_mspe_ratio(fitted: SyntheticControl) -> None:
        result = fitted.randomization_test(statistic="mspe_ratio")
        assert result["statistic"] == "mspe_ratio"
        assert result["observed_statistic"] > 1.0

    @staticmethod
    def test_invalid_statistic_raises(fitted: SyntheticControl) -> None:
        with pytest.raises(ValueError, match="statistic"):
            fitted.randomization_test(statistic="invalid")


class TestMaxPlacebos:
    @staticmethod
    def test_caps_number_of_placebos(fitted: SyntheticControl) -> None:
        result = fitted.randomization_test(max_placebos=3, seed=0)
        assert result["n_placebos"] == 3

    @staticmethod
    def test_seed_reproducibility(fitted: SyntheticControl) -> None:
        a = fitted.randomization_test(max_placebos=3, seed=42)
        b = fitted.randomization_test(max_placebos=3, seed=42)
        assert a["placebo_geos"] == b["placebo_geos"]
        assert a["placebo_statistics"] == b["placebo_statistics"]


class TestSignificance:
    @staticmethod
    def test_strong_effect_is_significant() -> None:
        df, pre, post = _make_panel(effect=50.0)
        model = _fit(df, pre, post)
        result = model.randomization_test()
        assert result["p_value"] <= 0.25

    @staticmethod
    def test_null_effect_not_extreme() -> None:
        df, pre, post = _make_panel(effect=0.0, seed=1)
        model = _fit(df, pre, post)
        result = model.randomization_test()
        assert result["p_value"] > 0.05
