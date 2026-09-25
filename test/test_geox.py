import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from GeoCausality import geox

LIFT_TYPES = ("incremental", "absolute", "relative", "cost-per", "revenue", "roas")


def _fit(data_df: pd.DataFrame) -> geox.GeoX:
    model = geox.GeoX(
        data_df,
        geo_variable="zipcode",
        treatment_variable="is_test",
        date_variable="date",
        pre_period="2022-06-30",
        post_period="2022-07-01",
        y_variable="orders",
        msrp=7.00,
        spend=500_000,
    )
    return model.pre_process().generate()


# The zipcodes dataset is vendored (gzipped) under test/data so the suite runs
# offline and deterministically rather than downloading it over the network at
# runtime. Source:
# https://raw.githubusercontent.com/juanitorduz/website_projects/master/data/zipcodes_data.csv
DATA_PATH = Path(__file__).parent / "data" / "zipcodes_data.csv.gz"


@pytest.fixture(scope="module")
def data_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df["is_test"] = df["variant"] == "treatment"
    return df


class TestGeoX:
    @staticmethod
    def test_geox_results(data_df: pd.DataFrame) -> None:
        geo_x = geox.GeoX(
            data_df,
            geo_variable="zipcode",
            treatment_variable="is_test",
            date_variable="date",
            pre_period="2022-06-30",
            post_period="2022-07-01",
            y_variable="orders",
            msrp=7.00,
            spend=500_000,
        )
        geo_x.pre_process().generate()
        # tbr = TBR()
        # tbr.fit(data_df, target="orders", key_geo="zipcode", key_group="variant", key_period="is_campaign",
        # key_date="date", group_control="control", group_treatment="treatment", period_test=True, period_pre=False)
        # tbr.summary(tails=2, report='last')
        #
        # {"date":{"0":1659225600000},"estimate":{"0":170826.0173391506},"precision":{"0":4131.0130609572},
        # "lower":{"0":166695.0042781934},"upper":{"0":174957.0304001078},"scale":{"0":2485.3351529485},
        # "level":{"0":0.9},"probability":{"0":1.0},"posterior_threshold":{"0":0.0},"p_value":{"0":3.22162521e-79}}'
        results = pd.DataFrame(geo_x.results).iloc[-1]
        estimate = 170826.0173391506
        ci_upper = 174957.0304001078
        ci_lower = 166695.0042781934
        p_value = 3.22162521e-79
        assert results["cumulative_incrementality"] == pytest.approx(
            estimate,
            abs=1e-6,
        )
        assert results["cumulative_incrementality_ci_upper"] == pytest.approx(ci_upper, abs=1e-6)
        assert results["cumulative_incrementality_ci_lower"] == pytest.approx(ci_lower, abs=1e-6)
        assert results["p_value"] == pytest.approx(p_value, abs=1e-10)

    @staticmethod
    def test_geox_wrongInputs(data_df: pd.DataFrame) -> None:
        geo_x = geox.GeoX(
            data_df,
            geo_variable="zipcode",
            treatment_variable="is_test",
            date_variable="date",
            pre_period="2022-06-30",
            post_period="2022-07-01",
            y_variable="orders",
            msrp=7.00,
            spend=500_000,
        )
        geo_x.pre_process().generate()
        with pytest.raises(ValueError, match="Cannot measure blarg"):
            geo_x.summarize(lift="blarg")

    @staticmethod
    @pytest.mark.parametrize("lift", LIFT_TYPES)
    def test_summarize_runs_for_all_lift_types(lift: str, data_df: pd.DataFrame) -> None:
        # Regression: results["test"]/["counterfactual"] must be numpy arrays so the
        # summary sums do not hit narwhals Series.sum(axis=...).
        geo_x = _fit(data_df)
        with redirect_stdout(io.StringIO()) as buffer:
            geo_x.summarize(lift)
        assert buffer.getvalue().strip()

    @staticmethod
    def test_plot_runs(data_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
        # Regression: the post-period date filter must handle a datetime/Timestamp
        # date column (data_df parses dates), not only ISO strings.
        geo_x = _fit(data_df)
        monkeypatch.setattr(go.Figure, "show", lambda self: None)
        geo_x.plot()


def _make_negative_slope_data() -> pd.DataFrame:
    """Synthetic panel where the natural OLS slope (control -> test) is negative."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-01-01", periods=60, freq="D")
    rows = []
    for d in dates:
        control_y = 100.0 + rng.normal(0, 1)
        test_y = 200.0 - 0.5 * control_y + rng.normal(0, 1)
        rows.append({"geo": "c1", "date": d, "orders": control_y, "is_test": False})
        rows.append({"geo": "t1", "date": d, "orders": test_y, "is_test": True})
    return pd.DataFrame(rows)


class TestNonNegativeSlope:
    @staticmethod
    def test_slope_clamped_when_negative() -> None:
        df = _make_negative_slope_data()
        model = geox.GeoX(
            df,
            geo_variable="geo",
            treatment_variable="is_test",
            date_variable="date",
            pre_period="2021-02-14",
            post_period="2021-02-15",
            y_variable="orders",
            non_negative=True,
        )
        model.pre_process().generate()
        assert model.model.params[1] >= 0.0

    @staticmethod
    def test_negative_slope_allowed_when_disabled() -> None:
        df = _make_negative_slope_data()
        model = geox.GeoX(
            df,
            geo_variable="geo",
            treatment_variable="is_test",
            date_variable="date",
            pre_period="2021-02-14",
            post_period="2021-02-15",
            y_variable="orders",
            non_negative=False,
        )
        model.pre_process().generate()
        assert model.model.params[1] < 0.0


class TestAlternative:
    @staticmethod
    def test_invalid_alternative_raises(data_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="alternative"):
            geox.GeoX(data_df, alternative="both")

    @staticmethod
    def test_two_sided_default(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        p = model.results["p_value"][-1]
        # Two-sided p-value is always <= 1.
        assert 0.0 <= p <= 1.0

    @staticmethod
    def test_greater_returns_one_sided(data_df: pd.DataFrame) -> None:
        model = geox.GeoX(
            data_df,
            geo_variable="zipcode",
            treatment_variable="is_test",
            date_variable="date",
            pre_period="2022-06-30",
            post_period="2022-07-01",
            y_variable="orders",
            alternative="greater",
        )
        model.pre_process().generate()
        p_greater = model.results["p_value"][-1]
        # This dataset has a large positive treatment effect, so the one-sided
        # "greater" p-value should be tiny and <= the two-sided p-value.
        two_sided = _fit(data_df)
        p_two = two_sided.results["p_value"][-1]
        assert p_greater <= p_two
        assert p_greater < 0.01

    @staticmethod
    def test_less_returns_one_sided(data_df: pd.DataFrame) -> None:
        model = geox.GeoX(
            data_df,
            geo_variable="zipcode",
            treatment_variable="is_test",
            date_variable="date",
            pre_period="2022-06-30",
            post_period="2022-07-01",
            y_variable="orders",
            alternative="less",
        )
        model.pre_process().generate()
        p_less = model.results["p_value"][-1]
        # Positive treatment effect tested against "less" should be near 1.
        assert p_less > 0.5

    @staticmethod
    def test_greater_ci_lower_bound_only(data_df: pd.DataFrame) -> None:
        model = geox.GeoX(
            data_df,
            geo_variable="zipcode",
            treatment_variable="is_test",
            date_variable="date",
            pre_period="2022-06-30",
            post_period="2022-07-01",
            y_variable="orders",
            alternative="greater",
        )
        model.pre_process().generate()
        assert all(np.isinf(model.results["cumulative_incrementality_ci_upper"]))
        assert all(np.isfinite(model.results["cumulative_incrementality_ci_lower"]))


class TestValidationRMSE:
    @staticmethod
    def test_rmse_present_in_results(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        assert "validation_rmse" in model.results
        assert model.results["validation_rmse"] > 0.0

    @staticmethod
    def test_rmse_is_finite(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        assert np.isfinite(model.results["validation_rmse"])


class TestPercentLiftCIs:
    @staticmethod
    def test_percent_lift_keys_present(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        for key in ("percent_lift", "percent_lift_ci_lower", "percent_lift_ci_upper"):
            assert key in model.results

    @staticmethod
    def test_ci_brackets_point_estimate(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        lift = model.results["percent_lift"]
        lo = model.results["percent_lift_ci_lower"]
        hi = model.results["percent_lift_ci_upper"]
        valid = ~np.isnan(lift)
        assert np.all(lo[valid] <= lift[valid])
        assert np.all(lift[valid] <= hi[valid])

    @staticmethod
    def test_percent_lift_positive_for_positive_effect(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        lift = model.results["percent_lift"]
        valid = ~np.isnan(lift)
        assert np.all(lift[valid] > 0)


class TestPlaceboTest:
    @staticmethod
    def test_requires_generate(data_df: pd.DataFrame) -> None:
        model = geox.GeoX(
            data_df,
            geo_variable="zipcode",
            treatment_variable="is_test",
            date_variable="date",
            pre_period="2022-06-30",
            post_period="2022-07-01",
            y_variable="orders",
        )
        model.pre_process()
        with pytest.raises(ValueError, match="generate"):
            model.placebo_test()

    @staticmethod
    def test_returns_expected_keys(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        result = model.placebo_test(n_placebos=10, min_placebo_r2=0.0)
        assert "placebo_t_stats" in result
        assert "real_t_stat" in result
        assert "empirical_p_value" in result
        assert "n_placebos_passed" in result
        assert "n_placebos_total" in result

    @staticmethod
    def test_stored_in_results(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        model.placebo_test(n_placebos=10, min_placebo_r2=0.0)
        assert "placebo" in model.results

    @staticmethod
    def test_empirical_p_value_in_range(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        result = model.placebo_test(n_placebos=20, min_placebo_r2=0.0)
        assert 0.0 <= result["empirical_p_value"] <= 1.0

    @staticmethod
    def test_r2_filter_reduces_placebos(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        no_filter = model.placebo_test(n_placebos=30, min_placebo_r2=0.0, seed=0)
        with_filter = model.placebo_test(n_placebos=30, min_placebo_r2=0.9, seed=0)
        assert with_filter["n_placebos_passed"] <= no_filter["n_placebos_passed"]

    @staticmethod
    def test_real_t_stat_is_finite(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        result = model.placebo_test(n_placebos=5, min_placebo_r2=0.0)
        assert np.isfinite(result["real_t_stat"])


class TestRandomizationInference:
    @staticmethod
    def test_requires_generate(data_df: pd.DataFrame) -> None:
        model = geox.GeoX(
            data_df,
            geo_variable="zipcode",
            treatment_variable="is_test",
            date_variable="date",
            pre_period="2022-06-30",
            post_period="2022-07-01",
            y_variable="orders",
        )
        with pytest.raises(ValueError, match="generate"):
            model.randomization_test()

    @staticmethod
    def test_returns_expected_keys(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        result = model.randomization_test(max_placebos=5, seed=0)
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
    def test_n_placebos_capped(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        result = model.randomization_test(max_placebos=5, seed=0)
        assert result["n_placebos"] <= 5

    @staticmethod
    def test_p_value_in_range(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        result = model.randomization_test(max_placebos=5, seed=0)
        assert 0.0 < result["p_value"] <= 1.0

    @staticmethod
    def test_stored_in_results(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        model.randomization_test(max_placebos=5, seed=0)
        assert "randomization_test" in model.results

    @staticmethod
    def test_sum_lift_statistic(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        result = model.randomization_test(statistic="sum_lift", max_placebos=5, seed=0)
        assert result["statistic"] == "sum_lift"

    @staticmethod
    def test_invalid_statistic_raises(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        with pytest.raises(ValueError, match="statistic"):
            model.randomization_test(statistic="mspe_ratio")

    @staticmethod
    def test_seed_reproducibility(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        a = model.randomization_test(max_placebos=5, seed=42)
        b = model.randomization_test(max_placebos=5, seed=42)
        assert a["placebo_geos"] == b["placebo_geos"]
        assert a["placebo_statistics"] == b["placebo_statistics"]

    @staticmethod
    def test_parallel_matches_sequential(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        seq = model.randomization_test(max_placebos=5, seed=0, n_jobs=1)
        par = model.randomization_test(max_placebos=5, seed=0, n_jobs=2)
        assert seq["placebo_geos"] == par["placebo_geos"]
        assert seq["placebo_statistics"] == pytest.approx(par["placebo_statistics"])
        assert seq["p_value"] == par["p_value"]


if __name__ == "__main__":
    pytest.main()
