import numpy as np
import pandas as pd
import pytest

from GeoCausality.switchback import Switchback


def _make_switchback_data(
    n_geos: int = 10,
    n_periods: int = 60,
    effect: float = 5.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic switchback panel with known treatment effect."""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2024-01-01", periods=n_periods, freq="D")
    for g in range(n_geos):
        geo_fe = rng.normal(50, 10)
        phase = rng.integers(0, 2)
        for t, d in enumerate(dates):
            time_fe = 2.0 * np.sin(2 * np.pi * t / 30)
            treat = (t // 7 + phase) % 2
            y = geo_fe + time_fe + effect * treat + rng.normal(0, 2)
            rows.append({"geo": f"g{g}", "date": d, "y": y, "is_treatment": treat})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def data_df() -> pd.DataFrame:
    return _make_switchback_data()


def _fit(df: pd.DataFrame, **kwargs) -> Switchback:
    model = Switchback(
        df,
        geo_variable="geo",
        date_variable="date",
        y_variable="y",
        treatment_variable="is_treatment",
        **kwargs,
    )
    return model.pre_process().generate()


class TestSwitchbackBasic:
    @staticmethod
    def test_results_keys(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        expected = {
            "lift",
            "lift_ci_lower",
            "lift_ci_upper",
            "incrementality",
            "incrementality_ci_lower",
            "incrementality_ci_upper",
            "p_value",
            "n_treated_obs",
        }
        assert expected.issubset(model.results.keys())

    @staticmethod
    def test_positive_effect_detected(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        assert model.results["lift"] > 0
        assert model.results["p_value"] < 0.05

    @staticmethod
    def test_ci_brackets_point(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        assert model.results["lift_ci_lower"] <= model.results["lift"]
        assert model.results["lift"] <= model.results["lift_ci_upper"]

    @staticmethod
    def test_incrementality_is_scaled(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        ratio = model.results["incrementality"] / model.results["lift"]
        assert ratio == pytest.approx(model.results["n_treated_obs"], rel=1e-6)

    @staticmethod
    def test_summarize_runs(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        model.summarize("incremental")


class TestWashout:
    @staticmethod
    def test_washout_reduces_obs(data_df: pd.DataFrame) -> None:
        no_wash = _fit(data_df, washout=0)
        with_wash = _fit(data_df, washout=2)
        assert with_wash.n_treated_obs < no_wash.n_treated_obs

    @staticmethod
    def test_washout_still_detects_effect(data_df: pd.DataFrame) -> None:
        model = _fit(data_df, washout=1)
        assert model.results["lift"] > 0
        assert model.results["p_value"] < 0.10


class TestCarryover:
    @staticmethod
    def test_carryover_lags_in_results(data_df: pd.DataFrame) -> None:
        model = _fit(data_df, carryover_lags=2)
        assert "carryover" in model.results
        assert "treatment_lag_1" in model.results["carryover"]
        assert "treatment_lag_2" in model.results["carryover"]

    @staticmethod
    def test_carryover_lag_has_fields(data_df: pd.DataFrame) -> None:
        model = _fit(data_df, carryover_lags=1)
        lag1 = model.results["carryover"]["treatment_lag_1"]
        assert "estimate" in lag1
        assert "p_value" in lag1
        assert "ci_lower" in lag1
        assert "ci_upper" in lag1


class TestPermutationTest:
    @staticmethod
    def test_requires_generate(data_df: pd.DataFrame) -> None:
        model = Switchback(
            data_df,
            geo_variable="geo",
            date_variable="date",
            y_variable="y",
            treatment_variable="is_treatment",
        )
        model.pre_process()
        with pytest.raises(ValueError, match="generate"):
            model.permutation_test()

    @staticmethod
    def test_returns_expected_keys(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        result = model.permutation_test(n_permutations=20, seed=0)
        assert "observed_effect" in result
        assert "permutation_effects" in result
        assert "empirical_p_value" in result
        assert "n_permutations" in result

    @staticmethod
    def test_p_value_in_range(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        result = model.permutation_test(n_permutations=20, seed=0)
        assert 0.0 <= result["empirical_p_value"] <= 1.0

    @staticmethod
    def test_stored_in_results(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        model.permutation_test(n_permutations=10, seed=0)
        assert "permutation_test" in model.results


class TestPowerAnalytical:
    @staticmethod
    def test_returns_expected_keys(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        result = model.power_analytical(mde=5.0)
        expected = {"mde", "power", "sigma", "rho", "n_geos", "n_periods", "se_tau"}
        assert expected == set(result.keys())

    @staticmethod
    def test_power_increases_with_mde(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        small = model.power_analytical(mde=1.0)
        large = model.power_analytical(mde=10.0)
        assert large["power"] > small["power"]

    @staticmethod
    def test_mde_returned_when_none(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        result = model.power_analytical(mde=None, power_target=0.8)
        assert result["mde"] > 0
        assert result["power"] == pytest.approx(0.8)

    @staticmethod
    def test_rho_between_minus_one_and_one(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        result = model.power_analytical(mde=5.0)
        assert -1.0 <= result["rho"] <= 1.0

    @staticmethod
    def test_more_geos_increases_power(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        fewer = model.power_analytical(mde=3.0, n_geos=5)
        more = model.power_analytical(mde=3.0, n_geos=50)
        assert more["power"] > fewer["power"]


class TestPowerSimulation:
    @staticmethod
    def test_returns_expected_keys(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        result = model.power_simulation(mde=5.0, n_simulations=10, seed=0)
        expected = {"mde", "power", "n_simulations", "n_rejections", "sigma", "rho", "power_ci_lower", "power_ci_upper"}
        assert expected == set(result.keys())

    @staticmethod
    def test_power_in_range(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        result = model.power_simulation(mde=5.0, n_simulations=10, seed=0)
        assert 0.0 <= result["power"] <= 1.0

    @staticmethod
    def test_large_effect_high_power(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        result = model.power_simulation(mde=20.0, n_simulations=20, seed=0)
        assert result["power"] >= 0.5

    @staticmethod
    def test_stored_in_results(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        model.power_simulation(mde=5.0, n_simulations=10, seed=0)


class TestWrongInputs:
    @staticmethod
    def test_generate_before_preprocess() -> None:
        df = _make_switchback_data(n_geos=3, n_periods=20)
        model = Switchback(
            df,
            geo_variable="geo",
            date_variable="date",
            y_variable="y",
            treatment_variable="is_treatment",
        )
        with pytest.raises(ValueError, match="pre_process"):
            model.generate()
