import narwhals as nw
import numpy as np
import pandas as pd
import pytest
from linearmodels.panel import PanelOLS

import GeoCausality.switchback as switchback_module
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
        assert with_wash.panel.shape[0] < no_wash.panel.shape[0]
        # Washed-out periods were still treated, so they count toward the
        # total incrementality.
        assert with_wash.n_treated_obs == no_wash.n_treated_obs

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


def _legacy_permutation_test(model: Switchback, n_permutations: int, seed: int) -> dict:
    """The original ``permutation_test``: a per-geo roll and a full ``PanelOLS`` refit per permutation."""
    rng = np.random.default_rng(seed)
    full_pd = model.data.sort([model.geo_variable, model.date_variable]).to_pandas()
    geos = full_pd[model.geo_variable].unique()
    treat_var = model.treatment_variable
    exog = " + ".join([treat_var, *model._lag_terms()])
    formula = f"{model.y_variable} ~ {exog} + EntityEffects + TimeEffects"
    perm_effects = []
    for _ in range(n_permutations):
        perm_df = full_pd.copy()
        for geo in geos:
            mask = perm_df[model.geo_variable] == geo
            treat_vals = perm_df.loc[mask, treat_var].values
            shift = rng.integers(1, len(treat_vals))
            perm_df.loc[mask, treat_var] = np.roll(treat_vals, shift)
        perm_panel = model._apply_washout(model._add_lags(nw.from_native(perm_df, eager_only=True)))
        perm_indexed = perm_panel.to_pandas().set_index([model.geo_variable, model.date_variable])
        try:
            perm_fit = PanelOLS.from_formula(formula, data=perm_indexed).fit(cov_type="clustered", cluster_entity=True)
            perm_effects.append(float(perm_fit.params[treat_var]))
        except Exception:
            continue
    perm_arr = np.asarray(perm_effects, dtype=float)
    # Tie-tolerant, as in ``permutation_test``: even PanelOLS reproduces a
    # flipped schedule's coefficient only up to rounding.
    n_extreme = np.sum(np.abs(perm_arr) >= abs(model.results["lift"]) * (1 - 1e-9))
    return {"permutation_effects": perm_arr, "empirical_p_value": float((n_extreme + 1) / (len(perm_arr) + 1))}


def _unbalanced(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Drop a random 10% of rows, so geos have different (gappy) date sets."""
    return df.sample(frac=0.9, random_state=seed).sort_index()


class TestPermutationTestEquivalence:
    """The closed-form refit reproduces the original PanelOLS-per-permutation results."""

    @staticmethod
    @pytest.mark.parametrize("balanced", [True, False])
    @pytest.mark.parametrize("washout", [0, 1, 2])
    @pytest.mark.parametrize("carryover_lags", [0, 1, 2])
    @pytest.mark.parametrize("seed", [0, 1])
    def test_matches_legacy(balanced: bool, washout: int, carryover_lags: int, seed: int) -> None:
        df = _make_switchback_data(n_geos=8, n_periods=40, seed=seed)
        if not balanced:
            df = _unbalanced(df, seed)
        model = _fit(df, washout=washout, carryover_lags=carryover_lags)
        new = model.permutation_test(n_permutations=10, seed=seed)
        old = _legacy_permutation_test(model, n_permutations=10, seed=seed)
        assert new["n_permutations"] == len(old["permutation_effects"])
        np.testing.assert_allclose(new["permutation_effects"], old["permutation_effects"], rtol=1e-8, atol=1e-8)
        assert new["empirical_p_value"] == old["empirical_p_value"]

    @staticmethod
    @pytest.mark.parametrize("n_geos, n_periods, seed", [(4, 6, 0), (3, 8, 1), (2, 12, 2)])
    def test_schedule_ties_count_as_extreme(n_geos: int, n_periods: int, seed: int) -> None:
        """Shifts that reproduce or flip an alternating schedule tie the observed effect.

        With few geos and short periodic schedules such ties are common; all of
        them must count as extreme even though the closed-form refit differs
        from the observed ``PanelOLS`` fit by rounding.
        """
        rng = np.random.default_rng(seed)
        rows = []
        for g in range(n_geos):
            geo_fe = rng.normal(50, 10)
            for t in range(n_periods):
                treat = (t + g) % 2
                y = geo_fe + np.sin(t) + 3.0 * treat + rng.normal(0, 2)
                rows.append({"geo": f"g{g}", "date": t, "y": y, "is_treatment": treat})
        model = _fit(pd.DataFrame(rows))
        new = model.permutation_test(n_permutations=200, seed=seed)
        old = _legacy_permutation_test(model, n_permutations=200, seed=seed)

        # Replay the draws: the schedule is reproduced when every geo's shift is
        # even and flipped (coefficient -observed) when every shift is odd.
        # Shifts that put every geo in the same phase leave a treatment absorbed
        # by the time effects, which PanelOLS cannot fit, so they are skipped.
        draws = np.random.default_rng(seed)
        parities = np.array([draws.integers(1, np.full(n_geos, n_periods)) % 2 for _ in range(200)])
        phases = (parities + np.arange(n_geos)) % 2
        parities = parities[np.any(phases != phases[:, :1], axis=1)]
        is_tie = np.all(parities == parities[:, :1], axis=1)
        assert new["n_permutations"] == len(parities) == len(old["permutation_effects"])
        observed = abs(new["observed_effect"])
        effects = np.abs(new["permutation_effects"])
        assert is_tie.any()
        np.testing.assert_allclose(effects[is_tie], observed, rtol=1e-9)
        n_extreme = is_tie.sum() + np.sum(effects[~is_tie] > observed)
        assert new["empirical_p_value"] == (n_extreme + 1) / (len(effects) + 1)
        assert new["empirical_p_value"] == old["empirical_p_value"]

    @staticmethod
    def test_shuffle_matches_per_geo_roll(monkeypatch: pytest.MonkeyPatch) -> None:
        df = _unbalanced(_make_switchback_data(n_geos=6, n_periods=30), seed=3)
        model = _fit(df)
        schedules = []
        original = Switchback._schedule_design

        def recording(self: Switchback, treat: np.ndarray, position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            schedules.append(treat.copy())
            return original(self, treat, position)

        monkeypatch.setattr(Switchback, "_schedule_design", recording)
        model.permutation_test(n_permutations=5, seed=7)
        assert len(schedules) == 5

        rng = np.random.default_rng(7)
        full_pd = df.sort_values(["geo", "date"])
        for schedule in schedules:
            expected = []
            for geo in full_pd["geo"].unique():
                vals = full_pd.loc[full_pd["geo"] == geo, "is_treatment"].to_numpy()
                expected.append(np.roll(vals, rng.integers(1, len(vals))))
            np.testing.assert_array_equal(schedule, np.concatenate(expected))

    @staticmethod
    @pytest.mark.parametrize("washout, carryover_lags", [(0, 0), (1, 0), (2, 1), (0, 2), (1, 2)])
    def test_each_coefficient_matches_panelols(washout: int, carryover_lags: int) -> None:
        df = _unbalanced(_make_switchback_data(n_geos=12, n_periods=50, seed=5), seed=5)
        model = _fit(df, washout=washout, carryover_lags=carryover_lags)
        full_pd = df.sort_values(["geo", "date"]).reset_index(drop=True)
        y, entity, time = model._fast_refit_inputs(full_pd, "is_treatment")
        position = full_pd.groupby("geo").cumcount().to_numpy()
        formula = f"y ~ {' + '.join(['is_treatment', *model._lag_terms()])} + EntityEffects + TimeEffects"
        rng = np.random.default_rng(0)
        for _ in range(10):
            treat = rng.permutation(full_pd["is_treatment"].to_numpy())
            x, keep = model._schedule_design(treat.astype(float), position)
            fast = model._twfe_treatment_coefficient(y[keep], x[keep], entity[keep], time[keep])
            reference = model._panelols_permutation_effect(full_pd.assign(is_treatment=treat), formula)
            assert fast is not None
            assert fast == pytest.approx(reference, rel=1e-8, abs=1e-8)

    @staticmethod
    def test_absorbed_treatment_falls_back() -> None:
        """A treatment that varies only over time is absorbed: PanelOLS decides (and fails)."""
        rng = np.random.default_rng(0)
        entity = np.repeat(np.arange(5), 20)
        time = np.tile(np.arange(20), 5)
        x = ((time // 4) % 2).astype(float)[:, None]
        assert Switchback._twfe_treatment_coefficient(rng.normal(size=100), x, entity, time) is None

    @staticmethod
    def test_disconnected_effects_fall_back() -> None:
        """Two groups of geos observed on disjoint dates do not identify the effects jointly."""
        rng = np.random.default_rng(0)
        entity = np.repeat(np.arange(6), 10)
        time = np.tile(np.arange(10), 6) + np.where(entity < 3, 0, 10)
        x = rng.integers(0, 2, size=60).astype(float)[:, None]
        assert Switchback._twfe_treatment_coefficient(rng.normal(size=60), x, entity, time) is None

    @staticmethod
    @pytest.mark.filterwarnings("ignore::linearmodels.shared.exceptions.MissingValueWarning")
    def test_missing_outcome_uses_panelols() -> None:
        df = _make_switchback_data(n_geos=6, n_periods=30)
        df.loc[[3, 50, 77], "y"] = np.nan
        model = _fit(df, washout=1)
        assert model._fast_refit_inputs(df.sort_values(["geo", "date"]), "is_treatment") is None
        new = model.permutation_test(n_permutations=5, seed=0)
        old = _legacy_permutation_test(model, n_permutations=5, seed=0)
        assert new["n_permutations"] == 5
        np.testing.assert_allclose(new["permutation_effects"], old["permutation_effects"], rtol=1e-12)
        assert new["empirical_p_value"] == old["empirical_p_value"]


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
        fewer = model.power_analytical(mde=0.5, n_geos=5)
        more = model.power_analytical(mde=0.5, n_geos=50)
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
    def test_rejects_negative_sigma(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        with pytest.raises(ValueError, match="sigma must be non-negative"):
            model.power_simulation(mde=5.0, n_simulations=2, sigma=-1.0)

    @staticmethod
    def test_large_effect_high_power(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        result = model.power_simulation(mde=20.0, n_simulations=20, seed=0)
        assert result["power"] >= 0.5

    @staticmethod
    def test_large_effect_two_geos_not_capped(data_df: pd.DataFrame) -> None:
        """With random phases, ~half of two-geo panels were collinear with TimeEffects."""
        model = _fit(data_df)
        result = model.power_simulation(mde=50.0, n_geos=2, n_simulations=20, seed=0)
        assert result["power"] > 0.8

    @staticmethod
    def test_stored_in_results(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        model.power_simulation(mde=5.0, n_simulations=10, seed=0)

    @staticmethod
    def test_simulated_panels_match_scalar_draws(data_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
        """The vectorised panels equal the original row-by-row construction, bit for bit."""
        model = _fit(data_df)
        n_geos, n_periods, sigma, rho, mde = 3, 17, 1.3, 0.6, -2.0
        frames = []

        class Recorder:
            @staticmethod
            def from_formula(formula: str, data: pd.DataFrame) -> None:
                frames.append(data.reset_index())
                raise RuntimeError("recorded")

        monkeypatch.setattr(switchback_module, "PanelOLS", Recorder)
        model.power_simulation(
            mde=mde, n_simulations=3, n_geos=n_geos, n_periods=n_periods, sigma=sigma, rho=rho, switching_freq=3, seed=4
        )
        assert len(frames) == 3
        schedule = Switchback._schedule(n_periods, 0.5, 3, np.arange(n_geos) % 2)
        for frame, sim_seed in zip(frames, np.random.SeedSequence(4).generate_state(3), strict=True):
            sim_rng = np.random.default_rng(int(sim_seed))
            rows = []
            for g in range(n_geos):
                geo_fe = sim_rng.normal(0, 10)
                noise = np.zeros(n_periods)
                noise[0] = sim_rng.normal(0, sigma)
                for t in range(1, n_periods):
                    noise[t] = rho * noise[t - 1] + sim_rng.normal(0, sigma * np.sqrt(1 - rho**2))
                for t in range(n_periods):
                    treat = int(schedule[g, t])
                    y = geo_fe + 2.0 * np.sin(2 * np.pi * t / 30) + mde * treat + noise[t]
                    rows.append({"geo": f"g{g}", "date": t, "is_treatment": treat, "y": y})
            pd.testing.assert_frame_equal(frame, pd.DataFrame(rows), check_exact=True)


class TestParallel:
    @staticmethod
    def test_parallel_power_simulation(data_df: pd.DataFrame) -> None:
        model = _fit(data_df)
        seq = model.power_simulation(mde=5.0, n_simulations=10, seed=0, n_jobs=1)
        par = model.power_simulation(mde=5.0, n_simulations=10, seed=0, n_jobs=2)
        assert seq["n_rejections"] == par["n_rejections"]
        assert seq["power"] == par["power"]


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
