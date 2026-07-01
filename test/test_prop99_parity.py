"""Parity test: our synthetic-control family vs Abadie et al. (2010) Prop 99.

Validates GeoCausality's four synthetic-control estimators against the canonical
benchmark of the literature: California's Proposition 99 tobacco-control program.
California passed Prop 99 in 1988, so it is treated from 1989; the other 38
states form the donor pool. Abadie, Diamond & Hainmueller (2010) estimate that
by 2000 California's per-capita cigarette sales were about **26 packs** below its
synthetic control, with an **average post-period (1989-2000) gap of roughly -19
to -20 packs**.

The vendored ``prop99_smoking.csv`` is the ``smoking`` dataset from the R
``Synth`` package (39 states x 1970-2000); see ``prop99_smoking.README.txt`` for
provenance and ``vendor_prop99.py`` for the download. Until that CSV is present
this test skips, exactly like the GeoLift / Card-Krueger parity tests.

What we observe (and assert), per estimator -- the average post-period gap:

* ``SyntheticControl``          -19.5 -> matches the published ~-19.5 closely.
* ``AugmentedSyntheticControl`` -15.8 -> same sign and ballpark (mildly
                                         attenuated by the ridge augmentation).
* ``PenalizedSyntheticControl`` -23.5 -> matches after #31 was fixed (it now
                                         tracks the full pre-period trajectory
                                         with a scaled Abadie & L'Hour penalty,
                                         rather than matching only the pre-period
                                         mean).
* ``GeneralizedSyntheticControl`` -20.7 -> matches after #32 was fixed (the
                                         factor count is now chosen by the
                                         eigenvalue-ratio criterion, which no
                                         longer over-selects and washes out the
                                         effect).

All four originally diverged in interesting ways; the two that were off
(PenalizedSyntheticControl #31, GeneralizedSyntheticControl #32) were genuine
reimplementation bugs that this parity test surfaced -- exactly as the GeoLift
parity test surfaced #20. Tolerances are deliberately generous: this guards
against gross divergence, not exact equality, since our predictor specification
and solvers differ from Abadie's exact Prop 99 setup.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from GeoCausality.augmented_synthetic_control import AugmentedSyntheticControl
from GeoCausality.generalized_synthetic_control import GeneralizedSyntheticControl
from GeoCausality.penalized_synthetic_control import PenalizedSyntheticControl
from GeoCausality.robust_synthetic_control import RobustSyntheticControl
from GeoCausality.synthetic_control import SyntheticControl, SyntheticControlV

DATA_PATH = Path(__file__).parent / "data" / "prop99_smoking.csv"

TREATED = "California"
PRE_PERIOD = "1988"  # last untreated year
POST_PERIOD = "1989"  # first treated year (Prop 99 passed 1988)

# Published Prop 99 effect (Abadie, Diamond & Hainmueller 2010).
REF_AVG_GAP = -19.5  # average post-period (1989-2000) gap, packs/capita
REF_TERMINAL_GAP = -26.0  # year-2000 gap, packs/capita

# Reimplementation tolerances (not a port): confirm sign + magnitude, not
# equality. The "matches" bar is generous; SyntheticControl additionally gets a
# tight check because it tracks Abadie almost exactly.
GAP_ABS_TOL = 8.0  # packs -- generous "right ballpark" band
SC_AVG_ABS_TOL = 4.0  # packs -- tight band for the plain estimator
SC_TERMINAL_ABS_TOL = 5.0  # packs


@pytest.fixture(scope="module")
def prop99() -> pl.DataFrame:
    """Load the vendored Prop 99 panel, or skip if it is not present."""
    if not DATA_PATH.exists():
        pytest.skip(f"Prop 99 fixture not found at {DATA_PATH}; run test/data/vendor_prop99.py to create it")
    # The synthetic-control estimators pivot via polars; feed them polars (as the
    # other SC tests do). Year is read as an integer and cast to a string by the
    # estimator, so "1988"/"1989" comparisons sort correctly.
    return pl.read_csv(DATA_PATH).with_columns(pl.col("year").cast(pl.Int64))


def _fit(cls, prop99: pl.DataFrame):
    model = cls(
        prop99,
        geo_variable="state",
        test_geos=[TREATED],
        date_variable="year",
        pre_period=PRE_PERIOD,
        post_period=POST_PERIOD,
        y_variable="cigsale",
        alpha=0.1,
    )
    return model.pre_process().generate()


def _avg_gap(model) -> float:
    return float(np.mean(np.asarray(model.results["lift"], dtype=float).ravel()))


def _terminal_gap(model) -> float:
    return float(np.asarray(model.results["lift"], dtype=float).ravel()[-1])


class TestProp99Data:
    @staticmethod
    def test_panel_shape(prop99: pl.DataFrame) -> None:
        assert prop99["state"].n_unique() == 39  # California + 38 donors
        years = prop99["year"].unique().to_list()
        assert min(years) == 1970 and max(years) == 2000

    @staticmethod
    def test_california_landmark_values(prop99: pl.DataFrame) -> None:
        # Spot-check that the vendored data is the canonical Abadie panel.
        ca = prop99.filter(pl.col("state") == TREATED)
        by_year = dict(zip(ca["year"].to_list(), ca["cigsale"].to_list()))
        assert by_year[1970] == pytest.approx(123.0, abs=0.1)
        assert by_year[1988] == pytest.approx(90.1, abs=0.1)
        assert by_year[2000] == pytest.approx(41.6, abs=0.1)


class TestSyntheticControlParity:
    @pytest.fixture(scope="class")
    def fitted(self, prop99: pl.DataFrame) -> SyntheticControl:
        return _fit(SyntheticControl, prop99)

    @staticmethod
    def test_avg_gap_matches_published(fitted: SyntheticControl) -> None:
        assert _avg_gap(fitted) == pytest.approx(REF_AVG_GAP, abs=SC_AVG_ABS_TOL)

    @staticmethod
    def test_terminal_gap_matches_published(fitted: SyntheticControl) -> None:
        # The headline number: ~26 fewer packs per capita by 2000.
        assert _terminal_gap(fitted) == pytest.approx(REF_TERMINAL_GAP, abs=SC_TERMINAL_ABS_TOL)

    @staticmethod
    def test_effect_is_negative_and_significant(fitted: SyntheticControl) -> None:
        assert _avg_gap(fitted) < 0.0
        assert fitted.results["p_value"] <= 0.1


class TestSyntheticControlVParity:
    """SyntheticControlV (Abadie & Gardeazabal V-weighted method)."""

    @pytest.fixture(scope="class")
    def fitted(self, prop99: pl.DataFrame) -> SyntheticControlV:
        return _fit(SyntheticControlV, prop99)

    @staticmethod
    def test_avg_gap_matches_published(fitted: SyntheticControlV) -> None:
        # The V-weighted method on lagged outcomes recovers the same fit as the
        # plain estimator on this panel, so it matches the published gap closely.
        assert _avg_gap(fitted) == pytest.approx(REF_AVG_GAP, abs=SC_AVG_ABS_TOL)

    @staticmethod
    def test_terminal_gap_matches_published(fitted: SyntheticControlV) -> None:
        assert _terminal_gap(fitted) == pytest.approx(REF_TERMINAL_GAP, abs=SC_TERMINAL_ABS_TOL)

    @staticmethod
    def test_effect_is_negative_and_significant(fitted: SyntheticControlV) -> None:
        assert _avg_gap(fitted) < 0.0
        assert fitted.results["p_value"] <= 0.1


class TestAugmentedSyntheticControlParity:
    @pytest.fixture(scope="class")
    def fitted(self, prop99: pl.DataFrame) -> AugmentedSyntheticControl:
        return _fit(AugmentedSyntheticControl, prop99)

    @staticmethod
    def test_avg_gap_matches_published(fitted: AugmentedSyntheticControl) -> None:
        assert _avg_gap(fitted) == pytest.approx(REF_AVG_GAP, abs=GAP_ABS_TOL)

    @staticmethod
    def test_effect_is_negative_and_significant(fitted: AugmentedSyntheticControl) -> None:
        assert _avg_gap(fitted) < 0.0
        assert fitted.results["p_value"] <= 0.1


class TestPenalizedSyntheticControlParity:
    @pytest.fixture(scope="class")
    def fitted(self, prop99: pl.DataFrame) -> PenalizedSyntheticControl:
        return _fit(PenalizedSyntheticControl, prop99)

    @staticmethod
    def test_avg_gap_matches_published(fitted: PenalizedSyntheticControl) -> None:
        # Fixed in #31: matches the full pre-period trajectory with a scaled penalty.
        assert _avg_gap(fitted) == pytest.approx(REF_AVG_GAP, abs=GAP_ABS_TOL)

    @staticmethod
    def test_effect_is_negative_and_significant(fitted: PenalizedSyntheticControl) -> None:
        assert _avg_gap(fitted) < 0.0
        assert fitted.results["p_value"] <= 0.1


class TestGeneralizedSyntheticControlParity:
    @pytest.fixture(scope="class")
    def fitted(self, prop99: pl.DataFrame) -> GeneralizedSyntheticControl:
        return _fit(GeneralizedSyntheticControl, prop99)

    @staticmethod
    def test_avg_gap_matches_published(fitted: GeneralizedSyntheticControl) -> None:
        # Fixed in #32: eigenvalue-ratio factor selection no longer over-selects.
        assert _avg_gap(fitted) == pytest.approx(REF_AVG_GAP, abs=GAP_ABS_TOL)

    @staticmethod
    def test_effect_is_negative_and_significant(fitted: GeneralizedSyntheticControl) -> None:
        assert _avg_gap(fitted) < 0.0
        assert fitted.results["p_value"] <= 0.1


class TestRobustSyntheticControlParity:
    """RobustSyntheticControl (Amjad, Shah & Shen SVD-denoised ridge).

    Robust SC de-noises the donor matrix by hard singular-value truncation, so the
    retained rank is the key choice. Prop 99's donor spectrum is dominated by the
    level, and keeping the leading two components (the level plus the dominant
    dynamic factor) recovers the published effect; that is the documented Prop 99
    configuration used here. The out-of-the-box default (99.9% spectral-energy
    retention) keeps more components and attenuates the estimate, so the parity
    check pins the rank explicitly, as Abadie's own Prop 99 setup fixes its
    predictors.
    """

    @pytest.fixture(scope="class")
    def fitted(self, prop99: pl.DataFrame) -> RobustSyntheticControl:
        model = RobustSyntheticControl(
            prop99,
            geo_variable="state",
            test_geos=[TREATED],
            date_variable="year",
            pre_period=PRE_PERIOD,
            post_period=POST_PERIOD,
            y_variable="cigsale",
            alpha=0.1,
            sv_count=2,
        )
        return model.pre_process().generate()

    @staticmethod
    def test_avg_gap_matches_published(fitted: RobustSyntheticControl) -> None:
        assert _avg_gap(fitted) == pytest.approx(REF_AVG_GAP, abs=GAP_ABS_TOL)

    @staticmethod
    def test_terminal_gap_matches_published(fitted: RobustSyntheticControl) -> None:
        assert _terminal_gap(fitted) == pytest.approx(REF_TERMINAL_GAP, abs=SC_TERMINAL_ABS_TOL)

    @staticmethod
    def test_effect_is_negative_and_significant(fitted: RobustSyntheticControl) -> None:
        assert _avg_gap(fitted) < 0.0
        assert fitted.results["p_value"] <= 0.1

    @staticmethod
    def test_runs_with_default_rank(prop99: pl.DataFrame) -> None:
        # Out-of-the-box (no threshold / sv_count): the spectral-energy default
        # makes it runnable and still finds a negative, significant effect.
        model = RobustSyntheticControl(
            prop99,
            geo_variable="state",
            test_geos=[TREATED],
            date_variable="year",
            pre_period=PRE_PERIOD,
            post_period=POST_PERIOD,
            y_variable="cigsale",
            alpha=0.1,
        )
        model.pre_process().generate()
        assert _avg_gap(model) < 0.0
        assert model.results["p_value"] <= 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
