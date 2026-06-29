"""Parity test: our GeoLift vs Meta GeoLift's published example results.

Validates GeoCausality's ``GeoLift`` against the numbers Meta documents for its
bundled ``GeoLift_Test`` dataset — a 15-day campaign in Chicago + Portland over
90 pre-period days — reported in the GeoLift walkthrough:

    percent lift 5.5%, incremental Y 4,704, ATT 156.8/day, p approximately 0.01
    (https://facebookincubator.github.io/GeoLift/docs/GettingStarted/Walkthrough/)

Our ``GeoLift`` is a reimplementation (our ``AugmentedSyntheticControl`` for the
point estimate, our Generalized Synthetic Control parametric bootstrap for
inference), not a port of the R ``augsynth`` / ``gsynth`` packages. So the point
estimate is checked **within tolerance** and significance only **qualitatively**
(GeoLift's published p-value comes from its conformal path, ours from the
bootstrap). The tolerances below are intentionally generous: this guards against
gross divergence, not exact equality.

The ``GeoLift_Test`` data ships as an ``.rda`` in the R package. Export it once:

    write.csv(GeoLift::GeoLift_Test, "geolift_test.csv", row.names = FALSE)

and drop it at ``test/data/geolift_test.csv``. Until then this test skips.
"""

from pathlib import Path

import polars as pl
import pytest

from GeoCausality.geolift import GeoLift

DATA_PATH = Path(__file__).parent / "data" / "geolift_test.csv"
TEST_GEOS = ["chicago", "portland"]
N_PRE_DAYS = 90

# Meta's published augmented-SC results on GeoLift_Test.
REF_PERCENT_LIFT = 5.5
REF_INCREMENTAL = 4704

# Tolerances (reimplementation, not a port): catch gross divergence only.
PERCENT_LIFT_ABS_TOL = 2.0  # percentage points
INCREMENTAL_REL_TOL = 0.20  # 20%

# xfail until #20 is fixed: AugmentedSyntheticControl is level-biased for
# aggregated treated units (its simplex-constrained counterfactual cannot reach
# the summed Chicago + Portland level), inflating the point estimate to ~33,100 /
# ~57% vs GeoLift's 4,704 / 5.5%. strict=True so the test flips to a failure
# (prompting us to promote it to a hard assertion) once ASC is fixed.
ASC_BIAS_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="#20: AugmentedSyntheticControl level-biased for aggregated treated units",
)


@pytest.fixture(scope="module")
def geolift_test() -> pl.DataFrame:
    """Load the vendored GeoLift_Test dataset, or skip if it is not present."""
    if not DATA_PATH.exists():
        pytest.skip(
            f"GeoLift_Test fixture not found at {DATA_PATH}; export it from R "
            "(write.csv(GeoLift::GeoLift_Test, ...)) to enable the parity test"
        )
    return pl.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def fitted(geolift_test: pl.DataFrame) -> GeoLift:
    """Run our GeoLift on GeoLift_Test with the walkthrough's split and markets."""
    df = geolift_test.with_columns(pl.col("location").str.to_lowercase())
    dates = sorted(df["date"].cast(pl.String).unique().to_list())
    model = GeoLift(
        df,
        geo_variable="location",
        test_geos=TEST_GEOS,
        date_variable="date",
        pre_period=dates[N_PRE_DAYS - 1],
        post_period=dates[N_PRE_DAYS],
        y_variable="Y",
    )
    return model.pre_process().generate()


class TestGeoLiftParity:
    @staticmethod
    def test_markets_present(geolift_test: pl.DataFrame) -> None:
        locations = {loc.lower() for loc in geolift_test["location"].unique().to_list()}
        assert set(TEST_GEOS) <= locations, f"expected {TEST_GEOS} in the dataset locations"

    @staticmethod
    @ASC_BIAS_XFAIL
    def test_percent_lift_matches_published(fitted: GeoLift) -> None:
        results = fitted.results
        baseline = float(results["counterfactual"]["Y"].sum())
        percent_lift = 100.0 * results["incrementality"] / baseline
        assert percent_lift == pytest.approx(REF_PERCENT_LIFT, abs=PERCENT_LIFT_ABS_TOL)

    @staticmethod
    @ASC_BIAS_XFAIL
    def test_incremental_matches_published(fitted: GeoLift) -> None:
        assert fitted.results["incrementality"] == pytest.approx(REF_INCREMENTAL, rel=INCREMENTAL_REL_TOL)

    @staticmethod
    def test_effect_is_significant(fitted: GeoLift) -> None:
        # GeoLift reports p approximately 0.01; we only require clear significance.
        results = fitted.results
        assert results["p_value"] <= 0.1
        assert results["incrementality_ci_lower"] > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
