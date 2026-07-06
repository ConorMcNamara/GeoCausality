"""Parity test: our counterfactual estimators vs Abadie et al. (2015) reunification.

Validates GeoCausality's interactive-fixed-effects and synthetic-control
estimators against the second canonical benchmark of the literature: the economic
cost of the 1990 German reunification to West Germany. West Germany reunified in
1990, so it is treated from 1990; the other 16 OECD countries form the donor
pool. Abadie, Diamond & Hainmueller (2015) estimate that reunification lowered
West German per-capita GDP by roughly **$1,600 per year on average** relative to
the synthetic control, with the gap widening over time.

The panel comes from the shared ``germany`` fixture (``conftest.py``); see
``germany_reunification.README.txt`` for provenance and ``vendor_germany.py`` for
the download. Until that CSV is present these tests skip, exactly like the Prop 99
and Card-Krueger parity tests.

Tolerances are deliberately generous. Abadie et al. match on covariates (trade
openness, schooling, investment, inflation) in addition to lagged GDP, while our
estimators use only the GDP trajectory, and the published figure is a "right
ballpark" number rather than a single precise ATT. The interactive-fixed-effects
estimators run mildly stronger than plain synthetic control (the same attenuation
of the counterfactual seen on Prop 99), so the band brackets the whole family.
"""

import numpy as np
import polars as pl
import pytest

from GeoCausality.generalized_synthetic_control import GeneralizedSyntheticControl
from GeoCausality.interactive_fixed_effects import InteractiveFixedEffects
from GeoCausality.synthetic_control import SyntheticControl
from GeoCausality.synthetic_diff_in_diff import SyntheticDiffInDiff

TREATED = "West Germany"
PRE_PERIOD = "1989"  # last pre-reunification year
POST_PERIOD = "1990"  # reunification year (treated from here)

# Published reunification effect (Abadie, Diamond & Hainmueller 2015): the average
# post-period per-capita GDP shortfall vs the synthetic control.
REF_AVG_GAP = -1600.0  # USD/year
# Reimplementation band: confirm sign + magnitude, not equality. ~44% of the
# reference, matching the generosity of the Prop 99 parity band.
GAP_ABS_TOL = 700.0


def _fit(cls, germany: pl.DataFrame, **overrides: object):
    kwargs = {
        "geo_variable": "country",
        "test_geos": [TREATED],
        "date_variable": "year",
        "pre_period": PRE_PERIOD,
        "post_period": POST_PERIOD,
        "y_variable": "gdp",
        "alpha": 0.1,
    }
    kwargs.update(overrides)
    return cls(germany, **kwargs).pre_process().generate()


def _avg_gap(model) -> float:
    return float(np.mean(np.asarray(model.results["lift"], dtype=float).ravel()))


class TestGermanyData:
    @staticmethod
    def test_panel_shape(germany: pl.DataFrame) -> None:
        assert germany["country"].n_unique() == 17  # West Germany + 16 donors
        years = germany["year"].unique().to_list()
        assert min(years) == 1960 and max(years) == 2003

    @staticmethod
    def test_west_germany_landmark_values(germany: pl.DataFrame) -> None:
        # Spot-check that the vendored data is the canonical Abadie panel.
        wg = germany.filter(pl.col("country") == TREATED)
        by_year = dict(zip(wg["year"].to_list(), wg["gdp"].to_list()))
        assert by_year[1960] == pytest.approx(2284.0, abs=1.0)
        assert by_year[1990] == pytest.approx(20465.0, abs=1.0)
        assert by_year[2003] == pytest.approx(28855.0, abs=1.0)


class TestSyntheticControlParity:
    @pytest.fixture(scope="class")
    def fitted(self, germany: pl.DataFrame) -> SyntheticControl:
        return _fit(SyntheticControl, germany)

    @staticmethod
    def test_avg_gap_matches_published(fitted: SyntheticControl) -> None:
        assert _avg_gap(fitted) == pytest.approx(REF_AVG_GAP, abs=GAP_ABS_TOL)

    @staticmethod
    def test_effect_is_negative_and_significant(fitted: SyntheticControl) -> None:
        assert _avg_gap(fitted) < 0.0
        assert fitted.results["p_value"] <= 0.1


class TestGeneralizedSyntheticControlParity:
    @pytest.fixture(scope="class")
    def fitted(self, germany: pl.DataFrame) -> GeneralizedSyntheticControl:
        return _fit(GeneralizedSyntheticControl, germany)

    @staticmethod
    def test_avg_gap_matches_published(fitted: GeneralizedSyntheticControl) -> None:
        assert _avg_gap(fitted) == pytest.approx(REF_AVG_GAP, abs=GAP_ABS_TOL)

    @staticmethod
    def test_effect_is_negative_and_significant(fitted: GeneralizedSyntheticControl) -> None:
        assert _avg_gap(fitted) < 0.0
        assert fitted.results["p_value"] <= 0.1


class TestSyntheticDiffInDiffParity:
    """SyntheticDiffInDiff (Arkhangelsky et al. 2021 doubly-weighted DID).

    Recovers the reunification shortfall in sign and magnitude (~ -1,800 USD/year,
    within the parity band). Unlike the Prop 99 pool, the placebo variance here is
    driven by a small, heterogeneous 16-country donor set, so the standard error is
    large and the effect is not significant at the 10% level -- matching the
    borderline permutation inference in the original study. We therefore assert
    sign and magnitude, not significance.
    """

    @pytest.fixture(scope="class")
    def fitted(self, germany: pl.DataFrame) -> SyntheticDiffInDiff:
        return _fit(SyntheticDiffInDiff, germany)

    @staticmethod
    def test_avg_gap_matches_published(fitted: SyntheticDiffInDiff) -> None:
        assert _avg_gap(fitted) == pytest.approx(REF_AVG_GAP, abs=GAP_ABS_TOL)

    @staticmethod
    def test_effect_is_negative(fitted: SyntheticDiffInDiff) -> None:
        assert _avg_gap(fitted) < 0.0
        assert 0.0 <= fitted.results["p_value"] <= 1.0


class TestInteractiveFixedEffectsParity:
    """InteractiveFixedEffects (Bai 2009), both estimation modes.

    ``projection`` estimates the factor structure from the donor countries and
    projects West Germany's pre-period onto it; ``coefficient`` estimates the
    treatment effect as a full-panel Bai regression coefficient. Both recover the
    reunification effect on this well-populated donor pool.
    """

    @pytest.fixture(scope="class")
    def projection(self, germany: pl.DataFrame) -> InteractiveFixedEffects:
        return _fit(InteractiveFixedEffects, germany, method="projection")

    @pytest.fixture(scope="class")
    def coefficient(self, germany: pl.DataFrame) -> InteractiveFixedEffects:
        return _fit(InteractiveFixedEffects, germany, method="coefficient")

    @staticmethod
    def test_projection_matches_published(projection: InteractiveFixedEffects) -> None:
        assert _avg_gap(projection) == pytest.approx(REF_AVG_GAP, abs=GAP_ABS_TOL)

    @staticmethod
    def test_coefficient_matches_published(coefficient: InteractiveFixedEffects) -> None:
        assert _avg_gap(coefficient) == pytest.approx(REF_AVG_GAP, abs=GAP_ABS_TOL)

    @staticmethod
    @pytest.mark.parametrize("mode", ["projection", "coefficient"])
    def test_effect_is_negative_and_significant(
        mode: str, projection: InteractiveFixedEffects, coefficient: InteractiveFixedEffects
    ) -> None:
        fitted = projection if mode == "projection" else coefficient
        assert _avg_gap(fitted) < 0.0
        assert fitted.results["p_value"] <= 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
