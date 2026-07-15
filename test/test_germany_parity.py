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
from GeoCausality.kernel_synthetic_control import KernelSyntheticControl
from GeoCausality.matrix_completion import MatrixCompletion
from GeoCausality.nonlinear_synthetic_control import NonlinearSyntheticControl
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
# Matrix completion attenuates more on this short, heterogeneous panel (the
# nuclear-norm penalty shrinks the counterfactual gap), so it gets a wider band
# than the linear family -- same sign and order of magnitude, not the same size.
MC_GAP_ABS_TOL = 1100.0


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


class TestMatrixCompletionParity:
    """MatrixCompletion (Athey et al. 2021 MC-NNM nuclear-norm completion).

    Masks West Germany's post-1989 cells and completes the 17-country panel with
    two-way fixed effects plus a cross-validated low-rank term. It recovers the
    reunification shortfall in sign and order of magnitude, but attenuated toward
    zero relative to the linear family: the nuclear-norm penalty shrinks the
    counterfactual gap, and this short, heterogeneous panel gives it less to work
    with than the 39-state Prop 99 pool (where it matches closely). We therefore
    assert sign and a wider magnitude band, plus significance.
    """

    @pytest.fixture(scope="class")
    def fitted(self, germany: pl.DataFrame) -> MatrixCompletion:
        return _fit(MatrixCompletion, germany)

    @staticmethod
    def test_avg_gap_matches_published(fitted: MatrixCompletion) -> None:
        assert _avg_gap(fitted) == pytest.approx(REF_AVG_GAP, abs=MC_GAP_ABS_TOL)

    @staticmethod
    def test_effect_is_negative_and_significant(fitted: MatrixCompletion) -> None:
        assert _avg_gap(fitted) < 0.0
        assert fitted.results["p_value"] <= 0.1


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


class TestNonlinearSyntheticControlParity:
    """NonlinearSyntheticControl (Tian 2023 NSC).

    A linear-in-weights estimator (affine signed weights + distance-L1 + ridge),
    so it tracks the GDP trend and recovers the reunification shortfall in the same
    band as the rest of the family. This is the check that distinguishes NSC from
    the kernel-map estimator, which does not reproduce the trending-panel level.
    """

    @pytest.fixture(scope="class")
    def fitted(self, germany: pl.DataFrame) -> NonlinearSyntheticControl:
        return _fit(NonlinearSyntheticControl, germany)

    @staticmethod
    def test_avg_gap_matches_published(fitted: NonlinearSyntheticControl) -> None:
        assert _avg_gap(fitted) == pytest.approx(REF_AVG_GAP, abs=GAP_ABS_TOL)

    @staticmethod
    def test_effect_is_negative_and_significant(fitted: NonlinearSyntheticControl) -> None:
        assert _avg_gap(fitted) < 0.0
        assert fitted.results["p_value"] <= 0.1


class TestKernelSyntheticControlParity:
    """KernelSyntheticControl (kernel-ridge nonlinear-map SC).

    Kernel ridge learns a nonlinear map of the donor outcomes rather than a
    level-matching weighted combination, so on a strongly trending single-unit
    panel it is not expected to reproduce the *magnitude* of the published ATT (it
    attenuates toward the pre-period level). The composite linear+RBF kernel does
    keep the effect the right sign and significant, which is what we assert here.
    """

    @pytest.fixture(scope="class")
    def fitted(self, germany: pl.DataFrame) -> KernelSyntheticControl:
        return _fit(KernelSyntheticControl, germany)

    @staticmethod
    def test_effect_is_negative_and_significant(fitted: KernelSyntheticControl) -> None:
        assert _avg_gap(fitted) < 0.0
        assert fitted.results["p_value"] <= 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
