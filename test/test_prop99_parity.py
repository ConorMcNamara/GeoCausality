"""Parity test: our synthetic-control family vs Abadie et al. (2010) Prop 99.

Validates GeoCausality's synthetic-control family against the canonical benchmark
of the literature: California's Proposition 99 tobacco-control program. California
passed Prop 99 in 1988, so it is treated from 1989; the other 38 states form the
donor pool. Abadie, Diamond & Hainmueller (2010) estimate that by 2000
California's per-capita cigarette sales were about **26 packs** below its
synthetic control, with an **average post-period (1989-2000) gap of roughly -19
to -20 packs**.

The vendored ``prop99_smoking.csv`` is the ``smoking`` dataset from the R
``Synth`` package (39 states x 1970-2000); see ``prop99_smoking.README.txt`` for
provenance and ``vendor_prop99.py`` for the download. Until that CSV is present
this test skips, exactly like the GeoLift / Card-Krueger parity tests.

Each estimator gets a single parity assertion -- average post-period gap within
band *and* negative *and* significant -- plus any estimator-specific check
(factor-count selection, affine weights, default-rank robustness). The plain
``SyntheticControl`` additionally pins the headline year-2000 terminal gap.
Tolerances are deliberately generous: this guards against gross divergence, not
exact equality, since our predictor specification and solvers differ from
Abadie's exact Prop 99 setup. This parity test has surfaced genuine
reimplementation bugs (PenalizedSyntheticControl #31, GeneralizedSyntheticControl
#32), exactly as the GeoLift parity test surfaced #20.
"""

import numpy as np
import polars as pl
import pytest

from GeoCausality.augmented_synthetic_control import AugmentedSyntheticControl
from GeoCausality.elastic_net_synthetic_control import ElasticNetSyntheticControl
from GeoCausality.generalized_synthetic_control import GeneralizedSyntheticControl
from GeoCausality.interactive_fixed_effects import InteractiveFixedEffects
from GeoCausality.kernel_synthetic_control import KernelSyntheticControl
from GeoCausality.matrix_completion import MatrixCompletion
from GeoCausality.nonlinear_synthetic_control import NonlinearSyntheticControl
from GeoCausality.penalized_synthetic_control import PenalizedSyntheticControl
from GeoCausality.robust_synthetic_control import RobustSyntheticControl
from GeoCausality.synthetic_control import SyntheticControl, SyntheticControlV
from GeoCausality.synthetic_diff_in_diff import SyntheticDiffInDiff

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


def _assert_parity(model, tol: float = GAP_ABS_TOL) -> None:
    """The shared per-estimator parity bar: within band, negative, significant."""
    assert _avg_gap(model) == pytest.approx(REF_AVG_GAP, abs=tol)
    assert _avg_gap(model) < 0.0
    assert model.results["p_value"] <= 0.1


def test_prop99_landmark_values(prop99: pl.DataFrame) -> None:
    # Spot-check that the vendored data is the canonical Abadie panel, so a parity
    # failure below points at the estimator, not at corrupted data.
    assert prop99["state"].n_unique() == 39  # California + 38 donors
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
    def test_matches_published(fitted: SyntheticControl) -> None:
        # The plain estimator tracks Abadie almost exactly -> tight band.
        _assert_parity(fitted, tol=SC_AVG_ABS_TOL)

    @staticmethod
    def test_terminal_gap_matches_published(fitted: SyntheticControl) -> None:
        # The headline number: ~26 fewer packs per capita by 2000.
        assert _terminal_gap(fitted) == pytest.approx(REF_TERMINAL_GAP, abs=SC_TERMINAL_ABS_TOL)


class TestSyntheticControlVParity:
    """SyntheticControlV (Abadie & Gardeazabal V-weighted method).

    The V-weighted method on lagged outcomes recovers the same fit as the plain
    estimator on this panel, so it matches the published gap closely.
    """

    @pytest.fixture(scope="class")
    def fitted(self, prop99: pl.DataFrame) -> SyntheticControlV:
        return _fit(SyntheticControlV, prop99)

    @staticmethod
    def test_matches_published(fitted: SyntheticControlV) -> None:
        _assert_parity(fitted, tol=SC_AVG_ABS_TOL)


class TestAugmentedSyntheticControlParity:
    @pytest.fixture(scope="class")
    def fitted(self, prop99: pl.DataFrame) -> AugmentedSyntheticControl:
        return _fit(AugmentedSyntheticControl, prop99)

    @staticmethod
    def test_matches_published(fitted: AugmentedSyntheticControl) -> None:
        _assert_parity(fitted)


class TestElasticNetSyntheticControlParity:
    """ElasticNetSyntheticControl (2016 synthesis: elastic-net / intercept-shifted SC).

    The unconstrained elastic-net regime with an intercept lands in the published
    Prop 99 band (average gap ~-15, mildly attenuated by the elastic-net
    shrinkage), negative and significant.
    """

    @pytest.fixture(scope="class")
    def fitted(self, prop99: pl.DataFrame) -> ElasticNetSyntheticControl:
        return _fit(ElasticNetSyntheticControl, prop99)

    @staticmethod
    def test_matches_published(fitted: ElasticNetSyntheticControl) -> None:
        _assert_parity(fitted)


class TestPenalizedSyntheticControlParity:
    @pytest.fixture(scope="class")
    def fitted(self, prop99: pl.DataFrame) -> PenalizedSyntheticControl:
        return _fit(PenalizedSyntheticControl, prop99)

    @staticmethod
    def test_matches_published(fitted: PenalizedSyntheticControl) -> None:
        # Fixed in #31: matches the full pre-period trajectory with a scaled penalty.
        _assert_parity(fitted)


class TestGeneralizedSyntheticControlParity:
    @pytest.fixture(scope="class")
    def fitted(self, prop99: pl.DataFrame) -> GeneralizedSyntheticControl:
        return _fit(GeneralizedSyntheticControl, prop99)

    @staticmethod
    def test_matches_published(fitted: GeneralizedSyntheticControl) -> None:
        # Fixed in #32: eigenvalue-ratio factor selection no longer over-selects.
        _assert_parity(fitted)


class TestSyntheticDiffInDiffParity:
    """SyntheticDiffInDiff (Arkhangelsky et al. 2021 doubly-weighted DID).

    The canonical SDID Prop 99 estimate is about -15 packs per capita on average --
    mildly attenuated relative to plain synthetic control because the unit fixed
    effect and L2 penalty trade a little bias for stability. Inference is the
    placebo variance over the 38 donor states, which is deterministic and finds the
    effect significant on this pool.
    """

    @pytest.fixture(scope="class")
    def fitted(self, prop99: pl.DataFrame) -> SyntheticDiffInDiff:
        return _fit(SyntheticDiffInDiff, prop99)

    @staticmethod
    def test_matches_published(fitted: SyntheticDiffInDiff) -> None:
        _assert_parity(fitted)


class TestInteractiveFixedEffectsParity:
    """InteractiveFixedEffects (Bai 2009 additive two-way FE + interactive factors).

    Estimates the additive unit/time fixed effects and the latent factors from the
    38 donor states, then projects California's pre-period onto that time structure.
    The eigenvalue-ratio criterion selects a single factor on this panel -- the
    configuration that recovers the effect; forcing more factors over-fits the
    pre-period and washes it out.
    """

    @pytest.fixture(scope="class")
    def fitted(self, prop99: pl.DataFrame) -> InteractiveFixedEffects:
        return _fit(InteractiveFixedEffects, prop99)

    @staticmethod
    def test_factor_count_selected(fitted: InteractiveFixedEffects) -> None:
        # Eigenvalue-ratio picks a parsimonious factor count on the Prop 99 donors.
        assert fitted.n_factors_selected == 1

    @staticmethod
    def test_matches_published(fitted: InteractiveFixedEffects) -> None:
        _assert_parity(fitted)


class TestMatrixCompletionParity:
    """MatrixCompletion (Athey et al. 2021 MC-NNM nuclear-norm completion).

    Stacks the 39 states into one panel, masks California's post-1988 cells, and
    completes the matrix with two-way fixed effects plus a cross-validated
    low-rank term. On the well-populated Prop 99 donor pool it recovers the
    published gap closely -- average ~-20 packs, year-2000 ~-30 -- with no rank
    or penalty tuning (unlike RobustSyntheticControl, which pins its rank here).
    """

    @pytest.fixture(scope="class")
    def fitted(self, prop99: pl.DataFrame) -> MatrixCompletion:
        return _fit(MatrixCompletion, prop99)

    @staticmethod
    def test_matches_published(fitted: MatrixCompletion) -> None:
        _assert_parity(fitted)


class TestRobustSyntheticControlParity:
    """RobustSyntheticControl (Amjad, Shah & Shen SVD-denoised ridge).

    Robust SC de-noises the donor matrix by hard singular-value truncation, so the
    retained rank is the key choice. Keeping the leading two components (the level
    plus the dominant dynamic factor) recovers the published effect; that is the
    documented Prop 99 configuration pinned here, since the default spectral-energy
    retention keeps more components and attenuates the estimate.
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
    def test_matches_published(fitted: RobustSyntheticControl) -> None:
        _assert_parity(fitted)

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


class TestNonlinearSyntheticControlParity:
    """NonlinearSyntheticControl (Tian 2023 NSC).

    Tian (2023) reports an ATT of about -19.1; our rolling-origin cross-validation
    lands at ~-20.2. Being linear-in-weights it tracks the trend and reproduces the
    effect, unlike the kernel-map estimator below.
    """

    @pytest.fixture(scope="class")
    def fitted(self, prop99: pl.DataFrame) -> NonlinearSyntheticControl:
        return _fit(NonlinearSyntheticControl, prop99)

    @staticmethod
    def test_matches_published(fitted: NonlinearSyntheticControl) -> None:
        _assert_parity(fitted)

    @staticmethod
    def test_weights_are_affine(fitted: NonlinearSyntheticControl) -> None:
        # Signed weights that sum to one (Tian drops the non-negativity constraint).
        assert float(np.sum(fitted.model)) == pytest.approx(1.0, abs=1e-3)
        assert float(np.min(fitted.model)) < 0.0


class TestKernelSyntheticControlParity:
    """KernelSyntheticControl (kernel-ridge nonlinear-map SC).

    A nonlinear map of the donor outcomes rather than a level-matching combination,
    so it is not expected to reproduce the exact published magnitude on this
    trending panel (it attenuates toward the pre-period level); the composite
    linear+RBF kernel keeps the effect negative and significant, which is asserted.
    """

    @pytest.fixture(scope="class")
    def fitted(self, prop99: pl.DataFrame) -> KernelSyntheticControl:
        return _fit(KernelSyntheticControl, prop99)

    @staticmethod
    def test_effect_is_negative_and_significant(fitted: KernelSyntheticControl) -> None:
        assert _avg_gap(fitted) < 0.0
        assert fitted.results["p_value"] <= 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
