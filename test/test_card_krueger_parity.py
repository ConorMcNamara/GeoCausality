"""Parity test: our DiffinDiff / FixedEffects vs Card & Krueger (1994).

Validates GeoCausality's two econometric estimators against the most famous
result in the difference-in-differences literature: Card & Krueger's New
Jersey/Pennsylvania minimum-wage natural experiment. NJ raised its minimum wage
on 1 April 1992; PA did not. Comparing full-time-equivalent (FTE) fast-food
employment before (Feb/Mar 1992) and after (Nov/Dec 1992) in each state gives
(Card & Krueger 1994, Table 3):

    NJ: 20.44 -> 21.03   PA: 23.33 -> 21.17
    DiD = (21.03 - 20.44) - (21.17 - 23.33) = +2.76 FTE

i.e. employment *rose* in NJ relative to PA -- the celebrated finding that a
modest minimum-wage increase did not reduce employment.

The vendored ``card_krueger_1994.csv`` is the authors' public ``public.dat``
(410 restaurants, two survey waves) reshaped to our ``geo / date / y`` long
format; see ``card_krueger_1994.README.txt`` for provenance and
``vendor_card_krueger.py`` for the reshape. Until that CSV is present this test
skips, exactly like the GeoLift parity test.

Two estimators, two natural framings of the same experiment:

* ``FixedEffects`` is a two-way fixed-effects panel model, so it runs directly
  on the full **store-level** panel (one entity per restaurant). Its
  ``campaign_treatment`` coefficient is the TWFE DiD and it carries real
  clustered standard errors -- the strong "golden master" check here.

* ``DiffinDiff`` sums ``y`` across geos within each (treatment, period) cell
  before regressing, which is the right 2x2 textbook DiD only when fed
  comparable group series. Card & Krueger's design is heavily unbalanced
  (~320 NJ stores vs ~77 PA stores), so summing raw store-level counts is
  dominated by store *counts*, not the per-store effect (see
  ``test_diff_in_diff_store_sums_diverge``). Fed the per-state **mean** FTE
  series (two geos), it recovers the published per-restaurant DiD exactly --
  though a saturated 2x2 has zero residual degrees of freedom, so only the point
  estimate is meaningful (its p-value/CI are degenerate, hence not asserted).

Tolerances are deliberately generous: this guards against gross divergence, not
exact equality. Our TWFE on the unbalanced panel weights observations slightly
differently from the simple balanced-sample DiD, so "right sign and magnitude"
is the bar.
"""

import warnings
from pathlib import Path

import pandas as pd
import pytest

from GeoCausality.diff_in_diff import DiffinDiff
from GeoCausality.fixed_effects import FixedEffects

DATA_PATH = Path(__file__).parent / "data" / "card_krueger_1994.csv"

WAVE1 = "1992-02-01"  # before the NJ minimum-wage increase
WAVE2 = "1992-11-01"  # after

# Published difference-in-differences (Card & Krueger 1994, Table 3, FTE).
REF_DID = 2.76
# Published per-state cell means (before, after).
REF_CELL_MEANS = {("NJ", WAVE1): 20.44, ("NJ", WAVE2): 21.03, ("PA", WAVE1): 23.33, ("PA", WAVE2): 21.17}

# Reimplementation tolerance (not a port): confirm sign + magnitude, not equality.
DID_ABS_TOL = 0.5  # FTE
CELL_MEAN_ABS_TOL = 0.05  # the vendored data should reproduce the table closely


@pytest.fixture(scope="module")
def card_krueger() -> pd.DataFrame:
    """Load the vendored Card & Krueger panel, or skip if it is not present."""
    if not DATA_PATH.exists():
        pytest.skip(
            f"Card & Krueger fixture not found at {DATA_PATH}; run test/data/vendor_card_krueger.py to create it"
        )
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


def _nj_geos(df: pd.DataFrame) -> list[str]:
    return df.loc[df["state"] == "NJ", "geo"].unique().tolist()


def _state_means(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the store panel to one mean-FTE series per state (two geos)."""
    return df.groupby(["state", "date"], as_index=False)["y"].mean().rename(columns={"state": "geo"})


class TestCardKruegerData:
    @staticmethod
    def test_states_and_waves_present(card_krueger: pd.DataFrame) -> None:
        assert set(card_krueger["state"].unique()) == {"NJ", "PA"}
        assert set(card_krueger["date"].dt.strftime("%Y-%m-%d")) == {WAVE1, WAVE2}

    @staticmethod
    def test_cell_means_match_published(card_krueger: pd.DataFrame) -> None:
        # Sanity-check that the vendored microdata reproduces Table 3, so a parity
        # failure below points at the estimator, not at corrupted data.
        means = card_krueger.copy()
        means["date"] = means["date"].dt.strftime("%Y-%m-%d")
        cell = means.groupby(["state", "date"])["y"].mean()
        for (state, date), expected in REF_CELL_MEANS.items():
            assert cell[(state, date)] == pytest.approx(expected, abs=CELL_MEAN_ABS_TOL)


class TestFixedEffectsParity:
    """FixedEffects (TWFE) on the full store-level panel."""

    @pytest.fixture(scope="class")
    def fitted(self, card_krueger: pd.DataFrame) -> FixedEffects:
        model = FixedEffects(
            card_krueger,
            geo_variable="geo",
            test_geos=_nj_geos(card_krueger),
            date_variable="date",
            pre_period=WAVE1,
            post_period=WAVE2,
            y_variable="y",
            alpha=0.05,
        )
        return model.pre_process().generate()

    @staticmethod
    def test_did_matches_published(fitted: FixedEffects) -> None:
        assert fitted.results["lift"] == pytest.approx(REF_DID, abs=DID_ABS_TOL)

    @staticmethod
    def test_effect_is_positive(fitted: FixedEffects) -> None:
        # The headline finding: employment rose in NJ relative to PA.
        assert fitted.results["lift"] > 0.0

    @staticmethod
    def test_confidence_interval_brackets_estimate(fitted: FixedEffects) -> None:
        results = fitted.results
        assert results["lift_ci_lower"] <= results["lift"] <= results["lift_ci_upper"]


class TestDiffinDiffParity:
    """DiffinDiff on the per-state mean-FTE series (the 2x2 textbook DiD)."""

    @pytest.fixture(scope="class")
    def fitted(self, card_krueger: pd.DataFrame) -> DiffinDiff:
        means = _state_means(card_krueger)
        model = DiffinDiff(
            means,
            geo_variable="geo",
            test_geos=["NJ"],
            date_variable="date",
            pre_period=WAVE1,
            post_period=WAVE2,
            y_variable="y",
            alpha=0.05,
        )
        # A saturated 2x2 has zero residual df, so statsmodels divides by zero
        # computing the (here meaningless) variance -- expected, and silenced.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return model.pre_process().generate()

    @staticmethod
    def test_did_matches_published(fitted: DiffinDiff) -> None:
        # A saturated 2x2 recovers the published per-restaurant DiD exactly; only
        # the point estimate is meaningful (no residual df for inference).
        assert fitted.results["lift"] == pytest.approx(REF_DID, abs=DID_ABS_TOL)

    @staticmethod
    def test_effect_is_positive(fitted: DiffinDiff) -> None:
        assert fitted.results["lift"] > 0.0

    @staticmethod
    def test_diff_in_diff_store_sums_diverge(card_krueger: pd.DataFrame) -> None:
        # Documents why we feed DiffinDiff the per-state *mean* series above:
        # summing raw store-level FTE across the unbalanced NJ/PA groups is
        # dominated by store counts, so it cannot recover the +2.76 per-store DiD.
        model = DiffinDiff(
            card_krueger,
            geo_variable="geo",
            test_geos=_nj_geos(card_krueger),
            date_variable="date",
            pre_period=WAVE1,
            post_period=WAVE2,
            y_variable="y",
            alpha=0.05,
        )
        with warnings.catch_warnings():  # saturated 2x2, as above
            warnings.simplefilter("ignore", RuntimeWarning)
            model.pre_process().generate()
        assert abs(model.results["lift"]) > 10 * REF_DID  # nowhere near +2.76


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
