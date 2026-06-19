"""Shared fixtures for the econometric estimator tests (Diff-in-Diff, FixedEffects).

Both estimators fit real regression models (statsmodels / linearmodels) and call
``.to_pandas()`` internally, so the fixtures hand them **pandas** input (which,
unlike polars, needs no pyarrow) with a ``datetime`` date column — exactly the
path ``EconometricEstimator.pre_process`` expects.

The generated panel gives every geo a *shared* time trend and equal numbers of
test and control geos. That keeps the geo-summed series parallel pre-period, so
Diff-in-Diff (which sums ``y`` across geos before regressing) can recover the
treatment effect, while FixedEffects (which works on the full geo-level panel)
recovers the per-geo effect directly.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

PRE_PERIOD = "2021-01-30"
POST_PERIOD = "2021-01-31"
N_DATES = 40
EFFECT = 8.0


@dataclass
class Panel:
    """A synthetic geo panel plus the ground-truth parameters used to build it."""

    df: pd.DataFrame
    effect: float
    n_test: int
    n_control: int
    n_post_dates: int
    pre_period: str = PRE_PERIOD
    post_period: str = POST_PERIOD
    extra: dict = field(default_factory=dict)

    def kwargs(self, **overrides: object) -> dict:
        """Constructor kwargs shared by every EconometricEstimator subclass."""
        base = {
            "geo_variable": "geo",
            "treatment_variable": "is_treatment",
            "date_variable": "date",
            "pre_period": self.pre_period,
            "post_period": self.post_period,
            "y_variable": "y",
            "msrp": 7.0,
            "spend": 10_000.0,
        }
        base.update(overrides)
        return base


def make_panel(
    effect: float,
    n_test: int = 3,
    n_control: int = 3,
    n_dates: int = N_DATES,
    seed: int = 0,
    noise: float = 1.0,
) -> Panel:
    """Builds a geo panel with parallel trends and a known post-period effect.

    Parameters
    ----------
    effect : float
        Additive per-geo, per-day treatment effect applied to the test geos in
        the post-period.
    n_test, n_control : int
        Number of treated and control geos. Equal counts keep the geo-summed
        series parallel, which Diff-in-Diff relies on.
    n_dates : int
        Total number of daily observations per geo.
    seed : int
        Seed for reproducibility (the estimators are otherwise deterministic).
    noise : float
        Standard deviation of the idiosyncratic noise.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n_dates, freq="D")
    post_start = pd.Timestamp(POST_PERIOD)
    common_trend = rng.normal(0, 1, n_dates).cumsum()  # shared by every geo

    rows = []
    for gi in range(n_test + n_control):
        is_test = gi < n_test
        level = 50.0 + 5.0 * gi
        for di, d in enumerate(dates):
            y = level + common_trend[di] + rng.normal(0, noise)
            if is_test and d >= post_start:
                y += effect
            rows.append({"geo": f"g{gi}", "date": d, "y": float(y), "is_treatment": int(is_test)})
    df = pd.DataFrame(rows)
    n_post_dates = int((dates >= post_start).sum())
    return Panel(df=df, effect=effect, n_test=n_test, n_control=n_control, n_post_dates=n_post_dates)


@pytest.fixture
def panel_factory():
    """Returns the panel builder for tests that need custom configurations."""
    return make_panel


@pytest.fixture(scope="module")
def effect_panel() -> Panel:
    return make_panel(effect=EFFECT)


@pytest.fixture(scope="module")
def null_panel() -> Panel:
    return make_panel(effect=0.0)
