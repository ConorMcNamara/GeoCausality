"""Regression tests for donor-column-order invariance.

``AugmentedSyntheticControl``, ``SyntheticControlV`` and ``RobustSyntheticControl``
fit donor weights on one ``_control_matrix`` pivot and predict on another. If the
two pivots order their donor columns differently, the weight vector is applied to
the wrong donors — so the estimated lift must not depend on the order in which the
donor rows happen to arrive in the input frame.
"""

import numpy as np
import polars as pl
import pytest

from GeoCausality.augmented_synthetic_control import AugmentedSyntheticControl
from GeoCausality.robust_synthetic_control import RobustSyntheticControl
from GeoCausality.synthetic_control import SyntheticControlV

ESTIMATORS = [AugmentedSyntheticControl, SyntheticControlV, RobustSyntheticControl]


def _panel() -> pl.DataFrame:
    # Donor geos with deliberately non-alphabetical insertion order and very
    # different levels, so a column-order bug produces a large, obvious error.
    dates = [f"2021-01-{day:02d}" for day in range(1, 11)]
    rng = np.random.default_rng(0)
    levels = {"z_donor": 500.0, "a_donor": 10.0, "m_donor": 100.0, "treated": 50.0}
    rows = []
    for geo, level in levels.items():
        is_treatment = 1 if geo == "treated" else 0
        for i, date in enumerate(dates):
            y = level + i * (level * 0.01) + rng.normal(0.0, level * 0.02)
            rows.append({"geo": geo, "date": date, "y": y, "is_treatment": is_treatment})
    return pl.DataFrame(rows)


@pytest.mark.parametrize("estimator_cls", ESTIMATORS)
def test_incrementality_invariant_to_donor_row_order(estimator_cls) -> None:
    df = _panel()
    kwargs = dict(
        geo_variable="geo",
        treatment_variable="is_treatment",
        date_variable="date",
        pre_period="2021-01-07",
        post_period="2021-01-08",
        y_variable="y",
    )
    # Geo-sorted vs. insertion-order (z, a, m, treated) — the latter is NOT sorted.
    geo_sorted = df.sort(["geo", "date"])
    unsorted = df.sort(["date"])

    lift_sorted = estimator_cls(geo_sorted, **kwargs).pre_process().generate().results["incrementality"]
    lift_unsorted = estimator_cls(unsorted, **kwargs).pre_process().generate().results["incrementality"]

    assert lift_sorted == pytest.approx(lift_unsorted, rel=1e-9)
