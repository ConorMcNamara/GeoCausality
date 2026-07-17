"""Regression tests for the split-conformal band rank.

The band is the ``ceil((n + 1)(1 - alpha))``-th smallest absolute residual
(saturating to the max when that rank exceeds ``n``) -- the same conformalized
quantile as ``_jackknife_quantile``. A previous implementation used
``np.quantile(scores, k / n, method="higher")`` and returned one order statistic
too high.
"""

import numpy as np
import pytest

from GeoCausality.synthetic_control import SyntheticControl


def _estimator(alpha: float = 0.1) -> SyntheticControl:
    model = object.__new__(SyntheticControl)
    model.alpha = alpha
    return model


@pytest.mark.parametrize(
    ("alpha", "expected"),
    [(0.1, 19.0), (0.2, 17.0)],  # n=20: k=ceil(0.9*21)=19 -> scores[18]; k=ceil(0.8*21)=17 -> scores[16]
)
def test_split_conformal_band_is_kth_smallest(alpha: float, expected: float) -> None:
    model = _estimator(alpha)
    scores = np.arange(1, 21, dtype=float)
    assert model._split_conformal_band(scores) == expected


def test_split_conformal_band_saturates_to_max() -> None:
    # n=3, alpha=0.1: k=ceil(0.9*4)=4 > 3 -> widen to the maximum residual.
    model = _estimator(0.1)
    assert model._split_conformal_band(np.array([2.0, 5.0, 1.0])) == 5.0


def test_split_conformal_band_matches_jackknife_quantile() -> None:
    model = _estimator(0.1)
    resid = np.random.default_rng(0).normal(size=37)
    assert model._split_conformal_band(resid) == model._jackknife_quantile(resid, 0.1)
