"""Regression tests: summarize() must not raise ZeroDivisionError when the
counterfactual/treated post-period sums to zero (the ``relative`` and
``cost-per`` rows divide by those sums, and ``roas`` divides by spend).
It should report ``inf`` instead.
"""

import io
from contextlib import redirect_stdout

import numpy as np
import pytest

from GeoCausality.synthetic_control import SyntheticControl


def _model_with_results(**results) -> SyntheticControl:
    model = object.__new__(SyntheticControl)
    model.alpha = 0.1
    model.msrp = 3.5
    model.spend = 1000.0
    model.y_variable = "y"
    model.results = results
    return model


def _degenerate() -> SyntheticControl:
    # test and counterfactual both sum to zero -> variant == baseline == 0.
    return _model_with_results(
        test=np.array([0.0, 0.0]),
        counterfactual=np.array([0.0, 0.0]),
        incrementality=5.0,
        incrementality_ci_lower=1.0,
        incrementality_ci_upper=9.0,
        p_value=0.2,
    )


@pytest.mark.parametrize("lift", ["relative", "cost-per"])
def test_summarize_zero_denominator_reports_inf(lift: str) -> None:
    model = _degenerate()
    with redirect_stdout(io.StringIO()) as buffer:
        model.summarize(lift)  # must not raise ZeroDivisionError
    assert "inf" in buffer.getvalue()


def test_summarize_roas_zero_variant_does_not_raise() -> None:
    """ROAS divides by spend (not variant/baseline), so a zero post-period
    sum is not degenerate for ROAS -- it should produce a finite result."""
    model = _degenerate()
    with redirect_stdout(io.StringIO()):
        model.summarize("roas")  # must not raise


def test_summarize_roas_zero_spend_reports_inf() -> None:
    """When spend is zero, ROAS = (incr * msrp) / 0 should report inf."""
    model = _model_with_results(
        test=np.array([100.0, 100.0]),
        counterfactual=np.array([90.0, 90.0]),
        incrementality=20.0,
        incrementality_ci_lower=10.0,
        incrementality_ci_upper=30.0,
        p_value=0.05,
    )
    model.spend = 0.0
    model.msrp = 5.0
    with redirect_stdout(io.StringIO()) as buffer:
        model.summarize("roas")
    assert "inf" in buffer.getvalue()
