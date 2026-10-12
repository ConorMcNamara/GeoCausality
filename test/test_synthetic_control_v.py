"""Tests for the nested-V synthetic control's outer optimiser."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from GeoCausality.synthetic_control import SyntheticControlV


def test_warns_when_v_search_does_not_converge() -> None:
    """Nelder-Mead over one V entry per pre-period date stops at maxiter on long pre-periods."""
    rng = np.random.default_rng(0)
    n_dates = 65
    dates = [date(2021, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    factors = rng.normal(0, 1, size=(2, n_dates)).cumsum(axis=1)
    rows = []
    for g in range(10):
        loading = rng.uniform(0.5, 2.0, size=2)
        for di, d in enumerate(dates):
            y = 100.0 + loading @ factors[:, di] + rng.normal(0, 2)
            rows.append({"geo": f"g{g}", "date": d, "y": float(y), "is_treatment": int(g < 2)})
    model = SyntheticControlV(pl.DataFrame(rows), pre_period="2021-03-01", post_period="2021-03-02")
    with pytest.warns(RuntimeWarning, match="SyntheticControlV: weight optimizer did not converge"):
        model.pre_process().generate()
