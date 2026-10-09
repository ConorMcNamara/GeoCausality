"""Regression tests: numeric time columns compare numerically, the CI label is
not truncated, and zero-spend ROAS keeps the sign of the revenue."""

import numpy as np
import polars as pl

from GeoCausality._base import period_key
from GeoCausality.synthetic_control import SyntheticControl


def _model(**results) -> SyntheticControl:
    model = object.__new__(SyntheticControl)
    model.alpha = 0.1
    model.msrp = 5.0
    model.spend = 0.0
    model.y_variable = "y"
    model.results = results
    return model


def test_integer_week_periods_compare_numerically() -> None:
    rng = np.random.default_rng(0)
    rows = [
        {"geo": g, "week": w, "y": float(rng.normal(100, 1)), "is_treatment": int(g == "t")}
        for g in ["t", "c1", "c2", "c3"]
        for w in range(1, 16)
    ]
    model = SyntheticControl(
        pl.DataFrame(rows),
        geo_variable="geo",
        test_geos=["t"],
        control_geos=["c1", "c2", "c3"],
        date_variable="week",
        pre_period=5,
        post_period=6,
    )
    model.pre_process()
    flags = dict(zip(model.data["week"].to_list(), model.data["treatment_period"].to_list(), strict=True))
    assert flags[5] == 0 and flags[6] == 1
    assert flags[10] == 1  # a string compare would class week 10 as pre-period
    assert all(flags[w] == 0 for w in range(1, 6))


def test_period_key_coerces_for_numeric_columns() -> None:
    frame = pl.DataFrame({"week": [1, 2, 3]})
    import narwhals as nw

    nwf = nw.from_native(frame)
    assert period_key("5", nwf, "week") == 5
    assert period_key(5, nwf, "week") == 5
    assert period_key("2021-01-01") == "2021-01-01"


def test_ci_label_keeps_fractional_percent() -> None:
    model = _model()
    model.alpha = 0.025
    assert model._get_ci_print() == "97.5%"
    model.alpha = 0.1
    assert model._get_ci_print() == "90%"


def test_zero_spend_roas_keeps_sign() -> None:
    model = _model(incrementality=-50.0, incrementality_ci_lower=-np.inf, incrementality_ci_upper=10.0)
    roas, lo, hi = model._get_roas()
    assert roas == -np.inf and lo == -np.inf and hi == np.inf


def test_zero_msrp_unbounded_ci_is_not_nan() -> None:
    model = _model(incrementality=5.0, incrementality_ci_lower=-np.inf, incrementality_ci_upper=np.inf)
    model.msrp = 0.0
    model.spend = 100.0
    assert model._get_roas() == (0.0, 0.0, 0.0)


def test_incremental_cells_round_to_nearest() -> None:
    model = _model()
    cells = model._format_lift_cells("incremental", 0.2, -0.7, 1.6)
    assert cells == (["0"], ["-1"], ["2"])
