"""Every estimator that pivots the panel accepts pandas as well as polars input.

The donor matrix used to be pivoted with polars' native ``pivot(on=...)``, which
pandas spells ``pivot(columns=...)``, so a pandas DataFrame raised ``TypeError``
in every synthetic-control style estimator. The suite only fed them polars, so
it never noticed. These tests fit each estimator on the same panel as pandas
and as polars and require identical results.
"""

import numpy as np
import polars as pl
import pytest

from GeoCausality.augmented_synthetic_control import AugmentedSyntheticControl
from GeoCausality.causal_impact import CausalImpact
from GeoCausality.elastic_net_synthetic_control import ElasticNetSyntheticControl
from GeoCausality.generalized_synthetic_control import GeneralizedSyntheticControl
from GeoCausality.geolift import GeoLift
from GeoCausality.interactive_fixed_effects import InteractiveFixedEffects
from GeoCausality.kernel_synthetic_control import KernelSyntheticControl
from GeoCausality.matrix_completion import MatrixCompletion
from GeoCausality.nonlinear_synthetic_control import NonlinearSyntheticControl
from GeoCausality.penalized_synthetic_control import PenalizedSyntheticControl
from GeoCausality.robust_synthetic_control import RobustSyntheticControl
from GeoCausality.synthetic_control import SyntheticControl, SyntheticControlV
from GeoCausality.synthetic_diff_in_diff import SyntheticDiffInDiff
from test.conftest import EFFECT, make_panel

ESTIMATORS = [
    SyntheticControl,
    SyntheticControlV,
    AugmentedSyntheticControl,
    CausalImpact,
    ElasticNetSyntheticControl,
    GeneralizedSyntheticControl,
    InteractiveFixedEffects,
    KernelSyntheticControl,
    MatrixCompletion,
    NonlinearSyntheticControl,
    PenalizedSyntheticControl,
    RobustSyntheticControl,
    SyntheticDiffInDiff,
]
RESULT_KEYS = ("incrementality", "incrementality_ci_lower", "incrementality_ci_upper", "p_value")


@pytest.fixture(scope="module")
def panel():
    # One treated geo keeps the treated series on the donors' scale.
    return make_panel(effect=EFFECT, n_test=1, n_control=6)


def _fit(cls, data, panel) -> dict:
    kwargs = panel.kwargs()
    if cls is GeoLift:
        kwargs["n_boot"] = 50
    return cls(data, **kwargs).pre_process().generate().results


def _assert_same_results(pandas_results: dict, polars_results: dict) -> None:
    for key in RESULT_KEYS:
        if pandas_results.get(key) is None:
            assert polars_results.get(key) is None, key
            continue
        np.testing.assert_allclose(float(pandas_results[key]), float(polars_results[key]), rtol=1e-9, err_msg=key)


@pytest.mark.parametrize("cls", ESTIMATORS, ids=lambda c: c.__name__)
def test_pandas_and_polars_input_agree(cls, panel) -> None:
    _assert_same_results(_fit(cls, panel.df, panel), _fit(cls, pl.from_pandas(panel.df), panel))


def test_geolift_accepts_pandas(panel) -> None:
    _assert_same_results(_fit(GeoLift, panel.df, panel), _fit(GeoLift, pl.from_pandas(panel.df), panel))
