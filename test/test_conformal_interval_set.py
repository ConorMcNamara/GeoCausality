"""Tests for the conformal confidence interval's search and its non-interval handling.

``_conformal_interval`` inverts the moving-block test. These tests swap in a
stub test whose acceptance region is known, so they check the search itself:
the outward search stops at ``_CI_SEARCH_MAX_SPREADS`` instead of doubling out
to ~1e13 spreads, an interval is found exactly as before, and a confidence set
that is not a single interval is reported as its outer hull, with its pieces,
and a warning -- so the reported bounds never silently hide effects the test
rejects, such as zero when ``p < alpha``.
"""

import warnings

import numpy as np
import polars as pl
import pytest

from GeoCausality.synthetic_control import SyntheticControl

ALPHA = 0.1


def _stub(accept, monkeypatch: pytest.MonkeyPatch) -> tuple[SyntheticControl, list[float]]:
    """An estimator whose conformal test accepts ``theta`` iff ``accept(theta)``.

    ``resid_fn`` passes ``theta`` through as the residuals, and the patched
    p-value reads it back, so every evaluated effect is also recorded.
    """
    model = SyntheticControl.__new__(SyntheticControl)
    model.alpha = ALPHA
    calls: list[float] = []

    def p_value(pre, post, q=1.0):
        calls.append(float(post[0]))
        return 1.0 if accept(float(post[0])) else 0.0

    monkeypatch.setattr(model, "_conformal_p_value", p_value)
    return model, calls


def _interval(model: SyntheticControl, center: float = 0.0):
    # Residuals with mean ``center`` (the point estimate) and unit spread.
    pre = np.array([-1.0, 1.0]) + center
    post = np.array([center - 1.0, center + 1.0])
    return model._conformal_interval(pre, post, resid_fn=lambda theta: (np.array([theta]), np.array([theta])))


class TestSearch:
    @staticmethod
    def test_single_interval_is_found_without_warning(monkeypatch: pytest.MonkeyPatch) -> None:
        model, _ = _stub(lambda t: -7.0 <= t <= 13.0, monkeypatch)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lower, upper, pieces = _interval(model)
        assert lower == pytest.approx(-7.0, abs=1e-3)
        assert upper == pytest.approx(13.0, abs=1e-3)
        assert pieces == [(lower, upper)]

    @staticmethod
    def test_unbounded_search_stops_at_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
        model, calls = _stub(lambda t: True, monkeypatch)
        lower, upper, pieces = _interval(model)
        assert (lower, upper) == (-np.inf, np.inf)
        assert pieces == [(-np.inf, np.inf)]
        # Doubling from 10 spreads reaches the 1,000-spread cap in 6 probes per
        # side (it used to make 40, out to ~1e13 spreads).
        assert max(abs(t) for t in calls) <= SyntheticControl._CI_SEARCH_MAX_SPREADS
        assert len(calls) <= 2 * 6 + 2

    @staticmethod
    def test_bound_beyond_the_cap_is_reported_unbounded(monkeypatch: pytest.MonkeyPatch) -> None:
        cap = SyntheticControl._CI_SEARCH_MAX_SPREADS
        model, _ = _stub(lambda t: t <= 2 * cap, monkeypatch)
        _, upper, _ = _interval(model)
        assert upper == np.inf


class TestNonInterval:
    @staticmethod
    def test_hole_around_the_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
        # Like Germany with ElasticNet: the estimate and zero are rejected, but
        # every effect far enough away is accepted.
        model, _ = _stub(lambda t: t <= -2.0 or t >= 1.0, monkeypatch)
        with pytest.warns(RuntimeWarning, match="not a single interval"):
            lower, upper, pieces = _interval(model, center=-0.5)
        assert (lower, upper) == (-np.inf, np.inf)
        assert len(pieces) == 2
        assert pieces[0][0] == -np.inf and pieces[0][1] == pytest.approx(-2.0, abs=1e-3)
        assert pieces[1][0] == pytest.approx(1.0, abs=1e-3) and pieces[1][1] == np.inf
        assert not any(lo <= 0.0 <= hi for lo, hi in pieces)

    @staticmethod
    def test_rejected_pocket_skipped_by_doubling_is_found(monkeypatch: pytest.MonkeyPatch) -> None:
        # The estimate (5) is accepted and the region is unbounded both ways, but
        # zero sits in a narrow rejected pocket the outward doubling steps over.
        model, _ = _stub(lambda t: not -0.5 <= t <= 0.5, monkeypatch)
        with pytest.warns(RuntimeWarning, match="not a single interval"):
            lower, upper, pieces = _interval(model, center=5.0)
        assert (lower, upper) == (-np.inf, np.inf)
        assert not any(lo <= 0.0 <= hi for lo, hi in pieces)
        assert [p for p in pieces if np.isfinite(p[1])][0][1] == pytest.approx(-0.5, abs=1e-3)

    @staticmethod
    def test_acceptance_resuming_beyond_a_rejection_widens_to_the_hull(monkeypatch: pytest.MonkeyPatch) -> None:
        # Accepted near the estimate, rejected beyond it, then accepted again far
        # out: the hull is unbounded even though the first rejection is near.
        model, _ = _stub(lambda t: -3.0 <= t <= 3.0 or t >= 400.0, monkeypatch)
        with pytest.warns(RuntimeWarning, match="not a single interval"):
            lower, upper, pieces = _interval(model)
        assert lower == pytest.approx(-3.0, abs=1e-3)
        assert upper == np.inf
        assert pieces[0][1] == pytest.approx(3.0, abs=1e-3)
        assert pieces[-1] == (pytest.approx(400.0, abs=1e-2), np.inf)

    @staticmethod
    def test_every_effect_rejected_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
        model, _ = _stub(lambda t: False, monkeypatch)
        with pytest.warns(RuntimeWarning, match="rejects every constant effect"):
            lower, upper, pieces = _interval(model)
        assert np.isnan(lower) and np.isnan(upper)
        assert pieces == []


def test_results_carry_the_confidence_set(effect_panel) -> None:
    model = SyntheticControl(pl.from_pandas(effect_panel.df), **effect_panel.kwargs()).pre_process().generate()
    results = model.results
    t1 = effect_panel.n_post_dates
    assert results["lift_ci_set"] == [(results["lift_ci_lower"], results["lift_ci_upper"])]
    lo, hi = results["incrementality_ci_set"][0]
    assert lo == pytest.approx(results["lift_ci_lower"] * t1)
    assert hi == pytest.approx(results["lift_ci_upper"] * t1)
