"""Tests for the NNLS simplex least-squares solver shared by the synthetic-control family.

``_solve_simplex_lsq`` solves ``min ||y - x w||^2 + ridge ||w||^2`` subject to
``w >= 0, sum(w) = 1`` as a non-negative least-squares problem. These tests
check it against an independent, tightly converged SLSQP reference across the
shapes the estimators produce (more donors than periods, collinear trending
donors, large outcome scales, ridge penalties), that it falls back to the
SLSQP solver when NNLS fails, and that a panel with a missing geo-date cell
raises a clear error in every estimator that uses it.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.optimize import Bounds, LinearConstraint, minimize

from GeoCausality import _base
from GeoCausality.augmented_synthetic_control import AugmentedSyntheticControl
from GeoCausality.synthetic_control import SyntheticControl, SyntheticControlV
from GeoCausality.synthetic_diff_in_diff import SyntheticDiffInDiff


@pytest.fixture
def solver() -> SyntheticControl:
    # The solver uses no instance state, so skip building a fitted model.
    return SyntheticControl.__new__(SyntheticControl)


def _reference(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """Tightly converged SLSQP with the analytic gradient, as an independent check."""
    n = x.shape[1]
    gram, cross = x.T @ x + ridge * np.identity(n), x.T @ y
    scale = float(np.full(n, 1 / n) @ gram @ np.full(n, 1 / n) - 2 * cross.mean() + y @ y) or 1.0
    res = minimize(
        lambda w: float(w @ gram @ w - 2 * cross @ w + y @ y) / scale,
        np.full(n, 1 / n),
        jac=lambda w: (2 * gram @ w - 2 * cross) / scale,
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(np.ones(n), 1, 1),
        method="SLSQP",
        options={"ftol": 1e-15, "maxiter": 10_000},
    )
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def _problem(n_rows: int, n_donors: int, scale: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Collinear trending donors plus a treated series near (or just outside) their hull."""
    rng = np.random.default_rng(seed)
    trend = rng.normal(size=n_rows).cumsum()
    x = scale * (1 + 0.05 * (rng.normal(0, 3, n_donors) + trend[:, None] + rng.normal(0, 1, (n_rows, n_donors))))
    k = min(n_donors, 4)
    y = x[:, :k] @ rng.dirichlet(np.ones(k)) + scale * 0.01 * rng.normal(size=n_rows)
    return x, y * (1 + 0.03 * rng.normal())


def _loss(x: np.ndarray, y: np.ndarray, ridge: float, w: np.ndarray) -> float:
    return float(np.sum((y - x @ w) ** 2) + ridge * (w @ w))


class TestSolveSimplexLsq:
    @staticmethod
    @pytest.mark.parametrize(
        ("n_rows", "n_donors"), [(40, 5), (96, 30), (20, 60), (120, 150)], ids=["tall", "square-ish", "wide", "large"]
    )
    @pytest.mark.parametrize("scale", [1.0, 3e4], ids=["unit", "gdp-scale"])
    @pytest.mark.parametrize("ridge_frac", [0.0, 1e-3], ids=["no-ridge", "ridge"])
    def test_matches_tight_reference(
        solver: SyntheticControl, n_rows: int, n_donors: int, scale: float, ridge_frac: float
    ) -> None:
        x, y = _problem(n_rows, n_donors, scale, seed=n_rows + n_donors)
        ridge = ridge_frac * scale**2
        weights = solver._solve_simplex_lsq(x, y, "test", ridge=ridge)
        reference = _reference(x, y, ridge)
        # At least as good as the reference optimum, to rounding.
        assert _loss(x, y, ridge, weights) <= _loss(x, y, ridge, reference) * (1 + 1e-9)

    @staticmethod
    def test_returns_simplex_weights(solver: SyntheticControl) -> None:
        x, y = _problem(60, 40, 1e3, seed=0)
        weights = solver._solve_simplex_lsq(x, y, "test")
        assert weights.min() >= 0.0
        assert weights.sum() == pytest.approx(1.0, abs=1e-12)

    @staticmethod
    def test_recovers_exact_convex_combination(solver: SyntheticControl) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=(50, 6))
        truth = np.array([0.5, 0.3, 0.2, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(solver._solve_simplex_lsq(x, x @ truth, "test"), truth, atol=1e-8)

    @staticmethod
    def test_falls_back_to_slsqp_when_nnls_fails(solver: SyntheticControl, monkeypatch: pytest.MonkeyPatch) -> None:
        def failing_nnls(*args, **kwargs):
            raise RuntimeError("Maximum number of iterations reached.")

        monkeypatch.setattr(_base, "nnls", failing_nnls)
        x, y = _problem(40, 8, 1.0, seed=2)
        weights = solver._solve_simplex_lsq(x, y, "test")
        assert weights.sum() == pytest.approx(1.0)
        assert _loss(x, y, 0.0, weights) == pytest.approx(_loss(x, y, 0.0, _reference(x, y, 0.0)), rel=1e-6)

    @staticmethod
    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_rejects_non_finite_input(solver: SyntheticControl, bad: float) -> None:
        x, y = _problem(20, 4, 1.0, seed=3)
        x[5, 1] = bad
        with pytest.raises(ValueError, match="test: the panel has missing or non-finite"):
            solver._solve_simplex_lsq(x, y, "test")


@pytest.mark.parametrize(
    "estimator",
    [SyntheticControl, AugmentedSyntheticControl, SyntheticControlV, SyntheticDiffInDiff],
    ids=lambda cls: cls.__name__,
)
def test_missing_geo_date_raises_clear_error(estimator: type, panel_factory) -> None:
    # A donor with no row for one pre-period date pivots to NaN; NNLS would
    # otherwise fail with scipy's opaque "array must not contain infs or NaNs".
    panel = panel_factory(effect=5.0)
    df = panel.df[~((panel.df["geo"] == "g4") & (panel.df["date"] == pd.Timestamp("2021-01-10")))]
    model = estimator(pl.from_pandas(df), **panel.kwargs())
    with pytest.raises(ValueError, match=f"{estimator.__name__}: the panel has missing or non-finite"):
        model.pre_process().generate()
