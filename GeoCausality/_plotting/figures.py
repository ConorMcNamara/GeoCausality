"""Backend-free descriptions of every figure the estimators draw.

An estimator's ``plot()`` computes one of these dataclasses and hands it to
``render``, which draws it with the requested backend. The figure's data and
any interval maths therefore live in one place, and each backend module only
decides how to draw it.
"""

import importlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from GeoCausality._plotting.theme import PlotTheme

PlotBackend = Literal["plotly", "altair"]
_BACKENDS: tuple[str, ...] = ("plotly", "altair")


@dataclass(frozen=True)
class Band:
    """A shaded interval drawn between ``lower`` and ``upper`` over ``dates``."""

    dates: list[Any]
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class CounterfactualPlotData:
    """The three-panel actual / pointwise-difference / cumulative-difference figure.

    Attributes
    ----------
    dates : list
        The full (pre- then post-period) date axis.
    actual, prediction : numpy array
        Observed and counterfactual outcome over ``dates``.
    post_dates : list
        The post-period dates the cumulative panel is drawn over.
    cumulative : numpy array
        Cumulative post-period difference over ``post_dates``.
    post_period : Any
        Where the dashed treatment-start line is drawn.
    ci_label : str
        Legend label for the confidence bands, e.g. ``"95% CI"``.
    prediction_band, difference_band, cumulative_band : Band or None
        The interval shaded on each panel, or ``None`` to draw none.
    """

    dates: list[Any]
    actual: np.ndarray
    prediction: np.ndarray
    post_dates: list[Any]
    cumulative: np.ndarray
    post_period: Any
    ci_label: str
    prediction_band: Band | None = None
    difference_band: Band | None = None
    cumulative_band: Band | None = None

    @property
    def difference(self) -> np.ndarray:
        """Pointwise actual minus counterfactual."""
        return self.actual - self.prediction


@dataclass(frozen=True)
class ParallelTrendsPlotData:
    """Treated and control averages with the level-shifted control counterfactual."""

    treated_dates: list[Any]
    treated: np.ndarray
    control_dates: list[Any]
    control: np.ndarray
    counterfactual: np.ndarray
    post_period: Any
    x_title: str
    y_title: str


@dataclass(frozen=True)
class EventStudyPlotData:
    """Event-study coefficients with their confidence intervals by relative period."""

    periods: np.ndarray
    estimates: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    y_title: str

    @property
    def filled(self) -> np.ndarray:
        """Whether each period's marker is drawn filled: hollow when its interval contains zero.

        The omitted reference period is pinned to a zero-width interval at zero,
        so it is always hollow.
        """
        return (self.lower > 0) | (self.upper < 0)


@dataclass(frozen=True)
class SwitchbackPlotData:
    """Outcome per geo over time, with each point marked treated or not.

    ``frame`` has one row per geo-date and the columns ``geo``, ``date``, ``y``
    and ``treated`` (bool).
    """

    frame: pd.DataFrame
    x_title: str
    y_title: str


@dataclass(frozen=True)
class MarketRankingPlotData:
    """Scores of the top-ranked candidate test sets, best first."""

    labels: list[str]
    scores: list[float]
    powers: list[float]


@dataclass(frozen=True)
class PowerCurvePlotData:
    """Power against effect size, one curve per duration.

    ``frame`` has the columns ``duration``, ``effect`` and ``power``.
    """

    frame: pd.DataFrame
    target: float
    x_title: str


@dataclass(frozen=True)
class MdeSensitivityPlotData:
    """Minimum detectable effect against experiment duration."""

    durations: list[int]
    mdes: list[float]
    target: float
    y_title: str


@dataclass(frozen=True)
class PowerHeatmapPlotData:
    """Power over the effect-size x duration grid; ``power[i, j]`` is for ``durations[i]``, ``effects[j]``."""

    effects: list[float]
    durations: list[int]
    power: np.ndarray
    x_title: str


def check_backend(backend: str) -> None:
    """Raise ``ValueError`` unless ``backend`` is a supported plotting backend."""
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")


def render(data: Any, theme: PlotTheme, backend: PlotBackend = "plotly") -> Any:
    """Draw one of this module's figure descriptions with ``backend``.

    Returns a Plotly ``Figure`` or an Altair chart. Altair is imported only
    when requested, so it stays an optional dependency.
    """
    check_backend(backend)
    renderer = importlib.import_module(f"GeoCausality._plotting._{backend}")
    return renderer.render(data, theme)
