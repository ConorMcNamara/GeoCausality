"""Figures for the estimators' ``plot()`` methods, drawn with Plotly or Altair.

Each ``plot()`` builds a backend-free description from ``figures`` and passes
it to ``render`` with the requested backend; ``_plotly`` and ``_altair`` hold
one renderer per description, and ``theme`` the colours both share.
"""

from GeoCausality._plotting.figures import (
    Band,
    CounterfactualPlotData,
    EventStudyPlotData,
    FinalInterval,
    MarketRankingPlotData,
    MdeSensitivityPlotData,
    ParallelTrendsPlotData,
    PlotBackend,
    PowerCurvePlotData,
    PowerHeatmapPlotData,
    SwitchbackPlotData,
    check_backend,
    render,
)
from GeoCausality._plotting.theme import DARK_THEME, LIGHT_THEME, PlotTheme, get_theme

__all__ = [
    "DARK_THEME",
    "LIGHT_THEME",
    "Band",
    "CounterfactualPlotData",
    "EventStudyPlotData",
    "FinalInterval",
    "MarketRankingPlotData",
    "MdeSensitivityPlotData",
    "ParallelTrendsPlotData",
    "PlotBackend",
    "PlotTheme",
    "PowerCurvePlotData",
    "PowerHeatmapPlotData",
    "SwitchbackPlotData",
    "check_backend",
    "get_theme",
    "render",
]
