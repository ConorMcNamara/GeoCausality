"""Tests for the backend-agnostic figures in ``GeoCausality._plotting``.

Each estimator's ``plot()`` builds a figure description and ``render`` draws it
with Plotly or Altair. These tests build the descriptions by hand so they check
the renderers alone; the estimator tests cover the data each model supplies.
"""

import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from GeoCausality._plotting import (
    DARK_THEME,
    LIGHT_THEME,
    Band,
    CounterfactualPlotData,
    EventStudyPlotData,
    MarketRankingPlotData,
    MdeSensitivityPlotData,
    ParallelTrendsPlotData,
    PowerCurvePlotData,
    PowerHeatmapPlotData,
    SwitchbackPlotData,
    render,
)

DAILY = [date(2021, 1, 1) + timedelta(days=i) for i in range(10)]


def _counterfactual(dates: list = DAILY, bands: bool = True) -> CounterfactualPlotData:
    n_pre = 6
    n_post = len(dates) - n_pre
    actual = np.linspace(10.0, 20.0, len(dates))
    prediction = actual - np.r_[np.zeros(n_pre), np.full(n_post, 2.0)]
    cumulative = np.cumsum(np.full(n_post, 2.0))
    ones = np.ones(len(dates))
    return CounterfactualPlotData(
        dates=dates,
        actual=actual,
        prediction=prediction,
        post_dates=dates[n_pre:],
        cumulative=cumulative,
        post_period=dates[n_pre],
        ci_label="90% CI",
        prediction_band=Band(dates, prediction - 1.5, prediction + 1.5) if bands else None,
        difference_band=Band(dates, -1.5 * ones, 1.5 * ones) if bands else None,
        cumulative_band=Band(dates[n_pre:], cumulative - 1.0, cumulative + 1.0) if bands else None,
    )


def _parallel_trends() -> ParallelTrendsPlotData:
    treated = np.linspace(5.0, 9.0, len(DAILY))
    control = treated - 1.0
    return ParallelTrendsPlotData(
        treated_dates=DAILY,
        treated=treated,
        control_dates=DAILY,
        control=control,
        counterfactual=control + 1.0,
        post_period=DAILY[6],
        x_title="date",
        y_title="orders",
    )


def _event_study() -> EventStudyPlotData:
    periods = np.arange(-3, 4)
    estimates = np.where(periods >= 0, 2.0, 0.0)
    return EventStudyPlotData(
        periods=periods, estimates=estimates, lower=estimates - 0.5, upper=estimates + 0.5, y_title="Effect on y"
    )


def _switchback() -> SwitchbackPlotData:
    frame = pd.DataFrame(
        [
            {"geo": geo, "date": d, "y": float(i), "treated": i % 2 == 0}
            for geo in ("g0", "g1")
            for i, d in enumerate(DAILY)
        ]
    )
    return SwitchbackPlotData(frame=frame, x_title="date", y_title="y")


def _ranking() -> MarketRankingPlotData:
    return MarketRankingPlotData(labels=["a, b", "c, d", "e, f"], scores=[0.9, 0.7, 0.4], powers=[0.95, 0.8, 0.5])


def _power_curve() -> PowerCurvePlotData:
    frame = pd.DataFrame(
        [{"duration": d, "effect": e, "power": min(1.0, e * d / 10)} for d in (14, 7) for e in (0.0, 0.2, 0.4)]
    )
    return PowerCurvePlotData(frame=frame, target=0.8, x_title="Effect size (multiplicative)")


def _mde() -> MdeSensitivityPlotData:
    return MdeSensitivityPlotData(durations=[7, 14, 21], mdes=[0.4, 0.25, 0.2], target=0.8, y_title="MDE (additive)")


def _heatmap() -> PowerHeatmapPlotData:
    power = np.array([[0.05, 0.4, np.nan], [0.05, 0.7, 0.95]])
    return PowerHeatmapPlotData(effects=[0.0, 0.3, 0.6], durations=[7, 14], power=power, x_title="Effect size")


ALL_FIGURES = [_counterfactual, _parallel_trends, _event_study, _switchback, _ranking, _power_curve, _mde, _heatmap]
IDS = ["counterfactual", "parallel-trends", "event-study", "switchback", "ranking", "power-curve", "mde", "heatmap"]


class TestPlotlyRenderer:
    @staticmethod
    @pytest.mark.parametrize("build", ALL_FIGURES, ids=IDS)
    @pytest.mark.parametrize("theme", [LIGHT_THEME, DARK_THEME], ids=["light", "dark"])
    def test_every_figure_renders(build, theme) -> None:
        fig = render(build(), theme, "plotly")
        assert isinstance(fig, go.Figure)
        assert fig.data

    @staticmethod
    def test_counterfactual_draws_a_band_per_panel() -> None:
        fig = render(_counterfactual(), LIGHT_THEME, "plotly")
        bands = [t for t in fig.data if t.fill == "tonexty"]
        assert len(bands) == 3
        assert len(fig.layout.shapes) == 3
        # Only the top band gets a legend entry.
        assert [t.name for t in bands if t.showlegend is not False] == ["90% CI"]

    @staticmethod
    def test_counterfactual_omits_missing_bands() -> None:
        fig = render(_counterfactual(bands=False), LIGHT_THEME, "plotly")
        assert not [t for t in fig.data if t.fill == "tonexty"]

    @staticmethod
    def test_ranking_puts_best_at_the_top() -> None:
        bar = render(_ranking(), LIGHT_THEME, "plotly").data[0]
        # Horizontal bars draw bottom-up, so the best candidate comes last.
        assert list(bar.y) == ["e, f", "c, d", "a, b"]
        assert list(bar.text) == ["power 50%", "power 80%", "power 95%"]

    @staticmethod
    def test_power_curve_has_one_trace_per_duration_in_order() -> None:
        fig = render(_power_curve(), LIGHT_THEME, "plotly")
        assert [t.name for t in fig.data] == ["7 periods", "14 periods"]
        assert list(fig.data[0].x) == [0.0, 0.2, 0.4]


class TestAltairRenderer:
    alt = pytest.importorskip("altair")

    @pytest.mark.parametrize("build", ALL_FIGURES, ids=IDS)
    @pytest.mark.parametrize("theme", [LIGHT_THEME, DARK_THEME], ids=["light", "dark"])
    def test_every_figure_is_valid_vega_lite(self, build, theme) -> None:
        chart = render(build(), theme, "altair")
        assert isinstance(chart, self.alt.TopLevelMixin)
        chart.to_dict()  # validates against the Vega-Lite schema

    @staticmethod
    def test_counterfactual_builds_three_panels() -> None:
        spec = render(_counterfactual(), LIGHT_THEME, "altair").to_dict()
        assert [panel["title"] for panel in spec["vconcat"]] == [
            "Expected vs Counterfactual",
            "Pointwise Difference",
            "Cumulative Difference",
        ]
        # Each panel: band, line(s), treatment-start rule.
        assert [len(panel["layer"]) for panel in spec["vconcat"]] == [3, 3, 3]

    @staticmethod
    def test_counterfactual_single_legend_lists_series_and_band() -> None:
        spec = render(_counterfactual(), LIGHT_THEME, "altair").to_dict()
        scale = spec["vconcat"][0]["layer"][1]["encoding"]["color"]["scale"]
        assert scale["domain"] == ["Actual", "Counterfactual", "Difference", "Cumulative Incrementality", "90% CI"]
        assert scale["range"][:2] == [LIGHT_THEME.actual, LIGHT_THEME.counterfactual]

    @staticmethod
    def test_counterfactual_omits_missing_bands() -> None:
        spec = render(_counterfactual(bands=False), LIGHT_THEME, "altair").to_dict()
        assert [len(panel["layer"]) for panel in spec["vconcat"]] == [2, 2, 2]
        assert "90% CI" not in spec["vconcat"][0]["layer"][0]["encoding"]["color"]["scale"]["domain"]

    @staticmethod
    def test_dark_mode_styles_background_and_text() -> None:
        spec = render(_counterfactual(), DARK_THEME, "altair").to_dict()
        assert spec["config"]["background"] == DARK_THEME.background
        assert spec["config"]["axis"]["labelColor"] == DARK_THEME.foreground

    @staticmethod
    def test_light_mode_keeps_altair_defaults() -> None:
        spec = render(_counterfactual(), LIGHT_THEME, "altair").to_dict()
        assert "background" not in spec.get("config", {})

    @staticmethod
    @pytest.mark.parametrize(
        ("dates", "x_type"),
        [
            (DAILY, "temporal"),
            ([d.isoformat() for d in DAILY], "temporal"),
            (list(range(2010, 2020)), "quantitative"),
        ],
        ids=["date", "iso-string", "integer"],
    )
    def test_x_axis_type_follows_dates(dates: list, x_type: str) -> None:
        spec = render(_counterfactual(dates), LIGHT_THEME, "altair").to_dict()
        assert spec["vconcat"][0]["layer"][1]["encoding"]["x"]["type"] == x_type

    @staticmethod
    def test_parallel_trends_dashes_only_the_counterfactual() -> None:
        spec = render(_parallel_trends(), LIGHT_THEME, "altair").to_dict()
        dash = spec["layer"][0]["encoding"]["strokeDash"]["scale"]
        assert dict(zip(dash["domain"], dash["range"], strict=True)) == {
            "Treated": [1, 0],
            "Control": [1, 0],
            "Counterfactual": [6, 4],
        }

    @staticmethod
    def test_ranking_keeps_best_first() -> None:
        spec = render(_ranking(), LIGHT_THEME, "altair").to_dict()
        assert spec["layer"][0]["encoding"]["y"]["sort"] == ["a, b", "c, d", "e, f"]

    @staticmethod
    def test_heatmap_drops_missing_cells() -> None:
        chart = render(_heatmap(), LIGHT_THEME, "altair")
        assert len(chart.data) == 5


def test_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="backend must be one of"):
        render(_counterfactual(), LIGHT_THEME, "matplotlib")  # type: ignore[arg-type]


def test_rejects_unknown_figure() -> None:
    with pytest.raises(TypeError, match="No Plotly renderer for dict"):
        render({}, LIGHT_THEME, "plotly")


def test_missing_altair_points_at_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "altair", None)
    monkeypatch.delitem(sys.modules, "GeoCausality._plotting._altair", raising=False)
    with pytest.raises(ImportError, match=r"geocausality\[altair\]"):
        render(_counterfactual(), LIGHT_THEME, "altair")
