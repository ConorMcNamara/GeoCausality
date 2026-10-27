"""Plotly renderers for the figure descriptions in ``figures``."""

from functools import singledispatch
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from GeoCausality._plotting.figures import (
    Band,
    CounterfactualPlotData,
    EventStudyPlotData,
    FinalInterval,
    MarketRankingPlotData,
    MdeSensitivityPlotData,
    ParallelTrendsPlotData,
    PowerCurvePlotData,
    PowerHeatmapPlotData,
    SwitchbackPlotData,
)
from GeoCausality._plotting.theme import PlotTheme


def apply_theme(fig: go.Figure, theme: PlotTheme) -> go.Figure:
    """Apply the theme's Plotly template to ``fig`` and return it.

    The light theme has no template, so it leaves the caller's
    ``plotly.io.templates.default`` in effect.
    """
    if theme.template is not None:
        fig.update_layout(template=theme.template)
    return fig


@singledispatch
def render(data: Any, theme: PlotTheme) -> go.Figure:
    """Draw ``data`` as a Plotly figure."""
    raise TypeError(f"No Plotly renderer for {type(data).__name__}")


def _band_traces(band: Band, fill: str, name: str | None = None, in_legend: bool = False) -> list[go.Scatter]:
    """Draw an invisible upper edge plus a lower edge filled up to it.

    ``name`` labels the band on hover; ``in_legend`` also lists it in the legend.
    """
    legend = {} if in_legend else {"showlegend": False}
    return [
        go.Scatter(x=band.dates, y=band.upper, mode="lines", line={"width": 0}, showlegend=False),
        go.Scatter(
            x=band.dates,
            y=band.lower,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=fill,
            name=name,
            **legend,
        ),
    ]


def _zero_free_trace(band: Band, fill: str) -> go.Scatter | None:
    """Shade the runs of ``band`` that exclude zero, over the band itself; ``None`` when none do.

    Each run is a closed polygon, separated from the next by a gap, so one trace draws them all.
    """
    x: list[Any] = []
    y: list[Any] = []
    for run in band.zero_free_runs():
        dates = band.dates[run]
        x += [*dates, *dates[::-1], None]
        y += [*band.upper[run], *band.lower[run][::-1], None]
    if not x:
        return None
    return go.Scatter(
        x=x, y=y, mode="lines", line={"width": 0}, fill="toself", fillcolor=fill, hoverinfo="skip", showlegend=False
    )


def _final_interval_trace(final: FinalInterval, colour: str) -> go.Scatter:
    """Draw the last cumulative estimate with its interval, hollow when the interval contains zero."""
    return go.Scatter(
        x=[final.date],
        y=[final.estimate],
        mode="markers",
        marker={
            "symbol": "circle" if final.filled else "circle-open",
            "size": 10,
            "color": colour,
            "line": {"width": 2, "color": colour},
        },
        error_y={
            "type": "data",
            "symmetric": False,
            "array": [final.upper - final.estimate],
            "arrayminus": [final.estimate - final.lower],
            "color": colour,
            "thickness": 2,
            "width": 6,
        },
        name="Final cumulative",
        showlegend=False,
    )


@render.register
def _(data: CounterfactualPlotData, theme: PlotTheme) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Expected vs Counterfactual", "Pointwise Difference", "Cumulative Difference"),
    )
    band_fill = theme.band(0.2)
    panels: list[list[go.Scatter]] = [[], [], []]

    # Bands first so each panel's series draw on top of them.
    if data.prediction_band is not None:
        panels[0] += _band_traces(data.prediction_band, band_fill, name=data.ci_label, in_legend=True)
    # Where a band excludes zero it is shaded more strongly.
    if data.difference_band is not None:
        panels[1] += _band_traces(data.difference_band, band_fill)
        if (trace := _zero_free_trace(data.difference_band, theme.band(0.3))) is not None:
            panels[1].append(trace)
    if data.cumulative_band is not None:
        panels[2] += _band_traces(data.cumulative_band, theme.cumulative_band(0.2), name=data.ci_label)
        if (trace := _zero_free_trace(data.cumulative_band, theme.cumulative_band(0.3))) is not None:
            panels[2].append(trace)
    panels[0] += [
        go.Scatter(x=data.dates, y=data.actual, marker={"color": theme.actual}, mode="lines", name="Actual"),
        go.Scatter(
            x=data.dates,
            y=data.prediction,
            marker={"color": theme.counterfactual},
            mode="lines",
            name="Counterfactual",
        ),
    ]
    panels[1].append(
        go.Scatter(x=data.dates, y=data.difference, marker={"color": theme.difference}, mode="lines", name="Difference")
    )
    panels[2].append(
        go.Scatter(
            x=data.post_dates,
            y=data.cumulative,
            marker={"color": theme.cumulative},
            mode="lines",
            name="Cumulative Incrementality",
        )
    )
    if (final := data.final_interval) is not None:
        panels[2].append(_final_interval_trace(final, theme.cumulative))

    for row, traces in enumerate(panels, start=1):
        for trace in traces:
            fig.add_trace(trace, row=row, col=1)
        fig.add_vline(
            x=data.post_period,
            line_width=1,
            line_dash="dash",
            line_color=theme.reference,
            row=row,
            col=1,
        )
    # The difference panels read against zero, the no-effect line.
    for row in (2, 3):
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color=theme.zero, row=row, col=1)
    return apply_theme(fig, theme)


@render.register
def _(data: ParallelTrendsPlotData, theme: PlotTheme) -> go.Figure:
    fig = go.Figure(
        [
            go.Scatter(
                x=data.treated_dates,
                y=data.treated,
                marker={"color": theme.actual},
                mode="lines",
                name="Treated",
            ),
            go.Scatter(
                x=data.control_dates,
                y=data.control,
                marker={"color": theme.control},
                mode="lines",
                name="Control",
            ),
            go.Scatter(
                x=data.control_dates,
                y=data.counterfactual,
                marker={"color": theme.counterfactual},
                mode="lines",
                line={"dash": "dash"},
                name="Counterfactual",
            ),
        ]
    )
    fig.add_vline(x=data.post_period, line_width=1, line_dash="dash", line_color=theme.reference)
    fig.update_layout(title="Parallel Trends", xaxis_title=data.x_title, yaxis_title=data.y_title)
    return apply_theme(fig, theme)


@render.register
def _(data: EventStudyPlotData, theme: PlotTheme) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=data.periods,
            y=data.estimates,
            error_y={
                "type": "data",
                "symmetric": False,
                "array": data.upper - data.estimates,
                "arrayminus": data.estimates - data.lower,
            },
            # Hollow when the interval contains zero.
            marker={
                "color": theme.difference,
                "symbol": np.where(data.filled, "circle", "circle-open").tolist(),
                # Unset, the outline takes the trace line's colour, which would ring filled markers.
                "line": {"width": 2, "color": theme.difference},
            },
            line={"color": theme.difference},
            mode="markers+lines",
            name="Effect",
        )
    )
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color=theme.zero)
    fig.add_vline(x=-0.5, line_width=1, line_dash="dash", line_color=theme.reference)
    fig.update_layout(
        title="Event Study (Two-Way Fixed Effects)",
        xaxis_title="Periods relative to treatment onset",
        yaxis_title=data.y_title,
    )
    return apply_theme(fig, theme)


@render.register
def _(data: SwitchbackPlotData, theme: PlotTheme) -> go.Figure:
    fig = go.Figure()
    for geo, geo_data in data.frame.groupby("geo", sort=True):
        geo_data = geo_data.sort_values(by="date")
        colors = [theme.counterfactual if t else theme.actual for t in geo_data["treated"]]
        fig.add_trace(
            go.Scatter(
                x=geo_data["date"],
                y=geo_data["y"],
                mode="lines+markers",
                marker={"color": colors, "size": 4},
                name=str(geo),
            )
        )
    fig.update_layout(
        title="Switchback Design: Outcome by Geo",
        xaxis_title=data.x_title,
        yaxis_title=data.y_title,
    )
    return apply_theme(fig, theme)


@render.register
def _(data: MarketRankingPlotData, theme: PlotTheme) -> go.Figure:
    # Reversed so the best candidate sits at the top of the horizontal bars.
    fig = go.Figure(
        go.Bar(
            x=data.scores[::-1],
            y=data.labels[::-1],
            orientation="h",
            text=[f"power {p:.0%}" for p in data.powers[::-1]],
        )
    )
    fig.update_layout(
        title="Top candidate test markets",
        xaxis_title="Score",
        yaxis_title="Test geos",
        xaxis_range=[0, 1],
    )
    return apply_theme(fig, theme)


@render.register
def _(data: PowerCurvePlotData, theme: PlotTheme) -> go.Figure:
    fig = go.Figure()
    for duration, curve in data.frame.groupby("duration", sort=True):
        curve = curve.sort_values(by="effect")
        fig.add_trace(
            go.Scatter(
                x=curve["effect"].tolist(), y=curve["power"].tolist(), mode="lines+markers", name=f"{duration} periods"
            )
        )
    fig.add_hline(y=data.target, line_dash="dash", annotation_text=f"target {data.target:.0%}")
    fig.update_layout(title="Power curve", xaxis_title=data.x_title, yaxis_title="Power", yaxis_range=[0, 1])
    return apply_theme(fig, theme)


@render.register
def _(data: MdeSensitivityPlotData, theme: PlotTheme) -> go.Figure:
    fig = go.Figure(go.Scatter(x=data.durations, y=data.mdes, mode="lines+markers"))
    fig.update_layout(
        title=f"MDE sensitivity (target power {data.target:.0%})",
        xaxis_title="Experiment duration (periods)",
        yaxis_title=data.y_title,
    )
    return apply_theme(fig, theme)


@render.register
def _(data: PowerHeatmapPlotData, theme: PlotTheme) -> go.Figure:
    z = np.asarray(data.power, dtype=float).tolist()
    fig = go.Figure(
        go.Heatmap(
            x=data.effects,
            y=data.durations,
            z=z,
            zmin=0,
            zmax=1,
            colorscale="RdYlGn",
            colorbar_title="Power",
            text=[[f"{v:.0%}" for v in row] for row in z],
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="Power heatmap",
        xaxis_title=data.x_title,
        yaxis_title="Experiment duration (periods)",
    )
    return apply_theme(fig, theme)
