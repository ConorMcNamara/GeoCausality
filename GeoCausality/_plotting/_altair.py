"""Altair renderers for the figure descriptions in ``figures``.

Importing this module imports Altair, so ``figures.render`` only imports it
when the Altair backend is requested.
"""

from functools import singledispatch
from typing import Any, Literal

import numpy as np
import pandas as pd

from GeoCausality._plotting.figures import (
    Band,
    CounterfactualPlotData,
    EventStudyPlotData,
    MarketRankingPlotData,
    MdeSensitivityPlotData,
    ParallelTrendsPlotData,
    PowerCurvePlotData,
    PowerHeatmapPlotData,
    SwitchbackPlotData,
)
from GeoCausality._plotting.theme import PlotTheme

try:
    import altair as alt
except ImportError as exc:
    raise ImportError(
        "The altair plotting backend requires altair. Install it with `pip install geocausality[altair]`."
    ) from exc

XType = Literal["quantitative", "temporal"]

_WIDTH = 600
_HEIGHT = 300


@singledispatch
def _draw(data: Any, theme: PlotTheme) -> Any:
    raise TypeError(f"No Altair renderer for {type(data).__name__}")


def render(data: Any, theme: PlotTheme) -> Any:
    """Draw ``data`` as an Altair chart styled with ``theme``."""
    return _style(_draw(data, theme), theme)


def _style(chart: Any, theme: PlotTheme) -> Any:
    """Apply the theme's background, text and grid colours to a top-level chart."""
    # Drop colliding tick labels. Flushed end labels shift inward after the
    # overlap check, so they would still collide with their neighbours.
    config: dict[str, Any] = {"axis": {"labelOverlap": True, "labelFlush": False}}
    if theme.foreground is not None:
        config["axis"].update(labelColor=theme.foreground, titleColor=theme.foreground, gridColor=theme.grid)
        config["legend"] = {"labelColor": theme.foreground, "titleColor": theme.foreground}
        config["title"] = {"color": theme.foreground}
    if theme.background is not None:
        config["background"] = theme.background
    # One ``configure`` call: each call replaces the chart's whole config.
    return chart.configure(**config)


def _x_column(values: Any) -> tuple[pd.Series, XType]:
    """Convert date-axis values to a pandas column and its Vega-Lite type.

    Numeric axes (integer years, week indices) stay quantitative; dates,
    datetimes and ISO strings are parsed so Altair draws a time axis.
    """
    series = pd.Series(list(values))
    if pd.api.types.is_numeric_dtype(series):
        return series, "quantitative"
    return pd.to_datetime(series), "temporal"


def _series_colour(names: list[str], colours: list[str]) -> Any:
    """Colour the ``series`` column with fixed per-series colours."""
    return alt.Color(
        "series:N",
        scale=alt.Scale(domain=names, range=colours),
        # Solid swatches, even when a translucent band shares the scale.
        legend=alt.Legend(symbolOpacity=1, symbolType="square"),
        title=None,
    )


def _start_rule(post_period: Any, x: Any, theme: PlotTheme) -> Any:
    """Draw the dashed vertical line marking the start of the post-period."""
    start, _ = _x_column([post_period])
    return alt.Chart(pd.DataFrame({"date": start})).mark_rule(color=theme.reference, strokeDash=[4, 4]).encode(x=x)


@_draw.register
def _(data: CounterfactualPlotData, theme: PlotTheme) -> Any:
    _, x_type = _x_column(data.dates)
    x = alt.X("date", type=x_type, title=None)
    # Fit the axis to the data like Plotly does, rather than Altair's zero baseline.
    y = alt.Y("value:Q", title=None, scale=alt.Scale(zero=False))

    # One shared colour scale gives a single legend across the three panels.
    names = ["Actual", "Counterfactual", "Difference", "Cumulative Incrementality"]
    colours = [theme.actual, theme.counterfactual, theme.difference, theme.cumulative]
    if data.prediction_band is not None:
        names.append(data.ci_label)
        colours.append(theme.band(1.0))
    colour = _series_colour(names, colours)

    def lines(dates: list[Any], **series: np.ndarray) -> Any:
        frame = pd.DataFrame({"date": _x_column(dates)[0], **series})
        long = frame.melt(id_vars="date", var_name="series", value_name="value")
        tooltip = [alt.Tooltip("date", type=x_type), "series:N", alt.Tooltip("value:Q", format=",.2f")]
        return alt.Chart(long).mark_line().encode(x=x, y=y, color=colour, tooltip=tooltip)

    def area(band: Band, legend: str | None = None, **mark: Any) -> Any:
        frame = pd.DataFrame({"date": _x_column(band.dates)[0], "lower": band.lower, "upper": band.upper})
        band_y = alt.Y("lower:Q", title=None, scale=alt.Scale(zero=False))
        chart = alt.Chart(frame.assign(series=legend)).mark_area(opacity=0.2, **mark)
        chart = chart.encode(x=x, y=band_y, y2="upper:Q")
        # A labelled band is coloured through the shared scale so it gets a legend entry.
        return chart.encode(color=colour) if legend is not None else chart

    top: list[Any] = []
    middle: list[Any] = []
    bottom: list[Any] = []
    if data.prediction_band is not None:
        top.append(area(data.prediction_band, legend=data.ci_label))
    if data.difference_band is not None:
        middle.append(area(data.difference_band, color=theme.band(1.0)))
    if data.cumulative_band is not None:
        bottom.append(area(data.cumulative_band, color=theme.cumulative_band(1.0)))
    top.append(lines(data.dates, Actual=data.actual, Counterfactual=data.prediction))
    middle.append(lines(data.dates, Difference=data.difference))
    bottom.append(lines(data.post_dates, **{"Cumulative Incrementality": data.cumulative}))

    start = _start_rule(data.post_period, x, theme)
    titles = ("Expected vs Counterfactual", "Pointwise Difference", "Cumulative Difference")
    return alt.vconcat(
        *(
            alt.layer(*layers, start).properties(title=title, width=_WIDTH, height=180)
            for layers, title in zip((top, middle, bottom), titles, strict=True)
        )
    )


@_draw.register
def _(data: ParallelTrendsPlotData, theme: PlotTheme) -> Any:
    treated_x, x_type = _x_column(data.treated_dates)
    control_x, _ = _x_column(data.control_dates)
    frame = pd.concat(
        [
            pd.DataFrame({"date": treated_x, "value": data.treated, "series": "Treated"}),
            pd.DataFrame({"date": control_x, "value": data.control, "series": "Control"}),
            pd.DataFrame({"date": control_x, "value": data.counterfactual, "series": "Counterfactual"}),
        ]
    )
    names = ["Treated", "Control", "Counterfactual"]
    x = alt.X("date", type=x_type, title=data.x_title)
    lines = (
        alt.Chart(frame)
        .mark_line()
        .encode(
            x=x,
            y=alt.Y("value:Q", title=data.y_title, scale=alt.Scale(zero=False)),
            color=_series_colour(names, [theme.actual, theme.control, theme.counterfactual]),
            # Dash only the counterfactual, as Plotly does.
            strokeDash=alt.StrokeDash(
                "series:N", scale=alt.Scale(domain=names, range=[[1, 0], [1, 0], [6, 4]]), legend=None
            ),
            tooltip=[alt.Tooltip("date", type=x_type), "series:N", alt.Tooltip("value:Q", format=",.2f")],
        )
    )
    return alt.layer(lines, _start_rule(data.post_period, x, theme)).properties(
        title="Parallel Trends", width=_WIDTH, height=_HEIGHT
    )


@_draw.register
def _(data: EventStudyPlotData, theme: PlotTheme) -> Any:
    frame = pd.DataFrame({"period": data.periods, "estimate": data.estimates, "lower": data.lower, "upper": data.upper})
    x = alt.X("period:Q", title="Periods relative to treatment onset")
    base = alt.Chart(frame).encode(x=x)
    tooltip = [
        "period:Q",
        alt.Tooltip("estimate:Q", format=",.2f"),
        alt.Tooltip("lower:Q", format=",.2f"),
        alt.Tooltip("upper:Q", format=",.2f"),
    ]
    effect = base.mark_line(color=theme.difference, point=alt.OverlayMarkDef(color=theme.difference)).encode(
        y=alt.Y("estimate:Q", title=data.y_title), tooltip=tooltip
    )
    intervals = base.mark_rule(color=theme.difference).encode(y="lower:Q", y2="upper:Q")
    zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color=theme.zero, strokeDash=[2, 2]).encode(y="y:Q")
    onset = (
        alt.Chart(pd.DataFrame({"period": [-0.5]}))
        .mark_rule(color=theme.reference, strokeDash=[4, 4])
        .encode(x="period:Q")
    )
    return alt.layer(zero, onset, intervals, effect).properties(
        title="Event Study (Two-Way Fixed Effects)", width=_WIDTH, height=_HEIGHT
    )


@_draw.register
def _(data: SwitchbackPlotData, theme: PlotTheme) -> Any:
    dates, x_type = _x_column(data.frame["date"])
    frame = data.frame.assign(
        date=dates.to_numpy(),
        geo=data.frame["geo"].astype(str),
        status=np.where(data.frame["treated"], "Treatment", "Control"),
    ).drop(columns="treated")
    base = alt.Chart(frame).encode(
        x=alt.X("date", type=x_type, title=data.x_title),
        y=alt.Y("y:Q", title=data.y_title, scale=alt.Scale(zero=False)),
    )
    # Lines take a colour per geo; points are filled by treatment status, as in Plotly.
    lines = base.mark_line().encode(color=alt.Color("geo:N", title="Geo"))
    points = base.mark_circle(size=25, opacity=1).encode(
        fill=alt.Fill(
            "status:N",
            scale=alt.Scale(domain=["Treatment", "Control"], range=[theme.counterfactual, theme.actual]),
            title="Status",
        ),
        tooltip=["geo:N", alt.Tooltip("date", type=x_type), "status:N", alt.Tooltip("y:Q", format=",.2f")],
    )
    return alt.layer(lines, points).properties(title="Switchback Design: Outcome by Geo", width=_WIDTH, height=_HEIGHT)


@_draw.register
def _(data: MarketRankingPlotData, theme: PlotTheme) -> Any:
    frame = pd.DataFrame({"label": data.labels, "score": data.scores, "power": [f"power {p:.0%}" for p in data.powers]})
    base = alt.Chart(frame).encode(
        x=alt.X("score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("label:N", title="Test geos", sort=data.labels),
    )
    tooltip = ["label:N", alt.Tooltip("score:Q", format=".3f"), "power:N"]
    bars = base.mark_bar().encode(tooltip=tooltip)
    text = base.mark_text(align="left", dx=4, color=theme.foreground or "black").encode(text="power:N")
    return alt.layer(bars, text).properties(title="Top candidate test markets", width=_WIDTH, height=alt.Step(28))


@_draw.register
def _(data: PowerCurvePlotData, theme: PlotTheme) -> Any:
    durations = sorted(data.frame["duration"].unique())
    labels = [f"{d} periods" for d in durations]
    frame = data.frame.assign(series=data.frame["duration"].map(lambda d: f"{d} periods"))
    curves = (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(
            x=alt.X("effect:Q", title=data.x_title),
            y=alt.Y("power:Q", title="Power", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("series:N", sort=labels, title=None),
            tooltip=["series:N", "effect:Q", alt.Tooltip("power:Q", format=".0%")],
        )
    )
    target = alt.Chart(pd.DataFrame({"power": [data.target], "label": [f"target {data.target:.0%}"]}))
    rule = target.mark_rule(color=theme.reference, strokeDash=[4, 4]).encode(y="power:Q")
    label = target.mark_text(align="right", baseline="bottom", x="width", dy=-2, color=theme.reference).encode(
        y="power:Q", text="label:N"
    )
    return alt.layer(curves, rule, label).properties(title="Power curve", width=_WIDTH, height=_HEIGHT)


@_draw.register
def _(data: MdeSensitivityPlotData, theme: PlotTheme) -> Any:
    frame = pd.DataFrame({"duration": data.durations, "mde": data.mdes})
    return (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(
            x=alt.X("duration:Q", title="Experiment duration (periods)"),
            y=alt.Y("mde:Q", title=data.y_title, scale=alt.Scale(zero=False)),
            tooltip=["duration:Q", alt.Tooltip("mde:Q", format=",.3f")],
        )
        .properties(title=f"MDE sensitivity (target power {data.target:.0%})", width=_WIDTH, height=_HEIGHT)
    )


@_draw.register
def _(data: PowerHeatmapPlotData, theme: PlotTheme) -> Any:
    power = np.asarray(data.power, dtype=float)
    frame = pd.DataFrame(
        [
            {"effect": effect, "duration": duration, "power": power[i, j]}
            for i, duration in enumerate(data.durations)
            for j, effect in enumerate(data.effects)
        ]
    ).dropna(subset=["power"])
    base = alt.Chart(frame).encode(
        x=alt.X("effect:O", title=data.x_title, axis=alt.Axis(labelAngle=0)),
        # Longest duration at the top, matching Plotly's upward y-axis.
        y=alt.Y("duration:O", title="Experiment duration (periods)", sort="descending"),
    )
    cells = base.mark_rect().encode(
        color=alt.Color("power:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[0, 1]), title="Power"),
        tooltip=["effect:O", "duration:O", alt.Tooltip("power:Q", format=".0%")],
    )
    # White on the scheme's dark ends, black on its light middle, as Plotly does.
    text = base.mark_text().encode(
        text=alt.Text("power:Q", format=".0%"),
        color=alt.condition("datum.power < 0.2 || datum.power > 0.85", alt.value("white"), alt.value("black")),
    )
    return alt.layer(cells, text).properties(title="Power heatmap", width=_WIDTH, height=alt.Step(40))
