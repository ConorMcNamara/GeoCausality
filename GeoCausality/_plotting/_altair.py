"""Altair renderers for the figure descriptions in ``figures``.

Importing this module imports Altair, so ``figures.render`` only imports it
when the Altair backend is requested.
"""

import warnings
from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Literal

import numpy as np
import pandas as pd

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

try:
    import altair as alt
except ImportError as exc:
    raise ImportError(
        "The altair plotting backend requires altair. Install it with `pip install geocausality[altair]`."
    ) from exc

XType = Literal["quantitative", "temporal", "ordinal"]

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


@dataclass(frozen=True)
class _XAxis:
    """The date axis shared by every layer of a chart.

    The type comes from the dates themselves, so values drawn against them
    (the post-period start, band dates) are coerced to match rather than
    typed on their own: a string ``"1989"`` on an integer-year axis is 1989.
    """

    type: XType
    # Category order for an ordinal axis, which Vega-Lite would otherwise sort.
    order: list[str] | None = None

    @classmethod
    def of(cls, *dates: Any) -> "_XAxis":
        """Pick the axis type for one or more lists of dates.

        Numeric axes (integer years, week indices) stay quantitative; dates,
        datetimes and ISO strings are drawn as time; other labels ("2021-W01",
        "week_01") are categories in data order, as Plotly draws them.
        """
        series = pd.Series([value for values in dates for value in values])
        if pd.api.types.is_numeric_dtype(series):
            return cls("quantitative")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # format-inference warnings on unparseable labels
                pd.to_datetime(series)
        except (ValueError, TypeError):
            return cls("ordinal", list(dict.fromkeys(series.astype(str))))
        return cls("temporal")

    def column(self, values: Any) -> pd.Series:
        """Convert ``values`` to this axis's type."""
        series = pd.Series(list(values))
        if self.type == "quantitative":
            return pd.to_numeric(series)
        if self.type == "temporal":
            return pd.to_datetime(series)
        return series.astype(str)

    def encode(self, title: str | None) -> Any:
        return alt.X("date", type=self.type, title=title, sort=self.order or alt.Undefined)

    def tooltip(self) -> Any:
        return alt.Tooltip("date", type=self.type)


def _chart(frame: pd.DataFrame) -> Any:
    """Start a chart on ``frame`` with its rows inlined.

    Inline values skip Altair's 5000-row limit, which long daily series and
    geo-by-date frames exceed, without changing the global data transformer.
    """
    return alt.Chart(alt.InlineData(values=alt.to_values(frame)["values"]))


def _series_colour(names: list[str], colours: list[str]) -> Any:
    """Colour the ``series`` column with fixed per-series colours."""
    return alt.Color(
        "series:N",
        scale=alt.Scale(domain=names, range=colours),
        # Solid swatches, even when a translucent band shares the scale.
        legend=alt.Legend(symbolOpacity=1, symbolType="square"),
        title=None,
    )


def _start_rule(post_period: Any, axis: _XAxis, theme: PlotTheme) -> Any:
    """Draw the dashed vertical line marking the start of the post-period."""
    start = pd.DataFrame({"date": axis.column([post_period])})
    return _chart(start).mark_rule(color=theme.reference, strokeDash=[4, 4]).encode(x=axis.encode(None))


@_draw.register
def _(data: CounterfactualPlotData, theme: PlotTheme) -> Any:
    axis = _XAxis.of(data.dates)
    x = axis.encode(None)
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
        frame = pd.DataFrame({"date": axis.column(dates), **series})
        long = frame.melt(id_vars="date", var_name="series", value_name="value")
        tooltip = [axis.tooltip(), "series:N", alt.Tooltip("value:Q", format=",.2f")]
        return _chart(long).mark_line().encode(x=x, y=y, color=colour, tooltip=tooltip)

    def area(band: Band, legend: str | None = None, **mark: Any) -> Any:
        frame = pd.DataFrame({"date": axis.column(band.dates), "lower": band.lower, "upper": band.upper})
        band_y = alt.Y("lower:Q", title=None, scale=alt.Scale(zero=False))
        chart = _chart(frame.assign(series=legend)).mark_area(opacity=0.2, **mark)
        chart = chart.encode(x=x, y=band_y, y2="upper:Q")
        # A labelled band is coloured through the shared scale so it gets a legend entry.
        return chart.encode(color=colour) if legend is not None else chart

    def zero_free_area(band: Band, color: str) -> list[Any]:
        """Shade the runs of ``band`` that exclude zero, over the band itself; nothing when none do."""
        runs = band.zero_free_runs()
        if not runs:
            return []
        frame = pd.concat(
            pd.DataFrame(
                {"date": axis.column(band.dates[run]), "lower": band.lower[run], "upper": band.upper[run], "run": i}
            )
            for i, run in enumerate(runs)
        )
        band_y = alt.Y("lower:Q", title=None, scale=alt.Scale(zero=False))
        # ``detail`` draws each run as its own area rather than bridging the gaps between them.
        chart = _chart(frame).mark_area(opacity=0.3, color=color)
        return [chart.encode(x=x, y=band_y, y2="upper:Q", detail="run:N")]

    def final_interval(final: FinalInterval) -> list[Any]:
        """Draw the last cumulative estimate with its interval, hollow when the interval contains zero."""
        frame = pd.DataFrame(
            {
                "date": axis.column([final.date]),
                "estimate": [final.estimate],
                "lower": [final.lower],
                "upper": [final.upper],
            }
        )
        base = _chart(frame).encode(x=x)
        tooltip = [
            axis.tooltip(),
            alt.Tooltip("estimate:Q", format=",.2f"),
            alt.Tooltip("lower:Q", format=",.2f"),
            alt.Tooltip("upper:Q", format=",.2f"),
        ]
        whisker = base.mark_rule(color=theme.cumulative, strokeWidth=2).encode(
            y=alt.Y("lower:Q", title=None), y2="upper:Q"
        )
        point = base.mark_point(color=theme.cumulative, filled=final.filled, opacity=1.0, size=80, strokeWidth=2)
        return [whisker, point.encode(y=alt.Y("estimate:Q", title=None), tooltip=tooltip)]

    # The difference panels read against zero, the no-effect line; drawing it also keeps zero in view.
    zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color=theme.zero, strokeDash=[2, 2])
    zero = zero.encode(y=alt.Y("y:Q", title=None))

    top: list[Any] = []
    middle: list[Any] = []
    bottom: list[Any] = []
    # Where a band excludes zero it is shaded more strongly.
    if data.prediction_band is not None:
        top.append(area(data.prediction_band, legend=data.ci_label))
    if data.difference_band is not None:
        middle.append(area(data.difference_band, color=theme.band(1.0)))
        middle += zero_free_area(data.difference_band, theme.band(1.0))
    if data.cumulative_band is not None:
        bottom.append(area(data.cumulative_band, color=theme.cumulative_band(1.0)))
        bottom += zero_free_area(data.cumulative_band, theme.cumulative_band(1.0))
    top.append(lines(data.dates, Actual=data.actual, Counterfactual=data.prediction))
    middle += [zero, lines(data.dates, Difference=data.difference)]
    bottom += [zero, lines(data.post_dates, **{"Cumulative Incrementality": data.cumulative})]
    if (final := data.final_interval) is not None:
        bottom += final_interval(final)

    start = _start_rule(data.post_period, axis, theme)
    titles = ("Expected vs Counterfactual", "Pointwise Difference", "Cumulative Difference")
    return alt.vconcat(
        *(
            alt.layer(*layers, start).properties(title=title, width=_WIDTH, height=180)
            for layers, title in zip((top, middle, bottom), titles, strict=True)
        )
    )


@_draw.register
def _(data: ParallelTrendsPlotData, theme: PlotTheme) -> Any:
    axis = _XAxis.of(data.treated_dates, data.control_dates)
    treated_x, control_x = axis.column(data.treated_dates), axis.column(data.control_dates)
    frame = pd.concat(
        [
            pd.DataFrame({"date": treated_x, "value": data.treated, "series": "Treated"}),
            pd.DataFrame({"date": control_x, "value": data.control, "series": "Control"}),
            pd.DataFrame({"date": control_x, "value": data.counterfactual, "series": "Counterfactual"}),
        ]
    )
    names = ["Treated", "Control", "Counterfactual"]
    lines = (
        _chart(frame)
        .mark_line()
        .encode(
            x=axis.encode(data.x_title),
            y=alt.Y("value:Q", title=data.y_title, scale=alt.Scale(zero=False)),
            color=_series_colour(names, [theme.actual, theme.control, theme.counterfactual]),
            # Dash only the counterfactual, as Plotly does.
            strokeDash=alt.StrokeDash(
                "series:N", scale=alt.Scale(domain=names, range=[[1, 0], [1, 0], [6, 4]]), legend=None
            ),
            tooltip=[axis.tooltip(), "series:N", alt.Tooltip("value:Q", format=",.2f")],
        )
    )
    return alt.layer(lines, _start_rule(data.post_period, axis, theme)).properties(
        title="Parallel Trends", width=_WIDTH, height=_HEIGHT
    )


@_draw.register
def _(data: EventStudyPlotData, theme: PlotTheme) -> Any:
    frame = pd.DataFrame(
        {
            "period": data.periods,
            "estimate": data.estimates,
            "lower": data.lower,
            "upper": data.upper,
            "filled": data.filled,
        }
    )
    x = alt.X("period:Q", title="Periods relative to treatment onset")
    base = _chart(frame).encode(x=x)
    tooltip = [
        "period:Q",
        alt.Tooltip("estimate:Q", format=",.2f"),
        alt.Tooltip("lower:Q", format=",.2f"),
        alt.Tooltip("upper:Q", format=",.2f"),
    ]
    y = alt.Y("estimate:Q", title=data.y_title)
    line = base.mark_line(color=theme.difference).encode(y=y)
    # Vega-Lite can't set ``filled`` per row, so filled and hollow markers are separate layers;
    # a marker is hollow when its interval contains zero.
    markers = [
        base.mark_point(color=theme.difference, filled=filled, opacity=1.0, strokeWidth=2)
        .encode(y=y, tooltip=tooltip)
        .transform_filter("datum.filled" if filled else "!datum.filled")
        for filled in (True, False)
    ]
    intervals = base.mark_rule(color=theme.difference).encode(y="lower:Q", y2="upper:Q")
    zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color=theme.zero, strokeDash=[2, 2]).encode(y="y:Q")
    onset = (
        alt.Chart(pd.DataFrame({"period": [-0.5]}))
        .mark_rule(color=theme.reference, strokeDash=[4, 4])
        .encode(x="period:Q")
    )
    return alt.layer(zero, onset, intervals, line, *markers).properties(
        title="Event Study (Two-Way Fixed Effects)", width=_WIDTH, height=_HEIGHT
    )


@_draw.register
def _(data: SwitchbackPlotData, theme: PlotTheme) -> Any:
    axis = _XAxis.of(data.frame["date"])
    frame = data.frame.assign(
        date=axis.column(data.frame["date"]).to_numpy(),
        geo=data.frame["geo"].astype(str),
        status=np.where(data.frame["treated"], "Treatment", "Control"),
    ).drop(columns="treated")
    base = _chart(frame).encode(
        x=axis.encode(data.x_title),
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
        tooltip=["geo:N", axis.tooltip(), "status:N", alt.Tooltip("y:Q", format=",.2f")],
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
