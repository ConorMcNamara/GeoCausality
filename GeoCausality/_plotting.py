"""Shared light/dark colour themes for the estimators' Plotly figures."""

from dataclasses import dataclass

import plotly.graph_objects as go


@dataclass(frozen=True)
class PlotTheme:
    """Plotly template plus the semantic colours each ``plot()`` draws with.

    Series colours are named by role rather than hue so dark mode can swap in
    lighter shades that keep their contrast against a dark background.
    """

    template: str | None
    actual: str
    counterfactual: str
    control: str
    difference: str
    cumulative: str
    reference: str
    zero: str
    band_rgb: str
    cumulative_band_rgb: str

    def band(self, alpha: float) -> str:
        """Return the neutral confidence-band fill at the given opacity."""
        return f"rgba({self.band_rgb}, {alpha})"

    def cumulative_band(self, alpha: float) -> str:
        """Return the cumulative-difference band fill at the given opacity."""
        return f"rgba({self.cumulative_band_rgb}, {alpha})"


LIGHT_THEME = PlotTheme(
    template=None,
    actual="blue",
    counterfactual="red",
    control="green",
    difference="purple",
    cumulative="orange",
    reference="black",
    zero="gray",
    band_rgb="68, 68, 68",
    cumulative_band_rgb="255, 165, 0",
)

DARK_THEME = PlotTheme(
    template="plotly_dark",
    actual="#5DA9FF",
    counterfactual="#FF6B6B",
    control="#5CD65C",
    difference="#C792EA",
    cumulative="#FFB347",
    reference="#E0E0E0",
    zero="#9E9E9E",
    band_rgb="200, 200, 200",
    cumulative_band_rgb="255, 179, 71",
)


def get_theme(dark_mode: bool) -> PlotTheme:
    """Return the dark theme when ``dark_mode`` is set, else the light theme."""
    return DARK_THEME if dark_mode else LIGHT_THEME


def apply_theme(fig: go.Figure, theme: PlotTheme) -> go.Figure:
    """Apply the theme's Plotly template to ``fig`` and return it.

    The light theme has no template, so it leaves the caller's
    ``plotly.io.templates.default`` in effect.
    """
    if theme.template is not None:
        fig.update_layout(template=theme.template)
    return fig
