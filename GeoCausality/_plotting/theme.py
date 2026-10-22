"""Light/dark colour themes shared by the Plotly and Altair renderers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlotTheme:
    """Plotly template plus the semantic colours each ``plot()`` draws with.

    Series colours are named by role rather than hue so dark mode can swap in
    lighter shades that keep their contrast against a dark background. The
    ``background``/``foreground``/``grid`` colours style Altair charts the way
    the Plotly template styles Plotly ones; ``None`` keeps Altair's defaults.
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
    background: str | None = None
    foreground: str | None = None
    grid: str | None = None

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
    # Mirrors the plotly_dark template so both backends look alike.
    background="#111111",
    foreground="#F2F5FA",
    grid="#283442",
)


def get_theme(dark_mode: bool) -> PlotTheme:
    """Return the dark theme when ``dark_mode`` is set, else the light theme."""
    return DARK_THEME if dark_mode else LIGHT_THEME
