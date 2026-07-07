"""Testing Statistical Hypotheses through Geo Experiments."""

__version__ = "0.10.0"


from GeoCausality import (
    augmented_synthetic_control,
    causal_impact,
    diff_in_diff,
    fixed_effects,
    generalized_synthetic_control,
    geolift,
    geox,
    interactive_fixed_effects,
    market_selection,
    penalized_synthetic_control,
    power,
    robust_synthetic_control,
    synthetic_control,
    synthetic_diff_in_diff,
)

__all__: list[str] = [
    "diff_in_diff",
    "fixed_effects",
    "interactive_fixed_effects",
    "geox",
    "synthetic_control",
    "synthetic_diff_in_diff",
    "penalized_synthetic_control",
    "robust_synthetic_control",
    "augmented_synthetic_control",
    "causal_impact",
    "generalized_synthetic_control",
    "geolift",
    "power",
    "market_selection",
]


def __dir__() -> list[str]:
    return __all__
