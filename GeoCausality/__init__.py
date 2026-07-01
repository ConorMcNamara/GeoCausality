"""Testing Statistical Hypotheses through Geo Experiments."""

__version__ = "0.7.0"


from GeoCausality import (
    augmented_synthetic_control,
    diff_in_diff,
    fixed_effects,
    generalized_synthetic_control,
    geolift,
    geox,
    market_selection,
    penalized_synthetic_control,
    power,
    robust_synthetic_control,
    synthetic_control,
)

__all__: list[str] = [
    "diff_in_diff",
    "fixed_effects",
    "geox",
    "synthetic_control",
    "penalized_synthetic_control",
    "robust_synthetic_control",
    "augmented_synthetic_control",
    "generalized_synthetic_control",
    "geolift",
    "power",
    "market_selection",
]


def __dir__() -> list[str]:
    return __all__
