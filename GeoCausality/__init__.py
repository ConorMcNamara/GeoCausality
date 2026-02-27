"""Testing Statistical Hypotheses through Geo Experiments"""

__version__ = "0.5.0"


from GeoCausality import (
    augmented_synthetic_control,
    diff_in_diff,
    fixed_effects,
    geox,
    penalized_synthetic_control,
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
]


def __dir__() -> list[str]:
    return __all__
