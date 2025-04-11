"""Testing Statistical Hypotheses through Geo Experiments"""

__version__ = "0.5.0"

from typing import List

from GeoCausality import diff_in_diff, fixed_effects, geox, synthetic_control, penalized_synthetic_control, robust_synthetic_control, augmented_synthetic_control

__all__: List[str] = [
    "diff_in_diff",
    "fixed_effects",
    "geox",
    "synthetic_control",
    "penalized_synthetic_control",
    "robust_synthetic_control",
    "augmented_synthetic_control",
]


def __dir__() -> List[str]:
    return __all__
