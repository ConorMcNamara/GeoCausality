# GeoCausality

[![Lint](https://github.com/ConorMcNamara/GeoCausality/actions/workflows/ci.yml/badge.svg)](https://github.com/ConorMcNamara/GeoCausality/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%20%7C%203.14-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![codecov](https://codecov.io/gh/ConorMcNamara/GeoCausality/branch/main/graph/badge.svg)](https://codecov.io/gh/ConorMcNamara/GeoCausality)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A Python library for measuring the causal impact of geo-level A/B experiments. GeoCausality provides a consistent, chainable API across seven estimators — from simple difference-in-differences to augmented synthetic control.

---

## Table of Contents

- [Installation](#installation)
- [Available Methods](#available-methods)
- [Quick Start](#quick-start)
- [API Overview](#api-overview)
- [Contributing](#contributing)
- [References](#references)

---

## Installation

```bash
pip install geocausality
```

**Requirements:** Python ≥ 3.13

---

## Available Methods

| Class | Module | Description |
|---|---|---|
| `GeoX` | `geox` | Time-based regression matched markets (TBR) |
| `DiffinDiff` | `diff_in_diff` | Difference-in-differences via OLS |
| `FixedEffects` | `fixed_effects` | Two-way fixed effects (entity + time) via PanelOLS |
| `SyntheticControl` | `synthetic_control` | Classic synthetic control (constrained weights) |
| `SyntheticControlV` | `synthetic_control` | Synthetic control with learned V matrix |
| `PenalizedSyntheticControl` | `penalized_synthetic_control` | Synthetic control with pairwise penalty (Abadie & L'Hour) |
| `RobustSyntheticControl` | `robust_synthetic_control` | SVD-denoised synthetic control (Amjad, Shah & Shen) |
| `AugmentedSyntheticControl` | `augmented_synthetic_control` | Augmented SC with ridge bias correction (Ben-Michael et al.) |
| `GeneralizedSyntheticControl` | `generalized_synthetic_control` | Interactive fixed effects / latent factor model (Xu) |

### Pre-experiment design

| Class | Module | Description |
|---|---|---|
| `PowerAnalysis` | `power` | Pre-experiment power / Minimum Detectable Effect via placebo simulation (GeoLift `GeoLiftPower` analog) |
| `MarketSelection` | `market_selection` | Rank candidate test-geo sets by power and pre-period fit (GeoLift `GeoLiftMarketSelection` analog) |

```python
from GeoCausality import power
from GeoCausality.augmented_synthetic_control import AugmentedSyntheticControl

pa = power.PowerAnalysis(
    df,
    geo_variable="geo",
    treatment_variable="is_treatment",
    date_variable="date",
    pre_period="2022-06-30",   # last date of clean history
    y_variable="orders",
    estimator=AugmentedSyntheticControl,
)
pa.simulate(effect_sizes=[0.0, 0.05, 0.10, 0.15], durations=[14, 28], n_sims=200).mde(target_power=0.8)
pa.summarize()   # power curve + MDE table
pa.plot()        # power-vs-effect curve, one line per duration
```

Don't know which geos to treat yet? `MarketSelection` searches candidate test
sets and ranks them, reusing `PowerAnalysis` as its scoring engine:

```python
from GeoCausality import market_selection

ms = market_selection.MarketSelection(
    df,
    geo_variable="geo",
    date_variable="date",
    pre_period="2022-06-30",
    y_variable="orders",
    estimator=AugmentedSyntheticControl,
)
ms.search(n_test_geos=[1, 2, 3], effect_size=0.10, duration=28, n_sims=200)
ms.summarize()   # ranked test markets: power, pre-fit, score
ms.plot()        # top candidates by score
```

---

## Quick Start

All estimators share the same three-step chainable interface: `pre_process()` → `generate()` → `summarize()`.

### GeoLift (one-call, GeoLift-style)

`GeoLift` mirrors Meta's GeoLift pipeline: it uses **Augmented Synthetic Control**
for the de-biased point estimate and **Generalized Synthetic Control** with
**parametric-bootstrap** inference for the uncertainty, behind one call.

```python
from GeoCausality import geolift

model = geolift.GeoLift(
    df,
    geo_variable="geo",
    treatment_variable="is_treatment",
    date_variable="date",
    pre_period="2022-06-30",
    post_period="2022-07-01",
    y_variable="orders",
    spend=500_000,
)
model.pre_process().generate().summarize(lift="incremental")
model.plot()
# model.results["incrementality"]  -> ASC point estimate
# model.results["p_value"], ["incrementality_ci_lower/upper"]  -> GSC bootstrap inference
```

### GeoX (Time-Based Regression)

```python
import pandas as pd
from GeoCausality import geox

df = pd.read_csv("geo_data.csv", parse_dates=["date"])

model = geox.GeoX(
    df,
    geo_variable="geo",
    treatment_variable="is_treatment",
    date_variable="date",
    pre_period="2022-06-30",
    post_period="2022-07-01",
    y_variable="orders",
    spend=500_000,
)
model.pre_process().generate().summarize(lift="incremental")
```

### Difference-in-Differences

```python
from GeoCausality import diff_in_diff

model = diff_in_diff.DiffinDiff(
    df,
    geo_variable="geo",
    date_variable="date",
    pre_period="2022-06-30",
    post_period="2022-07-01",
    y_variable="orders",
)
model.pre_process().generate().summarize(lift="relative")
```

### Synthetic Control

```python
from GeoCausality import synthetic_control

model = synthetic_control.SyntheticControl(
    df,
    test_geos=["geo_A", "geo_B"],
    date_variable="date",
    pre_period="2022-06-30",
    post_period="2022-07-01",
    y_variable="orders",
)
model.pre_process().generate().summarize(lift="roas")
```

### `summarize` lift options

| Value | Description |
|---|---|
| `"incremental"` | Total absolute lift over the post-period |
| `"absolute"` | Per-period absolute lift |
| `"relative"` | Percentage lift vs. counterfactual |
| `"revenue"` | Incremental revenue (requires `msrp`) |
| `"roas"` | Return on ad spend (requires `spend`) |
| `"cost-per"` | Cost per incremental unit (requires `spend`) |

### Inference on short pre-periods

Synthetic-control estimators report a distribution-free p-value and confidence
intervals via the Chernozhukov–Wüthrich–Zhu moving-block permutation test and a
split-conformal band. When the pre-treatment period is too short for those to be
reliable (the band quantile saturates), inference automatically falls back to
**jackknife+** (Barber et al., 2021), which reuses every pre-period point for
both fitting and calibration. Every synthetic-control estimator refits its own
weights leave-one-out for a faithful jackknife+ (`"jackknife+"`); estimators
without that refit fall back to a residual-only approximation
(`"jackknife+ (residual)"`). Setting `model.inference_method = "bootstrap"`
instead performs inference by **parametric bootstrap** (GeoLift's GSC-style
approach), supported across the whole synthetic-control family; `n_boot` and
`bootstrap_seed` are configurable. The method used is reported in
`results["method"]` (`"conformal"`, `"jackknife+"`, `"jackknife+ (residual)"`, or
`"bootstrap"`), and can be forced with
`model.inference_method = "conformal" | "jackknife" | "bootstrap"` before
`generate()`.

---

## API Overview

Every estimator accepts the same core constructor arguments:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `pd.DataFrame \| pl.DataFrame` | — | Geo-level time-series data |
| `geo_variable` | `str` | `"geo"` | Column identifying each geo unit |
| `test_geos` | `list[str] \| None` | `None` | Geos assigned to treatment |
| `control_geos` | `list[str] \| None` | `None` | Geos withheld from treatment |
| `treatment_variable` | `str \| None` | `"is_treatment"` | Binary treatment indicator column (used when geo lists are not provided) |
| `date_variable` | `str` | `"date"` | Date column |
| `pre_period` | `str` | `"2021-01-01"` | Last date of the pre-treatment window |
| `post_period` | `str` | `"2021-01-02"` | First date of the post-treatment window |
| `y_variable` | `str` | `"y"` | Outcome metric column |
| `alpha` | `float` | `0.1` | Significance level for confidence intervals |
| `msrp` | `float` | `0.0` | Average sale price (for revenue lift) |
| `spend` | `float` | `0.0` | Campaign spend (for ROAS / cost-per) |

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on setting up the development environment, running tests, and submitting pull requests.

---

## References

- Au, Tim. "A Time-Based Regression Matched Markets Approach for Designing Geo Experiments." (2018). [PDF](https://storage.googleapis.com/gweb-research2023-media/pubtools/5500.pdf)
- Kerman, Jouni, Peng Wang, and Jon Vaver. "Estimating ad effectiveness using geo experiments in a time-based regression framework." Google, 2017. [PDF](http://audentia-gestion.fr/Recherche-Research-Google/38355.pdf)
- Brodersen, Kay H., et al. "Inferring causal impact using Bayesian structural time-series models." *Annals of Applied Statistics* 9.1 (2015): 247–274. [Link](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-9/issue-1/Inferring-causal-impact-using-Bayesian-structural-time-series-models/10.1214/14-AOAS788.full)
- Abadie, Alberto, and Jérémy L'Hour. "A penalized synthetic control estimator for disaggregated data." *Journal of the American Statistical Association* (2021). [Link](https://www.tandfonline.com/doi/full/10.1080/01621459.2021.1971535)
- Ben-Michael, Eli, Avi Feller, and Jesse Rothstein. "The augmented synthetic control method." *Journal of the American Statistical Association* (2021). [Link](https://www.tandfonline.com/doi/full/10.1080/01621459.2021.1929245)
- Amjad, Mohammad, Devavrat Shah, and Dennis Shen. "Robust synthetic control." *Journal of Machine Learning Research* 19.1 (2018): 802–852. [Link](https://www.jmlr.org/papers/v19/17-777.html)
- Barber, Rina Foygel, Emmanuel J. Candès, Aaditya Ramdas, and Ryan J. Tibshirani. "Predictive inference with the jackknife+." *Annals of Statistics* 49.1 (2021): 486–507. [Link](https://projecteuclid.org/journals/annals-of-statistics/volume-49/issue-1/Predictive-inference-with-the-jackknife/10.1214/20-AOS1965.full)
- Xu, Yiqing. "Generalized Synthetic Control Method: Causal Inference with Interactive Fixed Effects Models." *Political Analysis* 25.1 (2017): 57–76. [Link](https://www.cambridge.org/core/journals/political-analysis/article/generalized-synthetic-control-method-causal-inference-with-interactive-fixed-effects-models/B63A8BD7C239DD4141C67DA10CD0E4F3)
- GeoLift (Meta). [Documentation](https://facebookincubator.github.io/GeoLift/)
