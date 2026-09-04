# GeoCausality

[![Release](https://img.shields.io/github/v/release/ConorMcNamara/GeoCausality)](https://github.com/ConorMcNamara/GeoCausality/releases/latest)
[![Lint](https://github.com/ConorMcNamara/GeoCausality/actions/workflows/ci.yml/badge.svg)](https://github.com/ConorMcNamara/GeoCausality/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%20%7C%203.14-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![codecov](https://codecov.io/gh/ConorMcNamara/GeoCausality/branch/main/graph/badge.svg)](https://codecov.io/gh/ConorMcNamara/GeoCausality)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A Python library for measuring the causal impact of geo-level A/B experiments. GeoCausality provides a consistent, chainable API across a family of estimators — from simple difference-in-differences to interactive fixed effects and linear, nonlinear, kernel, matrix-completion, and elastic-net synthetic control.

## Estimators

| Class | Description | Reference |
|---|---|---|
| `GeoX` | Time-based regression matched markets (TBR) | [Au 2018](https://storage.googleapis.com/gweb-research2023-media/pubtools/5500.pdf), [Kerman et al. 2017](http://audentia-gestion.fr/Recherche-Research-Google/38355.pdf) |
| `DiffinDiff` | Difference-in-differences via OLS | [Card & Krueger 1994](https://www.nber.org/papers/w4509) |
| `FixedEffects` | Two-way fixed effects (entity + time) via PanelOLS | — |
| `InteractiveFixedEffects` | Interactive fixed effects / latent factor panel model | [Bai 2009](https://onlinelibrary.wiley.com/doi/10.3982/ECTA6135) |
| `SyntheticControl` | Classic synthetic control (constrained weights) | [Abadie et al. 2010](https://www.tandfonline.com/doi/abs/10.1198/jasa.2009.ap08746) |
| `SyntheticControlV` | Synthetic control with learned V matrix | [Abadie et al. 2010](https://www.tandfonline.com/doi/abs/10.1198/jasa.2009.ap08746) |
| `PenalizedSyntheticControl` | Penalized synthetic control | [Abadie & L'Hour 2021](https://www.tandfonline.com/doi/full/10.1080/01621459.2021.1971535) |
| `RobustSyntheticControl` | SVD-denoised synthetic control | [Amjad et al. 2018](https://www.jmlr.org/papers/v19/17-777.html) |
| `AugmentedSyntheticControl` | Augmented SC with ridge bias correction | [Ben-Michael et al. 2021](https://www.tandfonline.com/doi/full/10.1080/01621459.2021.1929245) |
| `ElasticNetSyntheticControl` | Elastic-net synthesis of SC, DiD & regression | [Doudchenko & Imbens 2016](https://www.nber.org/papers/w22791) |
| `GeneralizedSyntheticControl` | Interactive fixed effects via control-only latent factors | [Xu 2017](https://www.cambridge.org/core/journals/political-analysis/article/generalized-synthetic-control-method-causal-inference-with-interactive-fixed-effects-models/B63A8BD7C239DD4141C67DA10CD0E4F3) |
| `MatrixCompletion` | Nuclear-norm matrix completion / MC-NNM | [Athey et al. 2021](https://www.tandfonline.com/doi/full/10.1080/01621459.2021.1891924) |
| `NonlinearSyntheticControl` | Nonlinear-outcome synthetic control | [Tian 2023](https://arxiv.org/abs/2306.01967) |
| `KernelSyntheticControl` | Kernel-ridge nonlinear-map synthetic control (linear + RBF) | — |
| `SyntheticDiffInDiff` | Doubly-weighted difference-in-differences | [Arkhangelsky et al. 2021](https://www.aeaweb.org/articles?id=10.1257/aer.20190159) |
| `CausalImpact` | Bayesian structural time-series counterfactual | [Brodersen et al. 2015](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-9/issue-1/Inferring-causal-impact-using-Bayesian-structural-time-series-models/10.1214/14-AOAS788.full) |

### Pre-experiment design

| Class | Description | Reference |
|---|---|---|
| `PowerAnalysis` | Pre-experiment power / MDE via placebo simulation | — |
| `MarketSelection` | Rank candidate test-geo sets by power and pre-period fit | — |

### GeoLift wrapper

| Class | Description | Reference |
|---|---|---|
| `GeoLift` | Meta GeoLift pipeline: ASC point estimate + GSC parametric-bootstrap inference | [GeoLift docs](https://facebookincubator.github.io/GeoLift/) |

## Installation

```bash
pip install geocausality
```

Requires Python >= 3.13.

## Quick Start

All estimators share the same three-step chainable interface: `pre_process()` → `generate()` → `summarize()`.

```python
from GeoCausality import synthetic_control

model = synthetic_control.SyntheticControl(
    df,
    test_geos=["geo_A", "geo_B"],
    date_variable="date",
    pre_period="2022-06-30",
    post_period="2022-07-01",
    y_variable="orders",
    spend=500_000,
)
model.pre_process().generate().summarize(lift="roas")
model.plot()
```

Estimators that take a `treatment_variable` column instead of explicit geo lists work the same way:

```python
from GeoCausality import diff_in_diff

model = diff_in_diff.DiffinDiff(
    df,
    geo_variable="geo",
    treatment_variable="is_treatment",
    date_variable="date",
    pre_period="2022-06-30",
    post_period="2022-07-01",
    y_variable="orders",
)
model.pre_process().generate().summarize(lift="relative")
model.plot()
```

## Reference Options

### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `pd.DataFrame \| pl.DataFrame` | — | Geo-level time-series data |
| `geo_variable` | `str` | `"geo"` | Column identifying each geo unit |
| `test_geos` | `list[str] \| None` | `None` | Geos assigned to treatment |
| `control_geos` | `list[str] \| None` | `None` | Geos withheld from treatment |
| `treatment_variable` | `str \| None` | `"is_treatment"` | Binary treatment indicator column |
| `date_variable` | `str` | `"date"` | Date column |
| `pre_period` | `str` | `"2021-01-01"` | Last date of the pre-treatment window |
| `post_period` | `str` | `"2021-01-02"` | First date of the post-treatment window |
| `y_variable` | `str` | `"y"` | Outcome metric column |
| `alpha` | `float` | `0.1` | Significance level for confidence intervals |
| `msrp` | `float` | `0.0` | Average sale price (for revenue lift) |
| `spend` | `float` | `0.0` | Campaign spend (for ROAS / cost-per) |

### `lift`

| Value | Description |
|---|---|
| `"incremental"` | Total absolute lift over the post-period |
| `"absolute"` | Per-period absolute lift |
| `"relative"` | Percentage lift vs. counterfactual |
| `"revenue"` | Incremental revenue (requires `msrp`) |
| `"roas"` | Return on ad spend (requires `spend`) |
| `"cost-per"` | Cost per incremental unit (requires `spend`) |

### `inference_method`

`"conformal"`, `"jackknife"`, `"bootstrap"` — set before `generate()` to override automatic selection. Reported in `results["method"]`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
