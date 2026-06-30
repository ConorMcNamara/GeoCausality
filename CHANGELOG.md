# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Card & Krueger (1994) parity test** (`test/test_card_krueger_parity.py`) —
  literature-validation ("golden master") tests for `FixedEffects` and
  `DiffinDiff` against the canonical NJ/PA minimum-wage difference-in-differences
  result (+2.76 FTE). `FixedEffects` (two-way fixed effects) is checked on the
  full store-level panel; `DiffinDiff` on the per-state mean series. The authors'
  public `public.dat` is vendored to `test/data/card_krueger_1994.csv` (reshaped
  by `test/data/vendor_card_krueger.py`, provenance in
  `card_krueger_1994.README.txt`); the test skips cleanly when the CSV is absent.

## [0.6.0] - 2026-06-29

A GeoLift-parity release: GeoCausality now mirrors Meta's GeoLift workflow
end-to-end — pre-experiment design, the augmented/generalized synthetic-control
estimators, and GeoLift-style inference — validated against Meta's published
`GeoLift_Test` example (within ~1 percentage point).

### Added

- **`GeoLift`** (`geolift`) — a unified, GeoLift-style entry point that uses
  Augmented Synthetic Control for the de-biased point estimate and Generalized
  Synthetic Control with parametric-bootstrap inference for the uncertainty.
- **`GeneralizedSyntheticControl`** (`generalized_synthetic_control`) — the
  interactive fixed effects / latent factor method of Xu (2017), with
  cross-validated factor selection.
- **`PowerAnalysis`** (`power`) — pre-experiment power / Minimum Detectable
  Effect via placebo simulation (GeoLift `GeoLiftPower` analog).
- **`MarketSelection`** (`market_selection`) — ranks candidate test-geo sets by
  power and pre-period fit (GeoLift `GeoLiftMarketSelection` analog).
- **Jackknife+ inference** — a faithful, refit-based jackknife+ (Barber et al.,
  2021) fallback for short pre-periods, implemented across the whole
  synthetic-control family via a shared leave-one-out loop.
- **Parametric-bootstrap inference** — `inference_method = "bootstrap"`
  (GeoLift's GSC-style approach), supported across the whole synthetic-control
  family; configurable `n_boot` / `bootstrap_seed`.
- **GeoLift parity test** — validates `GeoLift` against the vendored
  `GeoLift_Test` dataset and Meta's published results.
- `results["method"]` now records which inference path produced the interval
  (`"conformal"`, `"jackknife+"`, `"jackknife+ (residual)"`, or `"bootstrap"`),
  selectable via `inference_method`.

### Changed

- **`AugmentedSyntheticControl`** rewritten as the intercept-augmented,
  ridge-regression ASCM of Ben-Michael et al. with a one-standard-error lambda
  rule, matching R `augsynth`. This fixes a level bias for aggregated treated
  units (the counterfactual could not reach the level of a summed treated unit
  outside the donor convex hull) that inflated incrementality, and brings the
  GeoLift point estimate in line with Meta's published numbers.

## [0.5.0]

Initial set of estimators sharing the chainable
`pre_process() → generate() → summarize()` interface: `GeoX`, `DiffinDiff`,
`FixedEffects`, `SyntheticControl`, `SyntheticControlV`,
`PenalizedSyntheticControl`, `RobustSyntheticControl`, and
`AugmentedSyntheticControl`, with distribution-free conformal inference.

[0.6.0]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.6.0
