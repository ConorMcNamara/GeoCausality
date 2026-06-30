# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Replaced the internal `assert` statements in the library (the `GeoCausality`
  package) with explicit `raise ValueError(...)`, so these invariant checks are
  preserved when Python runs with assertions disabled (`-O`). Test-suite
  assertions are unchanged.

### Added

- **Card & Krueger (1994) parity test** (`test/test_card_krueger_parity.py`) —
  literature-validation ("golden master") tests for `FixedEffects` and
  `DiffinDiff` against the canonical NJ/PA minimum-wage difference-in-differences
  result (+2.76 FTE). `FixedEffects` (two-way fixed effects) is checked on the
  full store-level panel; `DiffinDiff` on the per-state mean series. The authors'
  public `public.dat` is vendored to `test/data/card_krueger_1994.csv` (reshaped
  by `test/data/vendor_card_krueger.py`, provenance in
  `card_krueger_1994.README.txt`); the test skips cleanly when the CSV is absent.
- **Proposition 99 parity test** (`test/test_prop99_parity.py`) —
  literature-validation ("golden master") tests for the synthetic-control family
  against Abadie, Diamond & Hainmueller's (2010) California Prop 99 result
  (~-19.5 average / ~-26 year-2000 packs/capita). All four estimators now match
  within tolerance (`SyntheticControl` -19.5, `AugmentedSyntheticControl` -15.8,
  `PenalizedSyntheticControl` -23.5, `GeneralizedSyntheticControl` -20.7) after
  the two divergences the test originally surfaced (#31, #32) were fixed. The
  `Synth` `smoking` dataset is vendored to `test/data/prop99_smoking.csv` (via
  `test/data/vendor_prop99.py`, provenance in `prop99_smoking.README.txt`); the
  test skips cleanly when the CSV is absent.

### Fixed

- **`PenalizedSyntheticControl`** (#31) — now fits its donor weights against the
  full pre-period **trajectory** (as `SyntheticControl` does) instead of a single
  pre-period mean per geo, and the Abadie & L'Hour discrepancy penalty is scaled
  per-period so `lambda_` trades off fit and penalty in comparable units
  (`lambda_ -> 0` recovers the unpenalized estimator). On Prop 99 the average
  post-period gap moves from -33.3 to -23.5 and the pre-period fit RMSE from
  ~9.0 to ~3.7.
- **`GeneralizedSyntheticControl`** (#32) — the latent-factor count is now chosen
  by the eigenvalue-ratio criterion (Ahn & Horenstein, 2013) on the control
  panel's spectrum by default, instead of a treated-pre-period cross-validation
  that over-selected factors and overfit the counterfactual (washing out the
  effect). On Prop 99 the average post-period gap moves from -3.6 (not
  significant) to -20.7 (significant). The previous cross-validation remains
  available opt-in via the new ``factor_selection="cv"`` argument
  (``factor_selection="er"`` is the default).

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
