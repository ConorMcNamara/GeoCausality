# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.1] - 2026-07-01

A plotting release: the two econometric estimators gain the visual diagnostics
their synthetic-control and GeoX counterparts already had.

### Added

- **`DiffinDiff.plot()`** — a parallel-trends diagnostic plotting the treated and
  control group averages over time alongside the parallel-trends counterfactual
  for the treated group (the control series level-shifted by the pre-period gap
  between the groups). The post-period gap between the treated series and the
  counterfactual equals the fitted difference-in-differences estimand, and the
  pre-period overlay lets the parallel-trends assumption be checked visually.
- **`FixedEffects.plot()`** — an event-study plot of the dynamic treatment effect.
  It fits an auxiliary two-way fixed-effects model interacting the treated
  indicator with each period relative to treatment onset (the period just before
  onset is the omitted reference) and plots the coefficient path with confidence
  intervals: pre-onset coefficients near zero support parallel trends, post-onset
  coefficients trace the dynamic effect. The auxiliary model is fit independently
  of `generate()` (and uses entity-clustered standard errors, appropriate for the
  single-cohort event-study design), leaving the headline single-coefficient
  results unchanged.

## [0.7.0] - 2026-06-30

A correctness and validation release. Literature-validation ("golden master")
tests against the canonical Card & Krueger (1994) and Abadie, Diamond &
Hainmueller (2010) Proposition 99 datasets now guard the econometric and
synthetic-control estimators against their foundational papers' published
results. Building these tests surfaced — and this release fixes — four
reimplementation bugs across the synthetic-control family; all six SC-family
estimators now reproduce the Prop 99 effect within tolerance.

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
  (~-19.5 average / ~-26 year-2000 packs/capita). All six SC-family estimators
  now match within tolerance (`SyntheticControl` -19.5, `SyntheticControlV`
  -19.5, `AugmentedSyntheticControl` -15.8, `PenalizedSyntheticControl` -23.5,
  `GeneralizedSyntheticControl` -20.7, `RobustSyntheticControl` -17.4). The
  `Synth` `smoking` dataset is vendored to `test/data/prop99_smoking.csv` (via
  `test/data/vendor_prop99.py`, provenance in `prop99_smoking.README.txt`); the
  test skips cleanly when the CSV is absent.
- **`GeneralizedSyntheticControl`** gains a `factor_selection` argument (`"er"`,
  the eigenvalue-ratio default; `"cv"` for the previous cross-validation).
- **`RobustSyntheticControl`** gains an `sv_energy` argument (default 0.999) that
  selects the retained rank when neither `threshold` nor `sv_count` is given.

### Changed

- Replaced the internal `assert` statements in the library (the `GeoCausality`
  package) with explicit `raise ValueError(...)`, so these invariant checks are
  preserved when Python runs with assertions disabled (`-O`). Test-suite
  assertions are unchanged.

### Fixed

- **`SyntheticControlV`** — corrected the Abadie & Gardeazabal implementation,
  which diverged from the Prop 99 benchmark (average post-period gap −30.5 vs the
  published ~−19.5). It was matching only a single pre-period **mean** per geo (so
  the V matrix collapsed to a degenerate 1×1 and the outer V optimization was a
  no-op) and the simplex sum-to-one constraint on the donor weights was commented
  out. It now matches the full pre-period **trajectory** as the predictor set (V
  is `n_pre × n_pre`, as in pysyncon's `Synth`) with the simplex constraint
  restored, recovering −19.5 / −26.6 on Prop 99 with weights that sum to one and a
  pre-period RMSE of 1.66 (was 7.26).
- **`RobustSyntheticControl`** — is now usable out of the box: previously it
  raised unless one of `threshold` / `sv_count` was set, and its pre/post split
  parsed the split date with `date.fromisoformat`, which broke on non-ISO date
  columns (e.g. integer years). It now falls back to retaining a configurable
  fraction of the donor matrix's spectral energy (`sv_energy`, default 0.999)
  when no rank is given, and derives the split by backend-agnostic string
  comparison (with `daily_x` sorted by date).
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
  significant) to -20.7 (significant).
- Resolved the `zuban` type-check errors in `SyntheticControl.summarize` and
  `GeoX.summarize` (the `table_dict` was inferred as `list[float64]` from its
  numeric `np.sum(...)` entries, rejecting the later string-list assignments).
  Annotating it `dict[str, list[Any]]` clears all 40 errors, turning the CI
  type-check step green. No runtime change.

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

[0.7.1]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.7.1
[0.7.0]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.7.0
[0.6.0]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.6.0
