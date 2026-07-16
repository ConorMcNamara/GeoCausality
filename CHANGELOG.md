# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.13.0] - 2026-07-16

Adds the `ElasticNetSyntheticControl` estimator (elastic-net / intercept-shifted
synthetic control — the Doudchenko-Imbens synthesis).

### Added

- **`ElasticNetSyntheticControl`** (`elastic_net_synthetic_control`) — the synthesis
  estimator of Doudchenko & Imbens (2016), which unifies synthetic control,
  difference-in-differences and constrained regression as one penalized
  regression. The counterfactual is `mu + donor_matrix @ w`, fit on the pre-period
  by elastic net, relaxing classic synthetic control's three restrictions via
  explicit switches: an `intercept` (level shift), dropping the `sum_to_one` /
  `non_negative` weight constraints, and elastic-net regularisation (`l1_ratio`,
  `lambda_`, cross-validated by default). This MVP implements the unconstrained
  regime (the Doudchenko-Imbens default) on scikit-learn's elastic net; the
  sum-to-one constrained regime raises `NotImplementedError` pending a constrained
  solver. Because it produces donor weights, it supports the full shared inference
  stack — conformal, faithful jackknife+, and parametric bootstrap. Special cases
  reproduce classic SC, ridge-augmented SC, matched-market OLS, and DiD.

## [0.12.1] - 2026-07-15

### Changed

- Simplified the test suite by pruning redundant tests: **225 → 164 test
  functions** (~330 → 220 runtime cases, wall-clock roughly halved), with **no
  change to library source** and no loss of behavioral coverage. The cuts target
  pure redundancy — the shared conformal/jackknife/bootstrap inference machinery
  in `_base.py` is verified once rather than re-swept across every estimator;
  per-estimator parity assertions (avg-gap + significance) are merged into one
  check while the Prop 99, Germany, Card-Krueger and GeoLift benchmarks are all
  retained; and generic results-contract / plot-smoke checks already covered by
  the shared inference tests are removed. Estimator-specific behavior (factor
  selection, affine weights, ATT identities, structural-ts inference,
  market-selection ranking, power curves) is kept.

## [0.12.0] - 2026-07-15

Adds the `MatrixCompletion` estimator — a nuclear-norm matrix-completion
counterfactual (MC-NNM) — to the synthetic-control family.

### Added

- **`MatrixCompletion`** (`matrix_completion`) — the MC-NNM estimator of Athey,
  Bayati, Doudchenko, Imbens & Khosravi (2021). Rather than regressing the
  treated series on a weighted combination of donors, it stacks every unit into
  one panel matrix, masks the treated unit's post-period cells, and completes the
  matrix under a nuclear-norm penalty (two-way fixed effects plus a low-rank term,
  solved by soft-impute with a cross-validated penalty). The treated geos are
  aggregated into a single row, matching the estimand and `summarize` contract of
  the rest of the synthetic-control family. Reuses the shared conformal p-values
  and confidence intervals, with the residual-only jackknife+ fallback for short
  pre-periods; it has no donor-weight vector, so the weight-based faithful
  jackknife+ and parametric bootstrap do not apply. Pure numpy — no new
  dependency.
- Parity coverage for `MatrixCompletion` on the Proposition 99 (matches the
  published ~−20 packs closely) and West German reunification (right sign and
  order of magnitude, attenuated by the nuclear-norm penalty) benchmarks, plus a
  dedicated `test_matrix_completion` suite.

## [0.11.0] - 2026-07-14

Adds two nonlinear-outcome synthetic-control estimators and documents them across
the README and Sphinx docs.

### Added

- **`NonlinearSyntheticControl`** (`nonlinear_synthetic_control`) — Tian's (2023)
  synthetic control for strictly monotonic nonlinear outcomes. Because the link is
  monotonic, matching observed pre-period outcomes still matches the latent index,
  so the counterfactual stays a linear-in-weights combination of donor outcomes
  with no link function specified. Weights are **affine** (sum to one, may be
  negative), with a **distance-weighted L1** penalty `a` that concentrates weight
  on nearby donors and an **L2 ridge** `b` that spreads it out; both default to
  values chosen by rolling-origin, predict-to-horizon cross-validation. Shares the
  `pre_process() → generate() → summarize()/plot()` lifecycle and result contract
  with the synthetic-control family.
- **`KernelSyntheticControl`** (`kernel_synthetic_control`) — kernel-ridge
  synthetic control that learns a nonlinear *map* of the donor outcomes via a
  composite **linear + RBF** kernel (linear backbone for trend extrapolation, RBF
  term for local nonlinear flexibility; treated series centred so its level rides
  on an intercept). `bandwidth` defaults to the median-pairwise-distance heuristic
  and `lambda_` to the one-standard-error rule over a leave-one-out grid (both
  pinnable); `linear_weight` trades off the two kernels.
- **Parity tests** for both estimators against the Prop 99 (`test_prop99_parity`)
  and German reunification (`test_germany_parity`) benchmarks.
  `NonlinearSyntheticControl` reproduces the published magnitude (Prop 99 avg gap
  ~−20.2); `KernelSyntheticControl` is checked for sign and significance, since a
  nonlinear map attenuates toward the pre-period level on strongly trending panels.

### Fixed

- **`NonlinearSyntheticControl`** penalty cross-validation is now
  platform-reproducible. `_solve_nsc` reports SLSQP convergence, and
  `_select_penalties` restricts the grid search to *converged* `(a, b)` cells.
  Non-converged cells collapse toward uniform weights and their objective value
  is BLAS-backend dependent, so the previous raw argmin could pick a degenerate
  cell on some platforms (e.g. the German reunification panel returned ~−855 on
  Python 3.14/Linux versus ~−1,500 elsewhere); the selection now stays on the
  stable region across platforms.

### Documentation

- README: new Available Methods rows, Quick Start sections, plotting-family and
  validation-table entries for both estimators, plus the Tian (2023) reference.
- Sphinx docs: new `nonlinear_synthetic_control` and `kernel_synthetic_control`
  pages, wired into the estimators toctree and the API reference, with the Tian
  (2023) bibliography entry. Fixed the `NonlinearSyntheticControl` docstring math
  block (now a literal block) so the API docs build cleanly under `-W`.

## [0.10.1] - 2026-07-07

### Changed

- The `summarize` and `_get_roas` methods are now defined once on the shared
  `EconometricEstimator` base instead of being duplicated across the
  synthetic-control family (`SyntheticControl`, `SyntheticControlV`,
  `AugmentedSyntheticControl`, `PenalizedSyntheticControl`,
  `RobustSyntheticControl`, `GeneralizedSyntheticControl`,
  `InteractiveFixedEffects`, `SyntheticDiffInDiff`, `CausalImpact`). `DiffinDiff`,
  `FixedEffects` and `GeoX` keep their own bespoke summaries. Net −640 lines.
- `results["test"]` and `results["counterfactual"]` are now consistently numpy
  arrays of the post-period outcome across the whole synthetic-control family.
  Previously some estimators stored them as narwhals DataFrames and others as
  arrays; the shared `summarize` relies on the uniform contract. Code that read
  these keys as a DataFrame (e.g. `results["counterfactual"][y].sum()`) should use
  `np.sum(results["counterfactual"])`.
- The `summarize(lift="revenue")` and `summarize(lift="roas")` tables now format
  the Variant/Baseline columns as currency (dollar revenue and cost-per,
  respectively) for **all** synthetic-control estimators. Previously
  `SyntheticControl`, `SyntheticDiffInDiff` and `CausalImpact` showed raw unit
  counts in those two columns while the rest showed currency; the output is now
  uniform.

## [0.10.0] - 2026-07-07

Adds the `CausalImpact` estimator — a Bayesian structural time-series
counterfactual — and wires up the documentation bibliography.

### Added

- **`CausalImpact`** (`causal_impact`) — a Bayesian structural time-series
  counterfactual built on statsmodels' `UnobservedComponents` (time-varying
  level/trend, optional seasonality, and a regression on the donor geos), fit on
  the pre-period and forecast forward over the post-period. It shares the
  `pre_process() → generate() → summarize()/plot()` lifecycle and result contract
  with the synthetic-control family, so the shared three-panel counterfactual plot
  works unchanged.
- **Native posterior inference** for `CausalImpact` — draws counterfactual
  post-period paths from the fitted state posterior via `simulate()` (seeded,
  reproducible) and reports the percentile interval of the cumulative-effect
  distribution with a two-sided posterior tail-area p-value
  (`method == "structural-ts"`). The conformal / jackknife path remains available
  via `inference_method`.
- Validation tests against CausalImpact's canonical simulated example (AR(1)
  control, `y = 1.2·x1 + noise`, known post-period effect), plus a null-effect
  false-positive guard and seed-reproducibility checks.
- **Documentation bibliography** — enabled `sphinxcontrib-bibtex` with a
  `references.bib` covering all previously-dangling `:cite:` keys, a References
  page, and a `CausalImpact` estimator page.

### Changed

- Documented `CausalImpact` in the README methods and plotting tables, with a
  quick-start snippet.
- Docs build moved to `sphinxcontrib-bibtex>=2.6` (the prior `<2.0.0` pin could
  not import on Python 3.10+).

## [0.9.2] - 2026-07-07

### Changed

- CI test matrix is now derived from the `Programming Language :: Python :: X.Y`
  trove classifiers in `pyproject.toml` (via
  `hynek/build-and-inspect-python-package`) instead of a hardcoded list. Adding or
  dropping a supported Python version is now a one-line classifier edit.
- `lint`, `docs`, and the Codecov upload now target the lowest supported version,
  computed with a version-aware sort.

## [0.9.1] - 2026-07-06

### Added

- Confidence bands on `plot()` across the whole synthetic-control family
  (`SyntheticControl`, `SyntheticControlV`, `AugmentedSyntheticControl`,
  `PenalizedSyntheticControl`, `RobustSyntheticControl`,
  `GeneralizedSyntheticControl`, `SyntheticDiffInDiff`). The top panel shades the
  pointwise prediction band around the counterfactual, the middle panel shades it
  around zero (excursions are the significant per-period effects), and the
  cumulative panel's band grows to the reported incrementality interval, matching
  `summarize()`. Bands are derived from the inference already in `results` — no new
  computation.

### Changed

- Factored the duplicated three-panel counterfactual figure into a single shared
  `EconometricEstimator._plot_counterfactual` helper; each estimator's `plot()` is
  now a one-line delegate.

## [0.9.0] - 2026-07-06

### Added

- **`SyntheticDiffInDiff`** — synthetic difference-in-differences (Arkhangelsky,
  Athey, Hirshberg, Imbens & Wager, 2021). Fits L2-penalized non-negative unit
  weights against the treated trend (a unit fixed effect absorbs the level gap, so
  donors need only move parallel to the treated series) plus non-negative time
  weights, and reports the scalar doubly-weighted DID average treatment effect with
  placebo-variance inference (`results["standard_error"]`, `method == "placebo"`).
  Recovers the canonical benchmarks: Prop 99 avg gap −15.6, West German
  reunification −1473/year.
- West German reunification parity test (Abadie, Diamond & Hainmueller 2015) across
  the synthetic-control and interactive-fixed-effects family.

### Documentation

- Documented `InteractiveFixedEffects` in the README.

## [0.8.0] - 2026-07-01

Adds a standalone interactive fixed effects estimator, completing the panel-model
family alongside `FixedEffects` and the synthetic-control estimators.

### Added

- **`InteractiveFixedEffects`** — the Bai (2009) interactive fixed effects model,
  `Y_it = delta*D_it + alpha_i + xi_t + lambda_i'f_t + eps_it`, where a small
  number of latent time factors load onto each geo with a geo-specific weight. This
  relaxes the parallel-trends assumption of two-way fixed effects (to which it
  reduces when the factor count is zero) and models heterogeneous responses to
  common shocks. The additive two-way fixed effects and factors are estimated via
  the Bai alternating algorithm; the factor count is auto-selected by the
  eigenvalue-ratio criterion (Ahn & Horenstein, 2013). Two estimation modes via
  `method`: `"projection"` (default) estimates the fixed effects and factors from
  the control geos alone, then projects the treated geos' pre-period onto that time
  structure (Xu, 2017); `"coefficient"` is the full-panel Bai model where the
  treatment effect is a coefficient estimated jointly with the factors (opt-in,
  weakly identified with few treated geos).

### Validated

- Parity test against the Abadie Proposition 99 benchmark: projection mode recovers
  an average post-period gap of ~-26 packs/capita (published ~-19.5, within the
  parity band) with a single auto-selected factor. The Prop 99 loader is promoted
  to a shared test fixture.

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

[0.10.1]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.10.1
[0.10.0]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.10.0
[0.9.2]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.9.2
[0.9.1]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.9.1
[0.9.0]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.9.0
[0.8.0]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.8.0
[0.7.1]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.7.1
[0.7.0]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.7.0
[0.6.0]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.6.0
