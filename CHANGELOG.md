# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.21.1] - 2026-10-28

### Added

- **Zero cues on the counterfactual figure** — the pointwise- and
  cumulative-difference panels show at a glance where their interval excludes
  zero, in both backends:
  - A dotted zero line, which also keeps zero on the y-axis. The Altair panels
    previously fit the axis to the data, so zero could fall off-screen.
  - The band is shaded more strongly over each run of dates where it lies
    wholly above or below zero. As with the event-study markers, an interval
    that ends exactly at zero counts as containing it.
  - The cumulative panel ends with a forest-style point and interval for the
    final cumulative effect: filled when the interval excludes zero, hollow
    when it contains it.

  These are pointwise intervals, so a darker stretch is not a simultaneous
  test across those dates.

### Changed

- **Pointwise-difference band centred on the difference** — the shared
  counterfactual figure (the synthetic-control family, `CausalImpact`,
  `SyntheticDiffInDiff`, `GeoLift`) now shades
  `difference ± conformal_band` on its middle panel instead of
  `± conformal_band` around zero. Since `counterfactual ± conformal_band` is
  a pointwise prediction interval for the untreated outcome, this is the
  per-period effect interval. It excludes zero exactly where the line used to
  leave the zero-centred band, so the significant periods are unchanged; the
  band now reads against the zero line and darkens where it clears it, as
  GeoX's already did. Darker pre-period stretches flag counterfactual misfit.

- **`SyntheticDiffInDiff` pointwise band** — `results["conformal_band"]` is
  now a split-conformal band calibrated on cross-fitted pre-period residuals,
  as for the conformal estimators, instead of `z × SE` of the placebo average
  effect. The placebo SE is for the average effect, so as a per-period band it
  was far too narrow: on simulated no-effect panels 55% of post-period points
  fell outside a 90% band, against 8% now. This widens the shaded bands in
  `plot()` and changes how `MarketSelection` ranks candidates for
  `SyntheticDiffInDiff`, which now scores the pre-period fit as its docstring
  describes. The ATT, its CIs, the p-value and `standard_error` are unchanged.
  As for the conformal estimators, a pre-period too short for a finite band
  gives an infinite one; the placebo band is kept only when there are fewer
  than two pre-period dates to cross-fit.

- **`CausalImpact` per-period plot intervals** — with the structural
  inference, `plot()` now draws a separate interval for each period instead
  of one representative half-width (`z × mean` per-period SD of the draws)
  everywhere:
  - Post-period: percentile intervals of the simulated counterfactual paths,
    so the band widens with the forecast horizon and follows seasonality.
  - Pre-period: the model's one-step-ahead prediction intervals. The first
    periods, whose intervals reflect only the diffuse initial state, get no
    band.
  - Cumulative panel: percentile intervals of the cumulative-effect draws at
    each post-period, replacing the band that grew linearly to the reported
    interval. Its endpoint is still the reported incrementality interval.

  The difference band is `actual` minus the counterfactual interval. Results,
  including the scalar `conformal_band` (which `MarketSelection` still reads),
  are unchanged, and the conformal route keeps the scalar band. Shared
  estimators can supply per-period intervals through the new
  `_counterfactual_interval` / `_cumulative_effect_interval` hooks.

- **License: Apache-2.0** — GeoCausality is now licensed under the Apache
  License 2.0 instead of MIT. A new `NOTICE` file carries the copyright
  notice and keeps the MIT notice of pysyncon, on which parts of
  `SyntheticControl`, `AugmentedSyntheticControl` and
  `RobustSyntheticControl` are based. Both files ship in the sdist and wheel.

## [0.21.0] - 2026-10-22

### Added

- **`lift_ci_set` / `incrementality_ci_set`** — the conformal-family results
  now include the confidence set as a list of `(lower, upper)` intervals. It
  is one interval in almost every case; see below for when it is not.
  `FixedEffects` (conformal inference) reports it too, with
  `incrementality_ci_set` on its `n_dates * n_geos` scale, and `GeoLift`
  reports its single recentred bootstrap interval.

### Changed

- **Faster `ElasticNetSyntheticControl` inference** — the elastic-net refits
  behind the conformal test, its confidence interval and the cross-fitted band
  (about 50–90 per `generate()`) run coordinate descent on the precomputed
  Gram matrix whenever the donors number at most 2.5 times the training
  dates. The penalty was already selected once and held fixed in every refit;
  the time went into coordinate descent itself, which hits its 100,000-sweep
  cap for the large candidate effects the interval search tries. The Gram
  solver reaches the same solution more cheaply, so results are unchanged up
  to floating-point rounding (about 1e-10 relative), though the number of
  sweeps can differ on those badly conditioned large-effect refits. With 100 donors and 365
  days, `generate()` drops from 171 s to 16 s; with 400 donors and 365 days,
  from 700 s to 312 s. Panels with far more donors than dates (400 donors,
  120 days) keep the plain solver and are not faster.
- **Faster `MatrixCompletion`** — `generate()` is 40-200x faster (100 donors
  x 120 days: ~53 s to ~1.3 s; 400 x 365: over 5 min to ~20 s). Every fit is
  now warm-started along the decreasing penalty path, including the final
  fit at the chosen penalty, instead of starting from zero. The
  singular-value soft-threshold eigendecomposes the smaller Gram matrix
  (matching the full SVD to ~1e-9 relative or better), and the fixed-effect
  sweeps use matrix-vector products. Soft-impute converges slowly from zero
  at small penalties, so the old cold-start fits often stopped short of the
  optimum; the new fits sit closer to the converged solution. Results
  therefore shift slightly: within 0.2% on most panels, while a few panels
  with a flat cross-validation curve now pick a different penalty. On the
  West Germany parity panel the average gap moves from -755 to -826
  (converged: -860). The final fit's continuation path is fixed, so a
  supplied `lambda_` gives the same fit whatever `n_lambda` and
  `lambda_ratio` are.
- **Faster DTW pre-filter in `MarketSelection`** — `search(dtw_top_k=...)` no
  longer runs a pure-Python double loop per geo pair. The DTW recurrence now
  runs one anti-diagonal at a time across thousands of pairs at once in numpy,
  pairs between geos that never enter a test set are skipped, and per-geo
  series are grouped in one pass instead of one filter per geo. Distances are
  bit-for-bit identical, so filtered candidates and rankings are unchanged.
  All pairwise distances for 200 geos over 365 dates take 2.4s instead of
  16 min; 1,000 geos take about 1 min instead of an estimated 6 h. A new
  opt-in `dtw_window` argument applies a Sakoe-Chiba band, which cuts the
  cost to about `(2 * dtw_window + 1) / T` of that but can change which
  candidates pass the filter; the default `None` keeps unconstrained DTW.
- **Faster switchback permutation test** — `Switchback.permutation_test`
  shifts every geo's schedule in one vectorised step (same random draws, so
  the same permuted schedules) and gets each permutation's treatment
  coefficient from a closed-form two-way fixed-effects fit instead of a full
  `PanelOLS` fit with a clustered covariance that was thrown away. Effects are
  swept out exactly, as `PanelOLS` does, so coefficients match the old
  implementation to ~1e-13. Permuted coefficients are compared with the
  observed one with a 1e-9 relative tolerance, so shifts that reproduce or
  flip a periodic schedule still count as ties (extreme) and p-values match
  the old implementation. Permutations whose treatment is absorbed or
  collinear, whose effects are not connected, or panels with missing values
  or duplicate geo-dates still go through `PanelOLS`. At 1,000 geos × 365
  days a permutation takes ~14 ms instead of ~13 s, so the default 1,000
  permutations run in ~15 s rather than ~3.5 hours. `power_simulation` builds
  its synthetic panels with array operations (bit-identical data).
- **Faster simplex donor weights** — `SyntheticControl`,
  `AugmentedSyntheticControl`, `SyntheticControlV` and `SyntheticDiffInDiff`
  fit their simplex-constrained weights (``w >= 0, sum(w) = 1``) with
  non-negative least squares (`_solve_simplex_lsq`) instead of SLSQP. SLSQP's
  cost grows roughly cubically with the donor count and conformal inference
  refits the weights dozens of times, so 400-donor panels took minutes; NNLS
  matches a tightly converged SLSQP to ~1e-14 in relative loss and is orders
  of magnitude faster. `SyntheticControl` and `AugmentedSyntheticControl`
  results are unchanged (to 1e-6). `PenalizedSyntheticControl` keeps SLSQP:
  its linear discrepancy penalty has no least-squares form.
- **`SyntheticControlV` and `SyntheticDiffInDiff` estimates shift slightly**
  — their SLSQP calls had no analytic gradient and a loose tolerance, and
  stopped short of the optimum while reporting success (SDID's near-exact
  time-weight fits stopped at a loss of ~1e-8 rather than ~1e-13, before the
  ridge that selects among them took effect). NNLS reaches the optimum, so
  their incrementality and intervals move by roughly 0.01-2% on test panels.
  The published-replication parity tests still pass.
- **Missing panel cells now raise a clear error** — a panel with a missing or
  non-finite outcome (e.g. a geo with no row for some date, which pivots to
  NaN) now raises a `ValueError` naming the estimator and the cause in
  `SyntheticControl`, `AugmentedSyntheticControl`, `SyntheticControlV` and
  `SyntheticDiffInDiff`. Previously SC returned a meaningless estimate, SDID
  returned NaN and ASC raised an unrelated `LinAlgError`.

### Fixed

- **Conformal interval search no longer runs away** — when the confidence set
  is unbounded (common for `ElasticNetSyntheticControl` and
  `GeneralizedSyntheticControl`, whose refits can absorb very large effects),
  the outward search doubled 40 times, to ~1e13 residual spreads, refitting
  the counterfactual at each step. It now stops at 1,000 spreads (probing the
  limit itself, so bounds just inside it are still found) and reports the
  bound as infinite. Finite bounds sit within a few tens of spreads, so
  this changes no finite interval; `ElasticNetSyntheticControl` at 100 donors
  x 365 days drops from 187 s to 17 s.
- **Confidence sets that are not a single interval** — the reported
  interval could exclude accepted effects or include rejected ones (e.g.
  `ElasticNetSyntheticControl` on the German reunification panel reported
  (-inf, inf) with p = 0.023, though zero and the point estimate are both
  rejected). The bounds now aim for the set's outer hull, a valid but
  conservative confidence set; when the set is found to have gaps, or the
  hull disagrees with the test at zero, its pieces are located, stored in
  `lift_ci_set` and reported in a `RuntimeWarning`. Detection is best-effort:
  the search evaluates finitely many effects, so an accepted piece lying
  wholly between them (beyond a rejected stretch, within 1,000 spreads) can
  still be missed. On 48 synthetic fits across eight
  estimators, 46 are unchanged; the two that changed (both ElasticNet) had
  been reporting a bound that cut off accepted effects.

## [0.20.0] - 2026-10-22

### Added

- **Altair plotting backend** — every `plot()` method (and
  `PowerAnalysis.plot_mde_sensitivity()` / `plot_power_heatmap()`) takes
  `backend="plotly" | "altair"`. Each Altair chart draws the same series,
  bands and reference lines as its Plotly figure, follows `dark_mode`, and
  accepts date, datetime, ISO-string or integer date axes. Altair is optional:
  `pip install "geocausality[altair]"`.
- **Hollow event-study markers** — `FixedEffects.plot()` draws a period's
  coefficient with a hollow marker when its confidence interval contains zero,
  and a filled one when it excludes zero, in both backends. Filled pre-onset
  markers flag a possible parallel-trends violation. The omitted reference
  period, pinned to zero, is always hollow. The Plotly line joining the
  markers is now drawn in the markers' colour, as in Altair, rather than the
  colorway's first colour.

### Changed

- **Breaking: `plot()` returns the figure instead of showing it** — every
  `plot()` method (and `PowerAnalysis.plot_mde_sensitivity()` /
  `plot_power_heatmap()`) now returns its figure rather than calling
  `fig.show()` and returning `None`. Notebooks still render the figure when it
  is the cell's last expression; scripts should call `.show()` on the result.
- **`GeoX.plot()` uses the shared counterfactual figure** — the same three
  panels as the synthetic-control estimators, so its bands now draw beneath
  the series, the cumulative band is orange like theirs, the top band has a
  legend entry, and the middle series is labelled "Difference" rather than
  "Residuals". The intervals shaded are unchanged. Every other Plotly figure
  is unchanged.
- **Plotting internals** — `GeoCausality._plotting` is now a package: each
  `plot()` builds a backend-free figure description (`figures`) and `render`
  draws it with a per-backend renderer (`_plotly`, `_altair`).

### Fixed

- **pandas input to the synthetic-control estimators** — `SyntheticControl`,
  `SyntheticControlV`, `AugmentedSyntheticControl`, `ElasticNetSyntheticControl`,
  `PenalizedSyntheticControl`, `RobustSyntheticControl`, `KernelSyntheticControl`,
  `NonlinearSyntheticControl`, `GeneralizedSyntheticControl`,
  `SyntheticDiffInDiff`, `MatrixCompletion`, `InteractiveFixedEffects`,
  `CausalImpact` and `GeoLift` raised `TypeError: DataFrame.pivot() got an
  unexpected keyword argument 'on'` on a pandas DataFrame. The donor matrix
  was pivoted with polars' native signature; it now uses narwhals' `pivot`,
  which works for both backends. Results on polars input are unchanged.

## [0.19.1] - 2026-10-16

### Changed

- **CI** — lint, test and build are skipped on doc-only changes; expensive
  tests are marked `slow` and run only on one CI cell per push, with the full
  suite running weekly on every OS/Python cell; tests run in parallel with
  `pytest-xdist`. No library code changed.

## [0.19.0] - 2026-10-12

### Added

- **Dark mode for plots** — every `plot()` method (and
  `PowerAnalysis.plot_mde_sensitivity()` / `plot_power_heatmap()`) takes
  `dark_mode: bool = False`. When set, figures use Plotly's `plotly_dark`
  template with lighter series, band and reference-line colours so they stay
  legible on a dark background. Light-mode output is unchanged.

### Fixed

- **Conformal test refits under the null** — the Chernozhukov, Wüthrich & Zhu
  moving-block test now refits the counterfactual on all pre- and post-period
  dates, with the post-period outcome reduced by each candidate effect, before
  permuting residuals. The test was fitted on the pre-period only and shifted
  the post-period residuals, so in-sample pre residuals were systematically
  smaller than post residuals. In simulation with no true effect and nominal
  10%, `SyntheticControl` rejected 35% of the time and
  `ElasticNetSyntheticControl` 48%. They now reject 12.5% and 9.5%, and CI
  coverage rose from 0.73 to 0.93 and from 0.62 to 0.98. The refit uses a new
  `_refit_counterfactual` hook:
  - the donor-weight estimators use it automatically;
  - `GeneralizedSyntheticControl` refits its loadings;
  - `DiffinDiff` and `FixedEffects` refit the parallel-trends gap;
  - `SyntheticControlV` refits unweighted simplex weights, because V only
    covers pre-period predictors.

  `InteractiveFixedEffects`, `MatrixCompletion` and `CausalImpact` have no hook
  yet and keep the old approximation. `results["conformal_refit"]` reports
  which path was used.
- **Prediction band** — the conformal band is calibrated on cross-fitted,
  out-of-sample pre-period residuals: blocks are held out and the model is
  refit without them. In-sample residuals made the band too narrow, worst with
  many donors: pointwise coverage was 0.44 against a nominal 0.90 for an OLS fit
  with 20 donors. On a 10-donor factor panel, `SyntheticControl`'s coverage
  rose from 0.80 to 0.86.
- **`DiffinDiff` with unequal group sizes** — groups are now averaged rather
  than summed per date before the 2x2 regression. With sums, a common trend
  entered each group scaled by its geo count and only cancelled when the groups
  were the same size. With 3 treated and 10 control geos and a true total
  effect of 600, the estimate was −7,800 and is now 600.
- **SLSQP stopping early** — `SyntheticControl`, `PenalizedSyntheticControl`
  and the shared simplex fit used by `AugmentedSyntheticControl` share a
  simplex QP solver. It normalises the objective by its starting value and uses
  an analytic gradient and a tight tolerance. On the rescaled problem the
  default absolute `ftol` let SLSQP report success near its uniform starting
  point: objective gaps reached 19% at the default penalty and 974% with no
  penalty, giving a counterfactual bias of about 1%.
- **`NonlinearSyntheticControl`** solves its QP at unit magnitude, with the
  penalties rescaled to keep the same minimiser. On raw outcomes SLSQP failed
  for almost every positive L1 penalty, so cross-validation could never select
  Tian's distance-weighted penalty.

### Changed

- **`DiffinDiff` results:**
  - `results["att"]` is new: the per-geo, per-date effect.
  - `results["lift"]` is `att` times the number of treated geos.
  - `results["test"]` and `results["control"]` are now per-geo group levels.
- **Fewer significant results for flexible estimators.** Ridge, elastic-net,
  kernel, factor-loading and penalized fits can absorb part of the post-period
  effect into the null refit, so the valid test is less powerful for them than
  the old, over-rejecting one. On Prop 99, the ASCM, elastic-net, penalized,
  GSC, Robust and Kernel estimates still match the published effect but are no
  longer significant at 10%. Plain synthetic control still is.
- **Runtime:** conformal inference costs a few dozen extra weight fits per
  `generate()`, about 1.3–1.4x the previous time for the synthetic-control
  estimators.

## [0.18.0] - 2026-10-09

### Fixed

- **Multiple treated geos** — `SyntheticControl`, `SyntheticControlV`,
  `PenalizedSyntheticControl`, `NonlinearSyntheticControl`,
  `SyntheticDiffInDiff` and `MatrixCompletion` now fit the per-geo average
  treated series and scale the counterfactual back up by the treated count.
  Sum-to-one weights cannot reproduce a sum of several geos, which badly biased
  estimates whenever more than one geo was treated. SDID's zeta rule now
  includes the treated count, and its placebo variance draws as many
  pseudo-treated donors as there are treated geos.
- **SLSQP failures** — `SyntheticControl` and `PenalizedSyntheticControl`
  rescale the weight problem to unit magnitude before solving and warn when the
  optimizer does not converge. Before this fix, outcomes in the thousands
  silently returned equal donor weights.
- **`AugmentedSyntheticControl`** now implements ridge ASCM (Ben-Michael,
  Feller & Rothstein): simplex synthetic-control weights plus a ridge
  correction, so the penalty shrinks toward synthetic control instead of
  toward zero.
- **`randomization_test()`** (all estimators and `GeoX`) excludes the real
  treated geos from placebo donor pools, keeps the estimator's hyperparameters,
  and uses placebos with the same number of geos as the treated group, so
  placebo and observed statistics share a scale.
- **`diagnose()` placebo-in-time** refits on pre-period data only. Before this
  fix, its fake post-period ran into the real post-period and picked up the
  treatment effect.
- **`control_geos`** is honoured when `test_geos` is also given: geos in
  neither list are dropped instead of silently joining the donor pool.
- **`GeoX`** derives its treatment indicator from `test_geos`/`control_geos`
  (so `randomization_test()` no longer fails on every placebo). With
  `non_negative=True` and a negative slope, it now refits the intercept-only
  model instead of keeping the OLS intercept.
- **`Switchback.power_analytical()`** uses the exact two-way fixed-effects
  variance under AR(1) errors for the switching schedule (new
  `switching_freq` argument). The earlier cluster design effect was far off
  and gave NaN for slightly negative autocorrelation.
- **Date handling** — dates are compared by calendar day via a shared
  `date_key`, so datetime columns (`"2021-01-01 00:00:00"`) no longer drop the
  last pre-period day. Dates strictly between `pre_period` and `post_period`
  are now dropped by every estimator. Before, they were trained on (most
  estimators) or counted as post-period (`MatrixCompletion`).
- **Conformal interval** — when the point estimate itself is rejected, the
  test inversion now searches for the accepted region instead of returning a
  zero-width interval, and it expands past the old ±10·spread cap. An empty
  region is reported as NaN with a warning.
- **Jackknife+ / split-conformal quantiles** — when there are too few
  pre-period points for a finite `(1 - alpha)` quantile, the interval is now
  unbounded (with a warning) instead of being clamped to the extreme
  residual, which voided the coverage guarantee.
- **Parametric bootstrap p-value** — the null replicates are centred on zero
  (removing the refit bias) and the p-value includes the +1 correction. The
  noise scale uses `ddof=1`.
- **`DiffinDiff`** — the relative lift and "Baseline" use the treated
  group's counterfactual level instead of the control group's. With
  `inference="ols"`, standard errors now allow AR(1) autocorrelation within
  each group's series; iid errors gave badly anti-conservative p-values.
- **`InteractiveFixedEffects`** — the automatic factor count is capped at
  `min(T, N0) - 2`. The old cap made the eigenvalue ratio divide by zero, so
  it always picked the maximum when there were few controls.
- **`Switchback`**:
  - Relative lift is measured against the untreated baseline. Before, it
    divided the incrementality by itself and always read 100%.
  - Carryover lags are built before washout rows are dropped.
  - `permutation_test()` re-applies lags and washout to each permuted schedule.
  - Incrementality counts every treated period, including washed-out ones,
    plus the carryover effects.
- **`GeoX`** — `alternative` now applies to pointwise and percent-lift
  intervals and to the placebo and randomization p-values, not only the
  cumulative interval.
- **`GeoLift`** — the p-value tests the ASC estimate against the GSC
  bootstrap null, so it agrees with the ASC-centred interval.
- **Residual jackknife+ interval** — the interval for the mean effect is now
  calibrated on means of `t1`-length blocks of pre-period residuals. The
  per-observation quantile was about `sqrt(t1)` times too wide. The faithful
  jackknife+ interval remains deliberately conservative; its docstring now
  says why.
- **Cost-per** — the incrementality is no longer rounded up before dividing,
  which biased the cost per acquisition downward.
- **`GeoX` percent-lift CI** — uses each day's full prediction standard error
  (including parameter uncertainty) over that day's counterfactual, instead
  of one residual sigma over the mean counterfactual.
- **`NonlinearSyntheticControl`** warns when the weight optimizer does not
  converge.
- **`RobustSyntheticControl`** treats only NaN cells as missing. Genuine zero
  outcomes no longer change the `p_hat` rescaling.
- **`KernelSyntheticControl`** — the leave-one-out leverage used to select
  the penalty now accounts for the fitted intercept.
- **`PowerAnalysis`** — a significant estimate with the wrong sign no longer
  counts as a detection. The MDE is searched by magnitude in one direction,
  so a negative effect is never reported as the MDE when positive effects
  were tested.
- **`CausalImpact`** — simulated counterfactual paths now draw the model
  parameters from their asymptotic distribution, so intervals include
  parameter uncertainty (null coverage of a nominal 90% interval went from
  ~0.80 to ~0.95 in simulation).
- **`Switchback.power_simulation()`** honours `treatment_fraction`, shares the
  analytical schedule, and reports a `1 - alpha` Monte Carlo interval for the
  power.
- **Docstrings** corrected for `ElasticNetSyntheticControl` (scikit-learn's
  `1/(2T)` loss scaling), `GeneralizedSyntheticControl` (zero factors gives a
  flat counterfactual), `MatrixCompletion` (no per-unit standardization or
  warm-started path), `InteractiveFixedEffects` (`tol` semantics), the
  factor-count range, `Switchback` (two-way FE, not the Horvitz-Thompson
  estimator; permutation-test assumptions) and `AugmentedSyntheticControl`
  (the penalty CV follows augsynth and is somewhat optimistic).

## [0.17.1] - 2026-09-25

### Added

- **`diagnose()`** on `EconometricEstimator` — runs pre-period diagnostics
  after fitting: pre-period MAPE and RMSE, Durbin-Watson autocorrelation
  statistic, augmented Dickey-Fuller residual stationarity test, and a
  placebo-in-time test that splits the pre-period at 70% and re-fits.
- **`randomization_test()`** on `GeoX` — Abadie-style placebo permutation
  test for the TBR estimator, with `avg_lift` and `sum_lift` statistics,
  `max_placebos`, `seed`, and `n_jobs` support.

## [0.17.0] - 2026-09-23

### Added

- **Randomization inference** (`randomization_test()`) — Abadie-style placebo
  permutation test on all `EconometricEstimator` subclasses. Treats each donor
  geo as a placebo treated unit, re-fits the full pipeline, and builds a null
  distribution. Supports three test statistics: `avg_lift`, `sum_lift`, and
  `mspe_ratio` (Abadie, Diamond & Hainmueller 2015). Rank-based two-sided
  p-value with optional `max_placebos` cap and `seed` for reproducibility.
- **README updates** — added Switchback estimator and GeoX Meridian
  enhancements to the quick-start examples and reference tables.

### Changed

- **Test fixture caching** — module-scoped pytest fixtures for
  `MatrixCompletion` and `MarketSelection` tests, eliminating redundant model
  fits (~30% faster test suite).

## [0.16.0] - 2026-09-20

### Added

- **`Switchback`** (`switchback`) — a switchback (crossover) experiment
  estimator for settings where treatment alternates on/off within geos over
  time (e.g. toggling paid-search ads). Fits a two-way fixed-effects panel
  regression (`PanelOLS`) with cluster-robust standard errors and provides:
  - **Washout periods** (`washout`) — discard observations immediately after
    each treatment switch to mitigate carryover bias.
  - **Carryover lag modeling** (`carryover_lags`) — explicit lagged treatment
    indicators as additional regressors.
  - **Randomization inference** (`permutation_test()`) — circular-shift
    permutation of switchback schedules for a non-parametric p-value.
  - **Analytical power** (`power_analytical()`) — closed-form MDE / power
    using an AR(1) autocorrelation-inflated variance formula.
  - **Simulation-based power** (`power_simulation()`) — Monte Carlo with
    AR(1) correlated errors and FE model fitting.
  - **Treatment schedule plot** (`plot()`) — color-coded treatment on/off per
    geo over time.
- **GeoX Meridian enhancements** — five features ported from Google's Meridian
  TBR implementation (Kerman, Wang & Vaver 2017):
  - `non_negative` — constrain the posterior credible interval to non-negative
    effects.
  - `alternative` (`"two-sided"` / `"greater"` / `"less"`) — one-sided
    hypothesis testing.
  - `validation_split` — hold out the first *k* pre-period days for model
    diagnostics.
  - Percent-lift confidence intervals in `results`.
  - `placebo_test()` — runs the TBR model on the pre-period as a
    specification check.
- **Estimator selection guide** (`docs/estimator_guide.rst`) — a decision
  flowchart and quick-reference table to help choose the right estimator for a
  given experiment design.

### Documentation

- New `switchback.rst` page with methodology (two-way FE, washout, carryover,
  randomization inference, power analysis) and usage examples.
- New "Advanced options" section in `geo_x.rst` documenting all Meridian
  enhancements.
- Updated `estimator_guide.rst` with the switchback design family and decision
  flowchart.

## [0.15.3] - 2026-09-20

### Added

- **`PowerAnalysis.plot_mde_sensitivity()`** — plots the Minimum Detectable
  Effect as a function of experiment duration, answering "how much does
  extending the experiment buy me?".
- **`PowerAnalysis.plot_power_heatmap()`** — renders a color-coded heatmap of
  power over the full (effect size × duration) grid with percentage labels.
- **DTW pre-filtering in `MarketSelection.search()`** — pass `dtw_top_k` to
  rank candidates by Dynamic Time Warping similarity (on z-scored pre-period
  series) and run the expensive power simulation only on the most promising
  ones. Inspired by the R *MarketMatching* package.

## [0.15.2] - 2026-09-03

### Fixed

- **ROAS now computes `(incrementality * msrp) / spend`** instead of
  `spend / incrementality` (which is cost-per-acquisition). `lift="roas"` and
  `lift="cost-per"` now return different values with correct metric labels.
- **RobustSyntheticControl SVD threshold off-by-one**: the truncation loop
  silently dropped the last singular value when all exceeded the threshold.
  Also swapped the loop condition order to prevent an `IndexError`.
- **RobustSyntheticControl fit-denoised / predict-raw inconsistency**: weights
  fitted on the SVD-denoised donor matrix are now applied to denoised data for
  prediction, matching Amjad, Shah & Shen (2018).
- **GeoX one-sided p-value**: changed from `P(X ≤ 0)` to a two-sided p-value
  (`2 * min(p, 1-p)`), consistent with every other estimator.
- **GeoX duplicate vlines**: `add_vline` was called once per trace instead of
  once per subplot.

## [0.15.1] - 2026-08-11

A packaging and tooling release. No changes to any estimator, the public API,
or runtime behaviour.

### Added

- **PyPI publishing workflow** (`.github/workflows/publish.yml`) that builds
  with `uv build` and publishes via PyPI Trusted Publishing (OIDC) on a
  published GitHub Release.

### Changed

- **Build backend switched from `poetry-core` to `uv_build`.** The package
  version is now declared once in `pyproject.toml` and read back at runtime via
  `importlib.metadata`, so `__version__` can no longer drift from the packaged
  metadata.
- **CI migrated from `pip` to `uv`** (`uv sync` / `uv run` / `uv build`, with
  `uvx` for the standalone `ruff` and `zuban` tools).
- Bumped the `pytest` dev dependency to `>=9.0.3,<10.0.0`.

### Removed

- `poetry.lock` and the `[tool.poetry]` section, superseded by `uv.lock` and
  `[tool.uv.build-backend]`.

### Fixed

- Behaviour-preserving type-checker and formatting cleanups to restore a green
  `lint` CI job: a closure-narrowing fix in `FixedEffects`, a `drop(index=...)`
  typing fix in the cross-validation splitter, and `ruff format` on `README.md`.
  No estimator outputs change.

## [0.15.0] - 2026-07-21

A feature release adding opt-in distribution-free inference to the two
econometric estimators. No breaking changes: the default behaviour of every
estimator is unchanged.

### Added

- **Conformal inference for `DiffinDiff` and `FixedEffects`** via a new
  `inference` constructor argument (`"ols"`, the default, `"conformal"`,
  `"jackknife"` or `"auto"`). The point estimate stays the OLS /
  fixed-effects coefficient; when `inference` is not `"ols"`, the p-value and
  confidence intervals are routed through the shared conformal engine
  (Chernozhukov-Wuthrich-Zhu moving-block permutation test, with a jackknife+
  residual fallback for short pre-periods) applied to a parallel-trends
  counterfactual. `DiffinDiff` builds the counterfactual on the geo-summed
  series; `FixedEffects` builds it on the treated/control group means so the
  residual scale matches its per-geo, per-day coefficient. Both keep their
  existing `"ols"` closed-form path as the default.

## [0.14.0] - 2026-07-17

A correctness and internal-consolidation release: five bug fixes (several
changing estimator outputs) and a large deduplication of the estimator family
into the shared base classes. No new estimators and no public API change.

### Fixed

- **Donor column-order dependence** in `AugmentedSyntheticControl`,
  `SyntheticControlV` and `RobustSyntheticControl`. These fit donor weights on a
  differently-ordered pivot than they predicted on, so on any input whose donor
  rows were not already geo-sorted the weight vector was applied to the wrong
  donors and the estimate was silently wrong. The donor matrix is now always
  built with a stable geo-sorted column order. **Results change for these
  estimators on non-geo-sorted inputs (they were previously incorrect).**
- **`GeoX.summarize()` crashed for every lift type** (`np.sum` on a narwhals
  `Series`) and **`GeoX.plot()` crashed on datetime/`Timestamp` date columns**
  (comparing a date to the `str` post-period). Both now work; `summarize` stores
  numpy arrays and `plot` compares on `datetime.date`.
- **Off-by-one in the split-conformal band** (`np.quantile(scores, k / n)`
  returned one order statistic too high), which made every conformal prediction
  band one residual too wide. Bands are now slightly tighter (correct); point
  estimates and effect confidence intervals are unaffected.
- **`summarize()` raised `ZeroDivisionError`** on the `relative` / `roas` /
  `cost-per` rows when the counterfactual or treated post-period summed to zero;
  it now reports `inf`, matching the existing `_get_roas` guard.
- **`MarketSelection` pre-fit metric** normalised the (geo-summed) conformal band
  by the per-geo-row mean, leaving a spurious `len(test_geos)` factor that biased
  `search()` against larger test sets. It now normalises by the summed-series
  mean; cross-size rankings change (single-size rankings are unchanged).

### Changed

- Large internal deduplication of the synthetic-control estimator family into
  `EconometricEstimator` / `Estimator`, with no public API change: the
  counterfactual `plot()`, `_get_roas`, the `_ols` / eigenvalue-ratio factor
  helpers, the narwhals frame helper, the results/inference assembly
  (`_finalize_counterfactual_results`), the `summarize()` lift-table formatting
  (`_validate_lift` / `_format_lift_cells`), and the donor-matrix construction
  (`_treated_series` / `_control_matrix`) are now defined once in the base.
  `InteractiveFixedEffects` now inherits the shared counterfactual `plot()`
  (which also restores the confidence bands its own plot had dropped).
- Standardised `_create_model` on the conventional `(x, y)` argument order across
  the estimator family and disambiguated the two `_loss_w` helpers.

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

[0.21.1]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.21.1
[0.21.0]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.21.0
[0.20.0]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.20.0
[0.19.1]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.19.1
[0.19.0]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.19.0
[0.18.0]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.18.0
[0.17.1]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.17.1
[0.17.0]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.17.0
[0.16.0]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.16.0
[0.15.3]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.15.3
[0.15.2]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.15.2
[0.15.1]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.15.1
[0.15.0]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.15.0
[0.14.0]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.14.0
[0.13.0]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.13.0
[0.12.1]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.12.1
[0.12.0]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.12.0
[0.11.0]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.11.0
[0.10.1]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.10.1
[0.10.0]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.10.0
[0.9.2]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.9.2
[0.9.1]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.9.1
[0.9.0]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.9.0
[0.8.0]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.8.0
[0.7.1]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.7.1
[0.7.0]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.7.0
[0.6.0]: https://github.com/ConorMcNamara/GeoCausality/releases/tag/v0.6.0
