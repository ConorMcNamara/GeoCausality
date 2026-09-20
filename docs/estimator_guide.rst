========================
Choosing an Estimator
========================

.. contents:: Table of Contents
   :depth: 2

GeoCausality ships 14 estimators plus one pipeline. This guide helps you pick
the right one for your experiment.

Quick reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 22 30 24 24

   * - Estimator
     - When to reach for it
     - Core assumption
     - Limitations
   * - :doc:`diff_in_diff`
     - Few geos, clear parallel pre-trends
     - Parallel trends
     - Biased when trends diverge
   * - :doc:`fixed_effects`
     - Rich panel, geo- and time-level confounders
     - Parallel trends + strict exogeneity
     - Same trend sensitivity as DiD
   * - :doc:`geo_x`
     - Very few geos (even a single matched pair)
     - Stable linear relationship between treated/control series
     - No donor weighting; aggregates across geos
   * - :doc:`synthetic_control`
     - Moderate donor pool, transparent weights needed
     - Treated lies in the convex hull of donors
     - Struggles when pre-fit is poor or donors are many
   * - ``PenalizedSyntheticControl``
     - Large donor pool relative to pre-period length
     - Same convex hull as classic SC
     - Still requires good pre-period fit
   * - ``RobustSyntheticControl``
     - Noisy outcome data with low-rank structure
     - Low-rank panel + noise
     - SVD tuning (threshold / rank) can matter
   * - :doc:`augmented_synthetic_control`
     - Default choice; imperfect pre-period fit
     - Donors untreated; ridge de-biases residual
     - Weights can be negative (less interpretable)
   * - ``ElasticNetSyntheticControl``
     - Want one estimator to span DiD–SC spectrum
     - Flexible (depends on constraint toggles)
     - Many knobs; harder to explain to stakeholders
   * - :doc:`generalized_synthetic_control`
     - Heterogeneous treatment effects, latent factors
     - Small number of latent time factors
     - Factor count selection matters; slower
   * - :doc:`matrix_completion`
     - Low-rank panel, possible missing data
     - Low-rank outcome matrix
     - No interpretable donor weights
   * - ``NonlinearSyntheticControl``
     - Outcome is a nonlinear function of a latent index
     - Monotonic nonlinear relationship
     - Newer method; less empirical track record
   * - ``KernelSyntheticControl``
     - Genuinely nonlinear donor–treated relationship
     - Nonlinear but stationary regime
     - Attenuates toward pre-period mean on trending data
   * - :doc:`synthetic_diff_in_diff`
     - Want robustness of both DiD and SC
     - Reweighted parallel trends (doubly robust)
     - Scalar ATT only; no per-period decomposition
   * - :doc:`causal_impact`
     - Strong trend/seasonality in the outcome
     - Pre-period dynamics continue post-treatment
     - Model specification (trend, seasonality) must be chosen
   * - :doc:`geolift`
     - Just want the recommended pipeline
     - Inherits ASC + GSC assumptions
     - Less control over individual method choices

Start here
----------

For most geo experiments, one of three paths covers you:

1. **Default / uncertain**: use :doc:`geolift` (the one-call pipeline) or
   :doc:`augmented_synthetic_control` directly. ASC de-biases classic synthetic
   control and is the default estimator for :doc:`power` and
   :doc:`market_selection`.

2. **Few geos or clear parallel trends**: use :doc:`diff_in_diff` or
   :doc:`fixed_effects`. These are the simplest estimators and the easiest to
   explain. If the pre-period trends are visually parallel, they work well.

3. **Strong seasonality or trend drift**: use :doc:`causal_impact`. It is the
   only estimator with a dynamic counterfactual that absorbs trend and seasonal
   components natively.

Estimator families
------------------

Classical econometric methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`diff_in_diff` and :doc:`fixed_effects` assume **parallel trends**: absent
treatment, the treated and control groups would have followed the same
trajectory. DiD is the textbook two-group, two-period design. Fixed Effects
extends it to a full panel with geo and time fixed effects, absorbing
time-invariant confounders and common shocks.

**Choose DiD/FE when** the parallel-trends assumption is credible and you want
the simplest, most interpretable result. **Avoid** when pre-period trends
clearly diverge — the bias goes directly into the estimate.

Time-based regression
~~~~~~~~~~~~~~~~~~~~~

:doc:`geo_x` (Kerman, Wang & Vaver) regresses the treated group's aggregated
time series on the control group's. It is designed for experiments with very
few geos — even a single matched pair.

**Choose GeoX when** you have minimal geo counts and a clear control match.
**Avoid** when you have a richer donor pool that synthetic control methods can
exploit.

Synthetic control family
~~~~~~~~~~~~~~~~~~~~~~~~

The core idea: build a weighted combination of untreated donors that
reproduces the treated unit's pre-period trajectory, then project forward as
the counterfactual.

- :doc:`synthetic_control` — the classic Abadie, Diamond & Hainmueller
  estimator. Convex weights (non-negative, sum-to-one). Transparent and
  interpretable but requires the treated unit to lie inside the donors' convex
  hull.
- ``PenalizedSyntheticControl`` (Abadie & L'Hour, 2021) — adds a ridge penalty
  to stabilize weights when the donor pool is large relative to the pre-period.
  Same constraints as classic SC, but less overfitting.
- ``RobustSyntheticControl`` (Amjad, Shah & Shen, 2018) — first denoises the
  donor matrix via SVD truncation, then fits SC on the cleaned data. Best when
  the panel is noisy but has clear low-rank structure.
- :doc:`augmented_synthetic_control` (Ben-Michael, Feller & Rothstein, 2021) —
  fits SC weights then de-biases with a ridge outcome model. Relaxes the
  requirement of near-perfect pre-period fit. **This is the library's default
  estimator.**
- ``ElasticNetSyntheticControl`` (Doudchenko & Imbens, 2016) — elastic-net
  regression with toggleable intercept, non-negativity, and sum-to-one. One
  estimator that can mimic DiD (intercept + uniform weights), classic SC (no
  intercept + simplex), or unconstrained regression. The Swiss army knife, but
  harder to explain.

**Choose a regularized SC variant when** classic SC overfits (weights
concentrated on one donor) or the pre-period fit is poor. ASC is the safest
default; Penalized SC when you want to stay on the simplex; Elastic Net when
you want maximum flexibility.

Latent factor and matrix methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`generalized_synthetic_control` (Xu, 2017) — interactive fixed effects.
  Learns latent time factors from the control panel and projects treated-unit
  loadings from the pre-period. Handles heterogeneous effects and relaxes
  parallel trends. Also provides parametric-bootstrap inference (used by
  :doc:`geolift`).
- :doc:`matrix_completion` (Athey et al., 2021) — nuclear-norm penalized matrix
  completion. Treats the treated unit's post-period as missing cells and imputes
  under a low-rank penalty. No donor-weight vector, but handles missing data
  naturally.

**Choose GSC when** you suspect the outcome is driven by a small number of
latent factors and want unit-level heterogeneity. **Choose Matrix Completion
when** the panel may have missing observations or you prefer a global low-rank
imputation over explicit donor weights.

Nonlinear methods
~~~~~~~~~~~~~~~~~

- ``NonlinearSyntheticControl`` (Tian, 2023) — generalizes SC to monotonic
  nonlinear outcomes. Solves for donor weights under a distance-weighted
  penalty without specifying a link function.
- ``KernelSyntheticControl`` — kernel ridge regression with a composite
  linear + RBF kernel. Learns a genuinely nonlinear mapping from donors to the
  treated outcome.

**Choose a nonlinear method when** the relationship between treated and donor
outcomes is visibly nonlinear (e.g. log-like or saturating). Prefer
``NonlinearSyntheticControl`` for trending data; ``KernelSyntheticControl``
attenuates toward the pre-period mean on strong trends.

Hybrid: Synthetic Difference-in-Differences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`synthetic_diff_in_diff` (Arkhangelsky et al., 2021) combines DiD and SC
by fitting both unit weights (matching treated *trend*, not level) and time
weights (focusing on the most predictive pre-periods). It is **doubly robust**:
consistent if either set of weights is correctly specified.

**Choose SDID when** you want insurance against both parallel-trends violations
(where SC helps) and poor donor fit (where DiD helps). Note that it produces a
scalar average treatment effect — no per-period decomposition.

Time-series model
~~~~~~~~~~~~~~~~~

:doc:`causal_impact` (Brodersen et al., 2015) fits a structural time-series
model (local level or local linear trend, optional seasonality, regression on
donors) via MLE. The counterfactual is a dynamic forecast, not a static
weighted average.

**Choose CausalImpact when** the outcome has strong trend or seasonal patterns
that a static counterfactual would mistrack. **Avoid** when you are unsure
about the right trend/seasonal specification — misspecification biases the
counterfactual.

Decision flowchart
------------------

.. code-block:: text

   Is this your first analysis or are you unsure?
   ├── Yes → GeoLift (or AugmentedSyntheticControl)
   └── No
       ├── Very few geos (1–3 treated)?
       │   ├── Yes → GeoX
       │   └── No
       │       ├── Strong seasonality or trend?
       │       │   ├── Yes → CausalImpact
       │       │   └── No
       │       │       ├── Parallel trends credible?
       │       │       │   ├── Yes → DiffinDiff or FixedEffects
       │       │       │   └── No
       │       │       │       ├── Nonlinear outcome?
       │       │       │       │   ├── Yes → NonlinearSC or KernelSC
       │       │       │       │   └── No
       │       │       │       │       ├── Want robustness to both DiD and SC?
       │       │       │       │       │   ├── Yes → SyntheticDiffInDiff
       │       │       │       │       │   └── No → AugmentedSyntheticControl
       │       │       │       │       └──
       │       │       │       └──
       │       │       └──
       │       └──
       └──
