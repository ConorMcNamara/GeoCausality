=============
CausalImpact
=============

.. contents:: Table of Contents
   :depth: 2

Introduction
------------
CausalImpact estimates the effect of an intervention on a single treated time
series by forecasting the counterfactual — what the series *would* have done had
the intervention never happened — and reading the effect off the gap between the
observed series and that forecast. It was introduced by Brodersen et al. (2015)
:cite:`causalimpact2015` and is widely used for measuring the lift of marketing
campaigns and product launches.

Motivation
----------
Like synthetic control, CausalImpact answers "what would have happened without the
treatment?" for a single treated unit with a pool of untreated controls. Where
synthetic control builds the counterfactual as a *static* weighted blend of donor
units, CausalImpact builds it from a *structural time-series model*, so it can
absorb a drifting baseline, seasonality, and a regression on the controls at the
same time — and it yields a full predictive distribution, not just a point path.

Methodology
-----------
The model is a structural (state-space) time series with three parts:

1. A **time-varying level or trend** (a random-walk baseline that can drift).
2. An optional **seasonal** component.
3. A **regression** on the control series (the donor geos).

The model is fit on the **pre-period** only, learning the relationship between the
treated series and the controls. It is then rolled forward over the
**post-period**, conditioned on the controls' observed post-period values, to
produce the counterfactual. The per-period effect is ``actual − counterfactual``;
the cumulative effect is its running sum.

Brodersen et al. fit the model as a Bayesian structural time series (BSTS) via
MCMC. This implementation fits the same state-space model by maximum likelihood
through statsmodels' ``UnobservedComponents``, so it is the frequentist
structural-time-series cousin of the original.

Inference
---------
By default, inference is **native to the model**: the estimator draws many
counterfactual post-period paths from the fitted state posterior (seeding each with
an initial state sampled from the model's final predicted-state distribution) and
forms the cumulative-effect distribution across those draws. The confidence
interval is the percentile interval of that distribution, and the reported
p-value is the two-sided posterior tail-area probability of a zero effect. Setting
``inference_method`` to ``"conformal"`` or ``"jackknife"`` instead routes the
pre-/post-period residuals through the shared conformal machinery, for parity with
the synthetic-control family (see :doc:`inference`).

Advantages
----------

* **Dynamic counterfactual:** absorbs trend and seasonality rather than assuming a
  fixed relationship to the donors.
* **Native uncertainty:** the structural model yields a full predictive
  distribution, so intervals and a tail-area p-value come for free.
* **Familiar interface:** shares the chainable
  ``pre_process() → generate() → summarize()`` API and the three-panel counterfactual
  plot with the rest of the library.

Limitations
-----------

* **Control exogeneity:** the controls must not be affected by the intervention,
  and the pre-period relationship must continue to hold absent the treatment.
* **Maximum-likelihood, not fully Bayesian:** unlike the original BSTS, there are
  no spike-and-slab priors doing donor selection.

References
----------

* Brodersen, K. H., Gallusser, F., Koehler, J., Remy, N., & Scott, S. L. (2015).
  Inferring causal impact using Bayesian structural time-series models.
  *The Annals of Applied Statistics*, 9(1), 247-274. :cite:`causalimpact2015`
