=====================================
Synthetic Difference-in-Differences
=====================================

.. contents:: Table of Contents
   :depth: 2

Introduction
------------
Synthetic Difference-in-Differences (SDID) was introduced by Arkhangelsky, Athey,
Hirshberg, Imbens, and Wager (2021). It combines the two workhorses of panel
causal inference — difference-in-differences (DID) and the synthetic control
method (SCM) — into a single doubly-weighted estimator that inherits the strengths
of both.

Motivation
------------
Difference-in-differences assumes that, absent treatment, the treated and control
units would have moved in *parallel*; it weights every control unit and every
pre-period equally. Synthetic control relaxes the parallel-trends assumption by
building a bespoke weighted control that tracks the treated unit's *level*, but it
drops the unit fixed effects and offers no time weighting.

SDID keeps the best of each: like SCM it fits unit weights so a reweighted control
pool tracks the treated unit, but like DID it retains unit and time fixed effects,
so the reweighted control only needs to move *parallel* to the treated unit rather
than match its level. It additionally fits **time** weights that concentrate the
comparison on the pre-periods most predictive of the post-period.

Methodology
-----------
SDID solves a weighted two-way fixed-effects regression,

.. math::

   \min_{\mu, \alpha_i, \beta_t, \tau} \sum_{i, t}
   \left( Y_{it} - \mu - \alpha_i - \beta_t - \tau W_{it} \right)^2
   \, \hat{\omega}_i \, \hat{\lambda}_t,

where :math:`W_{it}` indicates treatment and :math:`\tau` is the average treatment
effect on the treated. The two sets of weights are fit before the regression:

1. **Unit weights** :math:`\hat{\omega}` are chosen (with a free intercept and an
   :math:`L_2` penalty) so the reweighted control trajectory is parallel to the
   treated trajectory over the pre-period:

   .. math::

      \min_{\omega_0, \omega} \sum_{t \le T_0}
      \Big( \omega_0 + \sum_i \omega_i Y_{it} - Y_{\text{tr}, t} \Big)^2
      + \zeta^2 T_0 \lVert \omega \rVert^2,
      \quad \omega_i \ge 0, \ \sum_i \omega_i = 1.

   The penalty defaults to the Arkhangelsky et al. rule
   :math:`\zeta = T_{\text{post}}^{1/4}\, \hat{\sigma}`, where
   :math:`\hat{\sigma}` is the standard deviation of the first differences of the
   donor outcomes over the pre-period.

2. **Time weights** :math:`\hat{\lambda}` are chosen symmetrically so the weighted
   pre-periods predict each control unit's post-period average outcome.

Inference
---------
Because the treated geos are aggregated into a single series, the leave-one-out
jackknife is undefined, so SDID reports the **placebo variance** of Arkhangelsky
et al. (2021, §5): each donor is treated as a pseudo-treated unit against the
remaining donors, and the variance of the resulting placebo ATTs estimates the
sampling variance of the real ATT. This is deterministic (no resampling); the
standard error is exposed as ``results["standard_error"]`` and
``results["method"]`` is ``"placebo"``.

Advantages
----------

* **Double robustness:** consistent if *either* the unit or the time weights are
  correctly specified.
* **Relaxes parallel trends:** the reweighting only requires a reweighted control
  group to be on parallel trends, not the raw pool.
* **Level-robust:** the unit fixed effect means the donors need only match the
  treated *trend*, not its level — a weaker requirement than plain SCM.
* **Valid inference:** the placebo variance provides standard errors and
  confidence intervals.

Limitations
-----------

* **Donor scale:** like the rest of the synthetic-control family, the (summed)
  treated series and the donor pool should share a scale — the method is best
  suited to a single treated aggregate against a comparable donor pool.
* **Small donor pools:** the placebo variance is driven by the donor set, so a
  small or heterogeneous pool yields a large standard error.

Conclusion
----------
Synthetic Difference-in-Differences unifies difference-in-differences and
synthetic control: it draws robustness from reweighting units and times and
unbiasedness from the two-way fixed effects, and comes with a distribution-free
placebo variance for inference.

References
----------

* Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S.
  (2021). Synthetic Difference-in-Differences. *American Economic Review*,
  111(12), 4088-4118.
* Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for
  comparative case studies: Estimating the effect of California's Proposition 99
  on cigarette consumption. *Journal of the American Statistical Association*,
  105(490), 493-505.
