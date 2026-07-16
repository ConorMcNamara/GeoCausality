=============================
Elastic-Net Synthetic Control
=============================

.. contents:: Table of Contents
   :depth: 2

Introduction
------------
The Doudchenko-Imbens synthesis (2016) unifies synthetic control,
difference-in-differences, and constrained regression as special cases of one
penalized regression. It relaxes three restrictions that the classic Abadie
synthetic control imposes on the donor weights, and adds elastic-net
regularisation, giving a single configurable estimator that spans the family.

Motivation
----------
Classic synthetic control constrains the donor weights to be non-negative and to
sum to one, and uses no intercept. Those choices keep the synthetic unit inside
the convex hull of the donors, but they can bias the counterfactual when the
treated unit sits systematically above or below any weighted average of the
donors (a level gap), or when a good pre-period fit needs weights outside the
simplex.

Method
------
The counterfactual is ``mu + donor_matrix @ w``, with the intercept ``mu`` and the
donor weights ``w`` fit on the pre-period by penalised least squares:

.. math::

   \min_{\mu, w}\ \sum_{t \in \text{pre}}
   \Big(y_t - \mu - \sum_j w_j x_{jt}\Big)^2
   + \lambda\Big[\tfrac{1 - \rho}{2}\lVert w\rVert_2^2 + \rho\lVert w\rVert_1\Big],

where :math:`\rho` is the elastic-net mixing parameter (``l1_ratio``) and
:math:`\lambda` is the penalty strength (``lambda_``). Three switches recover the
Doudchenko-Imbens relaxations of classic synthetic control:

* **intercept** (``intercept``) -- a level shift, so the synthetic unit need not
  match the treated unit's level, only its movements;
* **sum-to-one** and **non-negativity** (``sum_to_one`` / ``non_negative``) --
  dropping these lets the synthetic unit leave the donor convex hull;
* **elastic-net** regularisation (``l1_ratio``, ``lambda_``) -- ridge shrinkage
  spreads weight across donors, while the L1 term selects a sparse subset.

This MVP implements the unconstrained regime (the Doudchenko-Imbens default) with
scikit-learn's elastic net: the intercept and non-negativity are toggled directly,
and ``lambda_`` is chosen by cross-validation over a data-driven penalty path when
not supplied. The sum-to-one constrained regime is not yet implemented.

Special cases
-------------
* ``intercept=False, non_negative=True, sum_to_one=True, lambda_=0`` -- classic
  synthetic control (once the constrained path lands).
* ``intercept=True, l1_ratio=0`` -- a ridge-augmented fit close to
  :class:`~GeoCausality.augmented_synthetic_control.AugmentedSyntheticControl`
  (which is the dedicated closed-form ridge estimator; prefer it for pure ridge).
* ``lambda_=0`` with no constraints -- matched-market ordinary least squares.
* uniform weights plus an intercept -- difference-in-differences.

Inference
---------
Because the fit produces a donor-weight vector, Doudchenko-Imbens reuses the whole
shared inference stack: the Chernozhukov-Wuthrich-Zhu moving-block conformal
p-value and confidence interval, the faithful refit-based jackknife+ on short
pre-periods, and the parametric bootstrap -- all driven through the weights via the
estimator's refit hook.

Advantages
----------
* **One knob per assumption:** intercept, sparsity, shrinkage, and constraints are
  explicit, so the estimator spans DiD, regression, and synthetic control.
* **Level robustness:** the intercept tracks a treated unit outside the donor hull.
* **Runs out of the box:** the penalty is cross-validated by default.

Limitations
-----------
* **Extrapolation risk:** negative weights plus an intercept let the counterfactual
  extrapolate beyond the donors, which can overfit a short pre-period -- lean on the
  cross-validated penalty.
* **Pure ridge is inefficient:** ``l1_ratio=0`` uses coordinate descent (poor for
  pure L2); prefer ``AugmentedSyntheticControl`` there.
* **Sum-to-one not yet available:** the constrained regime needs a constrained
  solver and is left to a follow-up.

Conclusion
----------
Doudchenko-Imbens is the "synthesis" estimator: a single configurable regression
that reproduces the rest of the family as special cases while unlocking the
intercept-shifted, elastic-net, off-simplex regime the others cannot express.

References
----------
* Doudchenko, N., & Imbens, G. W. (2016). Balancing, Regression,
  Difference-in-Differences and Synthetic Control Methods: A Synthesis. *NBER
  Working Paper No. 22791*.
