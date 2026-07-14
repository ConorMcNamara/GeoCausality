=============================
Nonlinear Synthetic Control
=============================

.. contents:: Table of Contents
   :depth: 2

Introduction
------------

* **Nonlinear Synthetic Control (NSC)**, introduced by Tian (2023), generalizes the classic synthetic control method to the case where the untreated outcome is a *strictly monotonic nonlinear* function of a latent linear index, :math:`y = F(x'\beta + \mu'\lambda + \varepsilon)`.
* The key insight is that because :math:`F` is monotonic, matching the treated unit's *observed* pre-period outcomes to a weighted combination of donor outcomes implicitly matches the underlying latent index. The counterfactual therefore remains the familiar linear-in-weights combination :math:`\sum_j w_j y_{jt}`, and **no link function has to be specified**.
* The "nonlinear" adaptation lives entirely in how the weights are solved for.

Methodology
-----------

The donor weights minimize a distance-weighted, doubly-penalized objective (Tian 2023, eq. 7):

.. math::

   \min_{w} \; \|z_1 - \textstyle\sum_j w_j z_j\|^2
   \; + \; a \sum_j |w_j|\,\|z_1 - z_j\|
   \; + \; b \sum_j w_j^2
   \quad \text{s.t.} \quad \textstyle\sum_j w_j = 1

where :math:`z_1` is the treated unit's pre-period trajectory and :math:`z_j` each donor's. Two departures from canonical synthetic control make it work under a nonlinear :math:`F`:

* **Affine (not simplex) weights** — they sum to one but may be negative, so the synthetic unit can match a treated unit that lies *outside* the donor convex hull (as ``AugmentedSyntheticControl`` also allows).
* **A distance-weighted L1 penalty** concentrates weight on donors close to the treated unit (nearest-neighbour matching as :math:`a \to \infty`), while the **L2 ridge** spreads it out (difference-in-differences as :math:`b \to \infty`).

The penalties :math:`a` and :math:`b` default to values chosen by **rolling-origin, predict-to-horizon cross-validation** over a scaled ``[0, 1]`` grid; either can be pinned directly.

Contrast with Kernel Synthetic Control
---------------------------------------

Unlike :doc:`kernel_synthetic_control` — which learns a genuinely nonlinear *map* of the donor outcomes and can revert to the pre-period mean when extrapolating — NSC keeps a linear combination of donor *outcomes*. It therefore tracks trends and reproduces the canonical Proposition 99 and German reunification results.

Inference
---------

Inference is the distribution-free moving-block permutation test (p-values and confidence intervals) shared across the synthetic-control family, with the jackknife+ fallback for short pre-periods. See :doc:`inference`.

References
----------

* Tian, W. (2023). The Synthetic Control Method with Nonlinear Outcomes. *arXiv:2306.01967*.
* Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for comparative case studies. *Journal of the American Statistical Association*, 105(490), 493-505.
