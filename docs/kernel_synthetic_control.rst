==========================
Kernel Synthetic Control
==========================

.. contents:: Table of Contents
   :depth: 2

Introduction
------------

* Where the linear synthetic-control family (and Tian's :doc:`nonlinear_synthetic_control`) build the counterfactual as a *combination of donor outcomes*, **Kernel Synthetic Control** learns a nonlinear *regression* of the treated series on the donor outcomes via kernel ridge.
* This is the estimator to reach for when the treated unit relates to the donors *nonlinearly* within a reasonably stationary regime.

Methodology
-----------

The counterfactual is a kernel-ridge fit with a composite **linear + RBF** kernel:

* each donor column is **standardised** so the kernel is not dominated by high-variance donors;
* a **linear** kernel term gives the model a global linear backbone, so it extrapolates trends (a treated unit whose donors drift outside their pre-period range does not collapse to the pre-period mean);
* an **RBF (Gaussian)** kernel term adds local nonlinear flexibility on top of that backbone;
* the treated series is **centred**, so its pre-period level is carried by an intercept rather than shrunk toward zero.

The fitted map is

.. math::

   f(x) = \bar{y} + K(x, X_{\text{pre}}) \,(K + \lambda I)^{-1} y_c

Tuning
------

* ``bandwidth`` (the RBF length-scale) defaults to the **median pairwise distance** of the pre-period donor rows in standardised space (the median heuristic).
* ``lambda_`` (the ridge penalty) defaults to the **one-standard-error rule** over a leave-one-out cross-validation grid.
* ``linear_weight`` sets the weight of the linear kernel term relative to the RBF term. ``0.0`` gives a pure RBF kernel (no linear extrapolation backbone); larger values lean more on the linear trend. A pure linear kernel (large ``linear_weight`` or large bandwidth) recovers a ridge-regularised linear synthetic control.

Both ``bandwidth`` and ``lambda_`` can be pinned directly.

When to prefer which estimator
------------------------------

* For **strongly trending** panels, prefer :doc:`nonlinear_synthetic_control` (Tian 2023) or the linear family — a nonlinear map attenuates toward the pre-period level and will not reproduce a trending-panel ATT magnitude.
* For treated/donor relationships that are **nonlinear within a stationary regime**, Kernel Synthetic Control captures structure the linear family cannot.

Inference
---------

Inference (conformal p-values and intervals, with the jackknife+ fallback for short pre-periods) is inherited unchanged from the shared nonlinear estimator base. See :doc:`inference`.

References
----------

* Kernel ridge regression / reproducing-kernel Hilbert space (RKHS) synthetic control.
* Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for comparative case studies. *Journal of the American Statistical Association*, 105(490), 493-505.
