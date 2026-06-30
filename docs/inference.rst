=========
Inference
=========

.. contents:: Table of Contents
   :depth: 2

Introduction
------------
Synthetic-control estimators produce a fitted counterfactual but no closed-form
standard errors, so GeoCausality attaches distribution-free inference on top.
Three paths are available, selected by ``inference_method`` and recorded in
``results["method"]``:

* ``"conformal"`` — the default for adequate pre-periods.
* ``"jackknife+"`` (with a ``"jackknife+ (residual)"`` approximation) — the
  automatic fallback for short pre-periods.
* ``"bootstrap"`` — the parametric bootstrap used by :doc:`geolift`.

By default ``inference_method="auto"`` chooses conformal for long pre-periods and
falls back to jackknife+ when the pre-period is too short. Setting it to
``"conformal"``, ``"jackknife"``, or ``"bootstrap"`` forces a path.

Conformal inference
-------------------
The default path uses the Chernozhukov, Wüthrich & Zhu (2021) moving-block
permutation test. For the sharp null of no effect it permutes the pre-/post-period
residuals over cyclic blocks and reports the fraction of blocks whose test
statistic is at least as large as the observed post-period block; the confidence
interval is obtained by inverting the test over candidate per-period effects. A
split-conformal band, calibrated on the absolute pre-period residuals, gives a
pointwise prediction band around the counterfactual.

Jackknife+ fallback
-------------------
When the pre-period is too short, the permutation p-value loses resolution and
the split-conformal quantile saturates. Inference then falls back to **jackknife+**
(Barber, Candès, Ramdas & Tibshirani, 2021), which reuses every pre-period point
for both fitting and calibration via leave-one-out and carries a distribution-free
coverage guarantee of at least :math:`1 - 2\alpha`.

Every synthetic-control estimator refits its own weights leave-one-out for a
faithful jackknife+ (``"jackknife+"``). An estimator without that refit hook uses
a residual-only approximation that holds the counterfactual fixed
(``"jackknife+ (residual)"``).

Parametric bootstrap
--------------------
Setting ``inference_method="bootstrap"`` performs inference by parametric
bootstrap — GeoLift's GSC-style approach and the inference engine behind
:doc:`geolift`. The fitted counterfactual is held fixed, parametric noise is drawn
at the pre-period residual scale, the weights are refit on each resampled
pre-period series, and the incrementality distribution is accumulated. The
confidence interval is the percentile interval centred on the reported
incrementality, and the p-value is the two-sided proportion of no-effect
replicates at least as extreme as the observed incrementality. ``n_boot`` and
``bootstrap_seed`` control the number of replicates and reproducibility.

The bootstrap is supported across the whole synthetic-control family.

References
----------
* Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). An Exact and Robust
  Conformal Inference Method for Counterfactual and Synthetic Controls.
  *Journal of the American Statistical Association*, 116(536), 1849-1864.
* Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2021).
  Predictive inference with the jackknife+. *Annals of Statistics*, 49(1),
  486-507.
