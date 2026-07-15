==================
Matrix Completion
==================

.. contents:: Table of Contents
   :depth: 2

Introduction
------------
Matrix Completion (MC-NNM), introduced by Athey, Bayati, Doudchenko, Imbens, and
Khosravi (2021), reframes causal panel-data estimation as a missing-data problem.
Where the linear synthetic-control family regresses the treated series on a
weighted combination of donor geos, matrix completion stacks **every** unit —
donors *and* the treated series — into a single panel matrix, marks the treated
unit's post-period cells as missing, and completes the whole matrix under a
low-rank (nuclear-norm) penalty. The imputed treated post-period cells are the
counterfactual.

Motivation
----------
* **No donor-weight restriction:** the counterfactual is not constrained to be a
  convex (or even linear) combination of donors, so it can track a treated unit
  whose relationship to the donors is captured by latent factors rather than a
  weight vector.
* **Uses the whole panel:** every observed cell — not just the treated row —
  informs the low-rank structure, which is especially valuable when the donor
  pool is large.
* **Handles missing data and staggered adoption naturally:** because it completes
  an arbitrary pattern of missing cells, the same machinery extends to units that
  adopt treatment at different times.

Method
------
The panel outcome matrix is modelled as a low-rank term plus two-way additive
fixed effects and noise,

.. math::

   Y_{it} = L_{it} + \Gamma_i + \Delta_t + \varepsilon_{it},

and estimated by minimising, over the observed set :math:`\mathcal{O}` (all cells
except the treated unit's post-period),

.. math::

   \min_{L,\Gamma,\Delta}\ \frac{1}{|\mathcal{O}|}
   \sum_{(i,t)\in\mathcal{O}} \bigl(Y_{it}-L_{it}-\Gamma_i-\Delta_t\bigr)^2
   + \lambda_L \lVert L \rVert_*,

where :math:`\lVert L \rVert_*` is the nuclear norm (the sum of the singular
values of :math:`L`). The problem is solved by **soft-impute** (Mazumder, Hastie
& Tibshirani, 2010): fill the missing cells with the current estimate, take the
SVD, soft-threshold the singular values by :math:`\lambda_L`, and iterate —
alternating with a least-squares update of the row and column fixed effects. The
penalty :math:`\lambda_L` is chosen by cross-validation over a warm-started
decreasing path, holding out a random subset of the observed control cells.

In ``GeoCausality`` the treated geos are aggregated into a single row (matching
the estimand of the rest of the synthetic-control family); the individual donor
geos are the remaining rows. Row fixed effects absorb unit-level location
differences, and the low-rank factor loadings absorb multiplicative scale, so the
aggregated treated row and the individual donor rows sit on a comparable footing.

Inference
---------
Matrix completion produces a fitted counterfactual but no donor-weight vector and
no closed-form standard errors, so it reuses the shared distribution-free
inference of the estimator family: the Chernozhukov, Wüthrich & Zhu (2021)
moving-block conformal p-value and confidence interval, with the residual-only
jackknife+ fallback when the pre-period is too short. The weight-based faithful
jackknife+ and parametric bootstrap of the linear family do not apply, because
there is no weight vector to refit.

Advantages
----------
* **Flexible counterfactual:** captures latent factor structure without a
  hand-tuned rank or weight simplex.
* **Automatic regularisation:** the penalty is cross-validated, so the estimator
  runs out of the box.
* **Whole-panel information:** more reliable when donors are many.

Limitations
-----------
* **Attenuation:** the nuclear-norm penalty shrinks the counterfactual toward the
  observed data, so estimates can be attenuated toward zero on short or
  heterogeneous panels relative to plain synthetic control.
* **Computational cost:** each cross-validation candidate runs a full
  soft-impute solve, so fitting is heavier than a single linear solve.
* **No interpretable weights:** unlike synthetic control, there is no donor-weight
  vector to inspect.

Conclusion
----------
Matrix completion is a strong, largely tuning-free complement to the
synthetic-control family: it recovers the canonical Proposition 99 effect closely
and the German reunification effect in sign and order of magnitude, while
extending naturally to missing-data and staggered-adoption settings that the
weight-based methods handle less directly.

References
----------
* Athey, S., Bayati, M., Doudchenko, N., Imbens, G., & Khosravi, K. (2021).
  Matrix Completion Methods for Causal Panel Data Models. *Journal of the
  American Statistical Association*, 116(536), 1716-1730.
* Mazumder, R., Hastie, T., & Tibshirani, R. (2010). Spectral Regularization
  Algorithms for Learning Large Incomplete Matrices. *Journal of Machine Learning
  Research*, 11, 2287-2322.
