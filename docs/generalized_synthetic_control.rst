===============================
Generalized Synthetic Control
===============================

.. contents:: Table of Contents
   :depth: 2

Introduction
------------
The Generalized Synthetic Control Method (GSC), introduced by Xu (2017), bridges
synthetic control and fixed-effects regression by modeling outcomes with an
*interactive fixed effects* (latent factor) structure. Where classic synthetic
control builds the counterfactual as a convex blend of donor units, GSC fits a
small number of latent time factors that are shared across units and weighted by
unit-specific loadings.

Motivation
----------
Two-way fixed effects assumes additive unit and time effects — equivalently,
that every unit shares the same time shocks up to a constant level (the parallel
trends assumption). That is often too rigid: regional outcomes typically share
seasonality and demand cycles that affect units to *different* degrees. GSC
relaxes parallel trends by letting each unit load differently on a set of common
latent factors, and it reduces to two-way fixed effects when the factor
structure is trivial.

Methodology
-----------
GSC models the outcome as

.. math::

   Y_{it} = \lambda_i^\top f_t + \delta_{it} + \varepsilon_{it}

where :math:`f_t` is a vector of :math:`r` latent time factors, :math:`\lambda_i`
the unit's factor loadings, and :math:`\delta_{it}` the treatment effect. The
procedure has three steps:

1. **Learn the factors from the controls.** The latent factors :math:`f_t` are
   estimated from the control units (which are never treated), so the post-period
   factors are uncontaminated by the treatment.
2. **Recover the treated loadings.** Each treated unit's loadings
   :math:`\lambda_i` are fit from its pre-period outcomes projected onto the
   factors.
3. **Project the counterfactual.** The untreated potential outcome is
   :math:`\hat Y_{it}(0) = \lambda_i^\top f_t`, and the effect is the gap between
   the observed outcome and this projection over the post-period.

The number of factors :math:`r` is selected by cross-validation on the
pre-period. In GeoCausality the implementation includes an intercept term so the
counterfactual matches the treated unit's level, and the factor count is chosen
via the :class:`~GeoCausality.utils.HoldoutSplitter` cross-validation utility.

Advantages
----------
* **Relaxes parallel trends:** different units may respond differently to common
  shocks.
* **Handles many treated units** natively through the shared factor model.
* **Degrades gracefully** to two-way fixed effects when the factor structure is
  trivial, providing a familiar anchor.

Limitations
-----------
* **Low-rank assumption:** GSC assumes the outcome is driven by a small number of
  latent factors; the estimate is sensitive to the chosen factor count.
* **Requires sufficient pre-period** to estimate factors and loadings reliably.

References
----------
* Xu, Y. (2017). Generalized Synthetic Control Method: Causal Inference with
  Interactive Fixed Effects Models. *Political Analysis*, 25(1), 57-76.
