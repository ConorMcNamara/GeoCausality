=============================
Augmented Synthetic Control
=============================

.. contents:: Table of Contents
   :depth: 2

Introduction
------------

* The Synthetic Control Method (SCM) estimates the effect of an intervention on a single treated unit by constructing a "synthetic" control as a weighted average of untreated units.
* A key limitation of the standard SCM is its reliance on strong pre-treatment fit.  When a close fit is not achievable, the Augmented Synthetic Control Method (ASCM) offers an improvement.
* ASCM, introduced by Ben-Michael, Feller, and Rothstein (2021), relaxes the requirement of near-perfect pre-treatment fit by incorporating an outcome model to adjust for bias.

Motivation for Augmented Synthetic Control
------------------------------------------

* The standard SCM, as proposed by Abadie and Gardeazabal (2003) and Abadie, Diamond, and Hainmueller (2010), performs best when the synthetic control closely matches the treated unit's pre-treatment outcomes.
* However, in many real-world scenarios, achieving this level of fit is challenging.  This can lead to biased estimates of the treatment effect.
* ASCM addresses this limitation by:
    * Estimating the bias due to imperfect pre-treatment fit.
    * Debiasing the original SCM estimate using an outcome model.

Methodology
-----------

ASCM involves the following key steps:

1.  **Standard Synthetic Control Estimation:** Initial weights are calculated using the standard SCM approach.
2.  **Outcome Model Estimation:** An outcome model (e.g., ridge regression) is used to model the relationship between pre-treatment outcomes and the treatment effect.
3.  **Bias Estimation:** The outcome model is used to estimate the bias in the treatment effect estimate that arises from imperfect pre-treatment fit.
4.  **Debiasing:** The initial SCM estimate is adjusted based on the estimated bias, resulting in the ASCM estimate.

The primary formulation of ASCM, as detailed in Ben-Michael, Feller, and Rothstein (2021), uses ridge regression as the outcome model.  This approach offers a balance between controlling pre-treatment fit and minimizing extrapolation.

Advantages of Augmented Synthetic Control
-----------------------------------------

* **Relaxed Pre-treatment Fit Requirement:** ASCM provides more reliable estimates in situations where achieving good pre-treatment fit with standard SCM is not feasible.
* **Bias Reduction:** By explicitly addressing bias due to imperfect pre-treatment fit, ASCM can lead to less biased estimates of the treatment effect.
* **Improved Robustness:** ASCM is shown to be more robust than standard SCM, particularly in the presence of model misspecification.

Limitations of Augmented Synthetic Control
------------------------------------------

* **Outcome Model Dependence:** The performance of ASCM depends on the choice and specification of the outcome model.
* **Potential for Extrapolation:** While ASCM aims to minimize extrapolation, it may still occur, especially when pre-treatment fit is poor.
* **Increased Complexity:** ASCM is computationally more complex than standard SCM.

Conclusion
----------

The Augmented Synthetic Control Method (ASCM) represents a significant advancement in the synthetic control methodology.  By relaxing the requirement of near-perfect pre-treatment fit and incorporating an outcome model to adjust for bias, ASCM provides a more robust and reliable approach to estimating treatment effects in a wider range of empirical settings.  The key methodological contributions are detailed in Ben-Michael, Feller, and Rothstein (2021).

References
----------

* Abadie, A., & Gardeazabal, J. (2003). The economic costs of conflict: A case study of the Basque Country. *American Economic Review*, *93*(1), 113-132.
* Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for comparative case studies: Estimating the effect of California's Proposition 99 on cigarette consumption. *Journal of the American Statistical Association*, *105*(490), 493-505.
* Ben-Michael, E., Feller, A., & Rothstein, J. (2021). The Augmented Synthetic Control Method. *Journal of the American Statistical Association*, *116*(536), 1789-1803.