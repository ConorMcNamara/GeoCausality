=============================
Penalized Synthetic Control
=============================

.. contents:: Table of Contents
   :depth: 2

Introduction
------------

* The Synthetic Control (SC) method estimates the effect of an intervention on a single treated unit.  It does this by constructing a "synthetic" control as a weighted average of untreated units.
* Standard SC, introduced by Abadie and Gardeazabal (2003) and Abadie, Diamond, and Hainmueller (2010), can struggle when the number of control units ($J$) is large relative to the number of pre-treatment periods ($T_0$).
* **Penalized Synthetic Control (PSC)** improves SC by adding a penalty to the weight estimation.  This leads to more stable and robust estimates.  See, for example, the work of Abadie and L'Hour (2021).

Motivation for Penalization
---------------------------

* When $J$ is large relative to $T_0$, standard SC can lead to:
    * **Sparsity Issues:** Many non-zero weights, making interpretation hard.
    * **Overfitting:** Perfect pre-treatment fit, but poor post-treatment predictions.
    * **Instability:** Weights very sensitive to small data changes.
* PSC addresses these by penalizing the weights, as discussed in Abadie and L'Hour (2021).

Common Penalization Techniques
------------------------------

* **L1 Penalty (Lasso):**
    * Adds the sum of absolute values of weights to the objective function.
    * Encourages sparsity (some weights become exactly zero).
    * Equation:
        $$\min_{w} \|Y_1 - Y_0 w\|_V^2 + \lambda \|w\|_1$$
* **L2 Penalty (Ridge):**
    * Adds the sum of squared weights to the objective function.
    * Shrinks weights towards zero (reduces variance).
    * Equation:
        $$\min_{w} \|Y_1 - Y_0 w\|_V^2 + \lambda \|w\|_2^2$$
* **Elastic Net Penalty:**
    * Combines L1 and L2 penalties.
    * Balances sparsity and stability.
    * Equation:
        $$\min_{w} \|Y_1 - Y_0 w\|_V^2 + \lambda_1 \|w\|_1 + \lambda_2 \|w\|_2^2$$

Methodology
-----------

1.  **Optimization Problem:**
    $$\min_{w} \|Y_1 - Y_0 w\|_V^2 + P(w, \lambda)$$
    * $Y_1$:  Pre-treatment outcomes for treated unit.
    * $Y_0$:  Pre-treatment outcomes for control units.
    * $w$:  Weights for control units.
    * $V$:  Weighting matrix.
    * $P(w, \lambda)$: Penalty function (L1, L2, Elastic Net) with tuning parameter $\lambda$.
2.  **Constraints:**
    * Weights sum to 1.
    * Weights are non-negative.
3.  **Tuning Parameter Selection:** Use cross-validation.
4.  **Post-treatment Prediction:**
    $$\hat{Y}_{1t}(0) = \sum_{j=1}^{J} \hat{w}_j Y_{jt}, \quad t > T_0$$
5.  **Treatment Effect:**
    $$\hat{\tau}_t = Y_{1t} - \hat{Y}_{1t}(0), \quad t > T_0$$

Advantages of Penalized Synthetic Control
-----------------------------------------

* **Improved Stability:** Less sensitive to small data changes.
* **Sparsity (with L1):** Selects a smaller, more interpretable set of control units.
* **Reduced Overfitting:** Improves out-of-sample prediction.
* **Handles High-Dimensional Settings:** Works well when $J$ is large relative to $T_0$.

Limitations of Penalized Synthetic Control
------------------------------------------

* **Bias:** Penalization introduces bias.
* **Tuning Parameter Selection:** Requires careful selection (e.g., cross-validation).
* **Interpretability (for L2, Elastic Net):** Can be harder to interpret than L1.
* **Choice of Penalty:** The "best" penalty varies by application.

Conclusion
----------

PSC is a valuable extension to standard SC, especially when there are many control units.  It improves stability and reduces overfitting, leading to more reliable estimates.  Key contributions to the development and understanding of Penalized Synthetic Control can be found in papers such as Abadie and L'Hour (2021).

References
----------

* Abadie, A., & Gardeazabal, J. (2003).  The economic costs of conflict: A case study of the Basque Country.  *American Economic Review*, *93*(1), 113-132.
* Abadie, A., Diamond, A., & Hainmueller, J. (2010).  Synthetic control methods for comparative case studies: Estimating the effect of California's Proposition 99 on cigarette consumption.  *Journal of the American Statistical Association*, *105*(490), 493-505.
* Abadie, A., & L'Hour, J. (2021).  A penalized synthetic control estimator for disaggregated data. *Journal of the American Statistical Association*, *116*(536), 1817-1834.
