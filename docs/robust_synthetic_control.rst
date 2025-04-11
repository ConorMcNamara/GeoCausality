==========================
Robust Synthetic Control
==========================

.. contents:: Table of Contents
   :depth: 2

Introduction
------------
The Synthetic Control Method (SCM) is a popular technique for estimating the effect of an intervention in comparative case studies. However, the standard SCM can be sensitive to noisy data and may not always select the most relevant control units. Robust Synthetic Control methods aim to address these limitations.

Motivation
----------
* **Sensitivity to Noise:** Standard SCM can be affected by noise in the pre-treatment outcomes and covariates, leading to unstable estimates.
* **Donor Unit Selection:** SCM may not always select the most appropriate donor units, potentially resulting in biased estimates.
* **High Dimensionality:** When the number of control units or covariates is large, standard SCM can become less reliable.

Robust Synthetic Control Methods
--------------------------------
Robust Synthetic Control methods aim to improve the stability and reliability of SCM by incorporating techniques that are less sensitive to noise and outliers. Here are a few approaches:

### 1. Robust Synthetic Control with Singular Value Thresholding
* **Denoising:** This method, proposed by Amjad, Shah, and Shen (2018), denoises the data matrix using singular value thresholding. This reduces the impact of noise and improves donor unit selection.
* **Algorithm:**
    1.  Denoise the data matrix using singular value thresholding.
    2.  Apply standard SCM to the denoised data.
* **Advantages:**
    * Less sensitive to noise.
    * Improved donor unit selection.
    * Works well with missing data.

### 2. Penalized Synthetic Control
* As discussed in the "Penalized Synthetic Control" document, adding a penalty term to the optimization can improve robustness.
* **L1 Penalty (Lasso):** Encourages sparsity, selecting a subset of control units.
* **L2 Penalty (Ridge):** Shrinks the weights, reducing the impact of any single control unit.
* **Elastic Net:** Combines L1 and L2 penalties.
* See the "Penalized Synthetic Controls" document for more details.

### 3. Augmented Synthetic Control
* The Augmented Synthetic Control Method (ASCM), introduced by Ben-Michael, Feller, and Rothstein (2021), also improves robustness.
* ASCM uses an outcome model to estimate and correct for bias due to imperfect pre-treatment fit. This can be helpful when a good match is hard to achieve with standard SCM.
* See the "Augmented Synthetic Controls" document for more details.

Advantages of Robust Synthetic Control
---------------------------------------
* **Increased Stability:** Less sensitive to noise and small data variations.
* **Improved Accuracy:** Can lead to more accurate estimates of the treatment effect.
* **Better Donor Selection:** Selects more relevant control units.
* **Handles High Dimensionality:** More reliable when there are many control units or covariates.

Limitations
----------
* **Computational Complexity:** Some robust methods can be computationally more intensive.
* **Parameter Tuning:** Robust methods may involve additional parameters that need to be tuned.
* **Bias-Variance Trade-off:** Robustness is often achieved by introducing some bias.

Conclusion
----------
Robust Synthetic Control methods offer valuable improvements over the standard SCM, particularly in the presence of noisy data or when dealing with a large number of control units.  Methods like those proposed by Amjad, Shah, and Shen (2018) and Ben-Michael, Feller, and Rothstein (2021) provide more reliable and stable estimates of treatment effects.

References
----------
* Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for comparative case studies: Estimating the effect of California's Proposition 99 on cigarette consumption. *Journal of the American Statistical Association*, 105(490), 493-505.
* Amjad, R. A., Shah, J. A., & Shen, C. (2018). Robust synthetic control. *arXiv preprint arXiv:1808.07167*.
* Ben-Michael, E., Feller, A., & Rothstein, J. (2021). The Augmented Synthetic Control Method. *Journal of the American Statistical Association*, *116*(536), 1789-1803.
