===================
Fixed Effects Models
===================

.. contents:: Table of Contents
   :depth: 2

Introduction
------------
Fixed effects models are a statistical approach used to analyze panel data, which consists of observations of multiple entities (individuals, firms, countries, etc.) over multiple time periods.  These models are particularly useful for controlling for unobserved heterogeneity that is constant over time but varies across entities.

Core Idea
---------
The core idea behind fixed effects models is that if we assume the unobserved heterogeneity is correlated with the observed variables, then ordinary least squares (OLS) estimates will be biased. Fixed effects models address this by including entity-specific intercepts, which effectively "sweeps away" the influence of these time-invariant unobserved factors.

Model Specification
-------------------
The typical fixed effects model can be represented as:

\(Y_{it} = \beta X_{it} + \alpha_i + \epsilon_{it}\)

Where:

* \(Y_{it}\) is the dependent variable for entity \(i\) at time \(t\).
* \(X_{it}\) is the vector of independent variables for entity \(i\) at time \(t\).
* \(\beta\) is the vector of coefficients for the independent variables.
* \(\alpha_i\) is the fixed effect for entity \(i\), representing the time-invariant unobserved heterogeneity.
* \(\epsilon_{it}\) is the error term.

Assumptions
-----------

* **Strict Exogeneity:** The error term (\(\epsilon_{it}\)) is uncorrelated with the independent variables (\(X_{it}\)) in all time periods, conditional on the fixed effects (\(\alpha_i\)).
* **Time-Invariant Unobserved Heterogeneity:** The unobserved factors represented by \(\alpha_i\) do not change over time.
* **Linearity:** The relationship between the dependent and independent variables is linear.
* **Homoskedasticity and No Serial Correlation:** The error term has constant variance and is not correlated across time.

Estimation
----------
There are two common ways to estimate fixed effects models:

* **Least Squares Dummy Variable (LSDV) Method:** This involves including a dummy variable for each entity in the regression. While simple, this can be computationally expensive with a large number of entities.
* **Within Transformation (or demeaning):** This method transforms the data by subtracting the time average of each variable for each entity. This eliminates the \(\alpha_i\) term, and the transformed equation can be estimated by OLS.

Advantages
----------

* **Controls for Time-Invariant Unobserved Heterogeneity:** This is the primary advantage of fixed effects models. By eliminating the bias from these factors, fixed effects models can provide more consistent estimates of the effects of the time-varying independent variables.
* **Relatively Simple to Implement:** Once the data is transformed, the model can be estimated using standard regression techniques.

Limitations
----------

* **Cannot Estimate Effects of Time-Invariant Variables:** Because the fixed effects absorb all time-invariant variation, it is not possible to estimate the effects of variables that do not change over time (e.g., gender, race) using a standard fixed effects model.
* **Potential for Increased Variance:** The demeaning transformation can reduce the precision of the estimates, especially when there is little within-entity variation in the independent variables.
* **Strict Exogeneity Assumption:** The assumption that the error term is uncorrelated with the independent variables in all time periods can be difficult to satisfy in practice.
* **Incidental Parameters Problem:** In short panels (small T), the estimation of a large number of fixed effects can lead to biased estimates of the other parameters (though this bias tends to diminish as T increases).

Alternatives
------------

* **Random Effects Models:** If the unobserved heterogeneity is assumed to be uncorrelated with the observed variables, a random effects model may be more efficient.
* **First-Difference Models:** These models eliminate time-invariant unobserved heterogeneity by taking the first difference of the variables.
* **Mundlak Approach:** This approach augments a random effects model with the group means of the time-varying variables.

Conclusion
----------
Fixed effects models are a powerful tool for analyzing panel data when the unobserved heterogeneity is believed to be correlated with the observed variables.  By controlling for time-invariant unobserved factors, these models can provide more reliable estimates of the effects of time-varying independent variables.  However, it's important to be aware of the limitations of fixed effects models, such as the inability to estimate the effects of time-invariant variables and the potential for increased variance.

References
----------

* Baltagi, B. H. (2008). *Econometric analysis of panel data*. John Wiley & Sons.
* Wooldridge, J. M. (2010). *Econometric analysis of cross section and panel data*. MIT press.
* Arellano, M. (2003). *Panel data econometrics*. Oxford university press.
