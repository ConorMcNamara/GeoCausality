==========================
Difference-in-Differences
==========================

.. contents:: Table of Contents
   :depth: 2

Introduction
------------
Difference-in-differences (DiD) is a statistical technique used to estimate the causal effect of a treatment or intervention by comparing the changes in outcomes over time between a treated group and a control group.  It's a quasi-experimental method, meaning it's applied in situations where controlled experiments are not feasible.

Core Idea
---------
DiD relies on comparing the difference in the outcome variable *before* and *after* the intervention for the treated group, to the difference in the outcome variable over the same period for the control group.  The idea is that the control group provides an estimate of how the treated group would have changed in the absence of the treatment.

Assumptions
-----------
The key assumption in DiD is the **parallel trends assumption**.  This assumes that, in the absence of the treatment, the treated and control groups would have followed similar trends in the outcome variable.

Mathematical Representation
--------------------------
Let:

* \(Y_{it}\) be the outcome for unit \(i\) at time \(t\).
* \(T_i\) be an indicator for whether unit \(i\) is in the treated group (\(T_i = 1\)) or the control group (\(T_i = 0\)).
* \(P_t\) be an indicator for the post-treatment period (\(P_t = 1\)) or the pre-treatment period (\(P_t = 0\)).

The DiD estimator is calculated as:

\(\qquad \hat{\tau} = (E[Y_{it} | T_i = 1, P_t = 1] - E[Y_{it} | T_i = 1, P_t = 0]) - (E[Y_{it} | T_i = 0, P_t = 1] - E[Y_{it} | T_i = 0, P_t = 0])\)

This can be simplified to:

\(\qquad \hat{\tau} = (Y_{1,post} - Y_{1,pre}) - (Y_{0,post} - Y_{0,pre})\)

Where:

* \(Y_{1,post}\) is the average outcome for the treated group in the post-treatment period.
* \(Y_{1,pre}\) is the average outcome for the treated group in the pre-treatment period.
* \(Y_{0,post}\) is the average outcome for the control group in the post-treatment period.
* \(Y_{0,pre}\) is the average outcome for the control group in the pre-treatment period.

Estimation
----------

In practice, DiD is often implemented using a regression model:

\(\qquad Y_{it} = \beta_0 + \beta_1 T_i + \beta_2 P_t + \beta_3 (T_i \times P_t) + \epsilon_{it}\)

Where:

* \(\beta_0\) is the intercept.
* \(\beta_1\) is the effect of being in the treated group.
* \(\beta_2\) is the effect of being in the post-treatment period.
* \(\beta_3\) is the DiD estimate (the treatment effect).
* \(\epsilon_{it}\) is the error term.

Advantages
----------

* **Relatively Simple:** DiD is conceptually and computationally straightforward.
* **Intuitive Interpretation:** The estimate has a clear interpretation as the difference in differences.
* **Handles Unobserved Confounding:** By differencing, DiD can eliminate the bias from time-invariant unobserved confounders.

Limitations
----------

* **Parallel Trends Assumption:** This is a strong assumption that is often difficult to verify.
* **Sensitivity to Functional Form:** The results can be sensitive to the specification of the regression model.
* **Potential for Serial Correlation:** With repeated observations, serial correlation can lead to underestimated standard errors.
* **Staggered Treatment Adoption:** Recent research has shown that with staggered treatment timing, the standard two-way fixed effects estimator can be biased.

Extensions and Recent Developments
---------------------------------

* **Multiple Time Periods:** DiD can be extended to settings with multiple pre- and post-treatment periods.
* **Variable Treatment Timing:** New methods have been developed to address the challenges of staggered treatment adoption (see, for example, Callaway and Sant'Anna, 2021).
* **Generalized DiD:** Researchers have worked on more flexible DiD methods that relax some of the core assumptions.

Conclusion
----------
Difference-in-differences is a widely used and valuable tool for estimating causal effects in quasi-experimental settings.  While it relies on the important parallel trends assumption, it provides a relatively simple and intuitive way to assess the impact of interventions when randomized controlled trials are not possible.  Researchers continue to develop new methods to improve the robustness and applicability of DiD in various contexts.

References
----------

* Angrist, J. D., & Pischke, J. S. (2008). *Mostly harmless econometrics: An empiricist's companion*. Princeton university press.
* Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, *225*(1), 68-95.
* Lechner, M. (2011). The estimation of causal effects by difference-in-differences methods. *Foundations and Trends® in Econometrics*, *4*(3), 165-224.
