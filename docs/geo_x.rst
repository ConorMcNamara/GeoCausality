======================
Time Based Regression
======================

.. contents:: Table of Contents
   :depth: 2

Introduction
------------
Time Based Regression (TBR) is a methodology used to analyze data from geo experiments, particularly when the number of available geographic units (geos) is limited. It provides a way to estimate the causal effect of an intervention (e.g., an advertising campaign) on a response metric (e.g., sales revenue) by modeling the relationship between treatment and control groups over time.

Motivation
----------
Traditional geo-based regression (GBR) methods, which aggregate data over time, may not be suitable when dealing with a small number of geos. TBR offers an alternative approach that leverages the time series nature of the data. This is especially useful in situations like matched market tests, where only a single pair of treatment and control geos might be available.

Methodology
-----------
TBR involves using regression techniques to model the relationship between the response variable in the treatment and control groups during a pre-intervention period. This model is then used to predict the counterfactual response in the treatment group during the intervention period (i.e., what would have happened without the intervention).

1.  **Data Aggregation:** Unlike GBR, which aggregates data over time for each geo, TBR aggregates data across geos to create time series for the treatment and control groups.

2.  **Model Estimation:** A regression model is used to estimate the relationship between the treatment and control group time series during the pre-test period. A common model is:

    $y_t = \alpha + \beta x_t + \epsilon_t$

    Where:

    * $y_t$ is the response in the treatment group at time t.
    * $x_t$ is the response in the control group at time t.
    * $\alpha$ and $\beta$ are regression coefficients.
    * $\epsilon_t$ is the error term.

3.  **Counterfactual Prediction:** The estimated regression model is used to predict the counterfactual response in the treatment group during the intervention period.

4.  **Causal Effect Estimation:** The causal effect of the intervention is estimated by comparing the observed response in the treatment group during the intervention period with the predicted counterfactual response.

Advantages
----------

* **Handles Small Number of Geos:** TBR is designed to work effectively when the number of available geos is limited.

* **Utilizes Time Series Data:** TBR leverages the time series nature of the data, potentially leading to more accurate estimates.

* **Flexibility:** TBR can be applied in situations where GBR is applicable, and also in situations where GBR is not.

Limitations
-----------

* **Model Assumptions:** The accuracy of TBR depends on the validity of the assumptions of the regression model.

* **Extrapolation Risk:** Predicting the counterfactual response during the intervention period involves extrapolation, which can be unreliable if the relationship between the treatment and control groups changes significantly.

* **Potential for Bias:** Like any statistical model, TBR can be susceptible to bias if the model is misspecified or if there are unobserved confounders.

Relationship to Geo Experiments
-------------------------------
TBR is a valuable tool for analyzing data from geo experiments, particularly when the experimental design involves a small number of geos or matched market tests. It complements other geo experiment methodologies and provides a framework for estimating causal effects in challenging situations.

Advanced options
----------------

Non-negative slope constraint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
By default (``non_negative=True``) the regression slope is clamped to zero when
the OLS estimate is negative. This prevents nonsensical counterfactuals where
increased control activity would predict decreased treatment activity.

.. code-block:: python

   model = GeoX(df, ..., non_negative=True)   # default: clamp
   model = GeoX(df, ..., non_negative=False)  # allow negative slope

One-sided hypothesis tests
~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``alternative`` parameter controls the sidedness of the test:

- ``"two-sided"`` (default) — standard two-tailed test.
- ``"greater"`` — H_a: treatment > control. The p-value is P(delta <= 0).
- ``"less"`` — H_a: treatment < control. The p-value is P(delta > 0).

.. code-block:: python

   model = GeoX(df, ..., alternative="greater")
   model.pre_process().generate()
   # results["p_value"] is one-sided; CI lower bound is finite, upper is inf

Validation split
~~~~~~~~~~~~~~~~
``generate()`` automatically holds out the last *post-period-length* rows of the
pre-period, fits on the remainder, and stores the holdout RMSE as
``results["validation_rmse"]``. This gives an early signal of counterfactual
quality before the post-period is observed.

Percent-lift confidence intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After ``generate()``, the results dictionary contains log-ratio-based percent
lift estimates and CIs:

- ``results["percent_lift"]`` — point estimate, ``exp(log(y/ŷ)) - 1``.
- ``results["percent_lift_ci_lower"]`` / ``results["percent_lift_ci_upper"]``

These use the model's residual scale and a *t*-quantile, following the Meridian
GeoX log-ratio approach.

Placebo permutation test
~~~~~~~~~~~~~~~~~~~~~~~~
``placebo_test()`` splits the control geos into pseudo-treatment and
pseudo-control groups, fits TBR on each split, and compares the real
experiment's *t*-statistic against the empirical null distribution.

.. code-block:: python

   model = GeoX(df, ...).pre_process().generate()
   placebo = model.placebo_test(n_placebos=500, min_placebo_r2=0.6)
   print(placebo["empirical_p_value"])

The returned dictionary contains:

- ``placebo_t_stats`` — array of placebo *t*-statistics that passed the R² filter.
- ``real_t_stat`` — the real experiment's *t*-statistic.
- ``empirical_p_value`` — fraction of placebos at least as extreme.
- ``n_placebos_passed`` — how many passed the R² filter.
- ``n_placebos_total`` — how many were evaluated.

The result is also stored in ``results["placebo"]``.

Key Research Papers
-------------------

* **Estimating Ad Effectiveness using Geo Experiments in a Time-Based Regression Framework**

    * Authors: Jouni Kerman, Peng Wang, and Jon Vaver

    * This paper introduces the Time-Based Regression (TBR) approach for analyzing geo experiments, especially when the number of geographic units is limited. It details the methodology and its advantages over traditional geo-based regression.

Conclusion
----------
Time Based Regression is a valuable technique for analyzing geo experiments, especially when the number of geos is limited. It provides a statistically sound approach for estimating causal effects from time series data, enabling researchers and practitioners to make informed decisions about interventions.
