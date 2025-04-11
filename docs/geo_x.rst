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
----------

* **Model Assumptions:** The accuracy of TBR depends on the validity of the assumptions of the regression model.

* **Extrapolation Risk:** Predicting the counterfactual response during the intervention period involves extrapolation, which can be unreliable if the relationship between the treatment and control groups changes significantly.

* **Potential for Bias:** Like any statistical model, TBR can be susceptible to bias if the model is misspecified or if there are unobserved confounders.

Relationship to Geo Experiments
-----------------------------
TBR is a valuable tool for analyzing data from geo experiments, particularly when the experimental design involves a small number of geos or matched market tests. It complements other geo experiment methodologies and provides a framework for estimating causal effects in challenging situations.

Key Research Papers
-------------------

* **Estimating Ad Effectiveness using Geo Experiments in a Time-Based Regression Framework**

    * Authors: Jouni Kerman, Peng Wang, and Jon Vaver

    * This paper introduces the Time-Based Regression (TBR) approach for analyzing geo experiments, especially when the number of geographic units is limited. It details the methodology and its advantages over traditional geo-based regression.

Conclusion
----------
Time Based Regression is a valuable technique for analyzing geo experiments, especially when the number of geos is limited. It provides a statistically sound approach for estimating causal effects from time series data, enabling researchers and practitioners to make informed decisions about interventions.
