========================
Switchback Experiments
========================

.. contents:: Table of Contents
   :depth: 2

Introduction
------------
A switchback (crossover) experiment alternates treatment assignment over time
within geographic units rather than fixing each geo as treatment or control for
the entire study. Each geo serves as both treatment and control at different
time windows, giving stronger within-unit identification when the number of
geos is small.

This design is common in marketplace and paid-search settings where:

* Treatment can be toggled quickly (e.g. turning paid-search ads on/off).
* The number of available geos is limited.
* Short-term causal effects are the primary interest.

Methodology
-----------
The estimator fits a two-way fixed-effects panel regression:

.. math::

   y_{it} = \alpha_i + \gamma_t + \tau D_{it} + \varepsilon_{it}

where :math:`\alpha_i` are geo fixed effects, :math:`\gamma_t` are time fixed
effects, :math:`D_{it}` is the time-varying binary treatment indicator, and
:math:`\tau` is the average treatment effect of interest.

Standard errors are clustered at the geo level to account for within-geo serial
correlation.

Washout periods
~~~~~~~~~~~~~~~
When treatment switches on or off, its effect may not vanish immediately.  The
``washout`` parameter discards observations immediately after each switch,
reducing carryover bias at the cost of sample size.

.. code-block:: python

   model = Switchback(df, ..., washout=1)  # drop 1 period after each switch

Carryover modeling
~~~~~~~~~~~~~~~~~~
Instead of (or in addition to) discarding observations, carryover can be
modeled explicitly by including lagged treatment indicators.  Setting
``carryover_lags=k`` adds ``treatment_lag_1`` through ``treatment_lag_k`` as
additional regressors. The main treatment coefficient then captures the
contemporaneous effect net of carryover.

.. code-block:: python

   model = Switchback(df, ..., carryover_lags=2)
   model.pre_process().generate()
   print(model.results["carryover"])  # per-lag estimates and p-values

Randomization inference
~~~~~~~~~~~~~~~~~~~~~~~
Because switchback experiments often have few geos, asymptotic standard errors
can be unreliable.  ``permutation_test()`` provides a non-parametric p-value by
circularly shifting the treatment schedule within each geo and comparing the
empirical null distribution of test statistics to the observed effect.

.. code-block:: python

   result = model.permutation_test(n_permutations=1000)
   print(result["empirical_p_value"])

Usage
-----

.. code-block:: python

   from GeoCausality.switchback import Switchback

   model = Switchback(
       df,
       geo_variable="geo",
       date_variable="date",
       y_variable="revenue",
       treatment_variable="ads_on",
       washout=1,
       carryover_lags=1,
   )
   model.pre_process().generate()
   model.summarize("incremental")
   model.plot()

   # Randomization inference
   placebo = model.permutation_test(n_permutations=1000)
   print(f"Empirical p-value: {placebo['empirical_p_value']:.4f}")

Power Analysis
--------------
The ``Switchback`` estimator provides both analytical (closed-form) and
simulation-based power calculations to help design experiments before they run.
Both methods require that ``generate()`` has already been called so that the
residual variance and autocorrelation structure can be estimated from the data.

Analytical power
~~~~~~~~~~~~~~~~
The analytical method uses a closed-form variance formula that accounts for
within-geo serial correlation via an AR(1) autocorrelation parameter
:math:`\rho`:

.. math::

   \text{Var}(\hat\tau) = \frac{\sigma^2 \bigl[1 + (T-1)\rho\bigr]}{G \cdot T \cdot p(1-p)}

where :math:`G` is the number of geos, :math:`T` the number of periods,
:math:`p` the treatment proportion, and :math:`\sigma^2` the residual variance.
Power is then computed from the non-central *t*-distribution.

Pass ``mde`` to get power for a given minimum detectable effect, or set
``mde=None`` with ``power_target`` to solve for the required MDE:

.. code-block:: python

   # Given MDE → power
   result = model.power_analytical(mde=5.0)
   print(f"Power: {result['power']:.2f}")

   # Given target power → required MDE
   result = model.power_analytical(mde=None, power_target=0.8)
   print(f"Required MDE: {result['mde']:.2f}")

You can also override the number of geos and periods to explore hypothetical
designs:

.. code-block:: python

   result = model.power_analytical(mde=3.0, n_geos=20, n_periods=90)

Simulation-based power
~~~~~~~~~~~~~~~~~~~~~~
The simulation method runs a Monte Carlo experiment: for each iteration it
generates synthetic data with AR(1) correlated errors, injects the specified
effect, fits the two-way fixed-effects model, and checks whether the treatment
coefficient is significant at the chosen level.

.. code-block:: python

   result = model.power_simulation(mde=5.0, n_simulations=500, seed=42)
   print(f"Simulated power: {result['power']:.2f}")
   print(f"95% CI: [{result['power_ci_lower']:.2f}, {result['power_ci_upper']:.2f}]")

The simulation approach is more flexible — it captures the exact finite-sample
behaviour including geo and time fixed effects — but is slower than the
analytical formula.

Limitations
-----------

* **Carryover assumption:** The validity of the estimate depends on carryover
  effects dissipating within the washout window or being fully captured by the
  lag terms.

* **Stationarity:** The treatment effect is assumed constant across switchback
  windows. Time-varying effects require additional modeling.

* **Few geos:** Cluster-robust standard errors may understate uncertainty with
  very few geos (< 5). The permutation test is preferred in this regime.

Key Research Papers
-------------------

* **Time Series Experiments and Causal Estimands: Exact Randomization Tests and
  Trading** — Bojinov & Shephard (2019). Foundational framework for exact
  randomization inference in time-series experiments.

* **Optimal Experimental Design for Staggered Rollouts** — Xiong et al. (2019).
  Uber's switchback design framework for marketplace experiments.
