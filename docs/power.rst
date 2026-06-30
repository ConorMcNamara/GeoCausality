==============
Power Analysis
==============

.. contents:: Table of Contents
   :depth: 2

Introduction
------------
``PowerAnalysis`` is GeoCausality's pre-experiment power tool, analogous to
GeoLift's ``GeoLiftPower``. Before spending on a campaign, it answers: given this
geo split and our historical data, what lift could we actually detect, and for
how long must we run? It estimates statistical power and the **Minimum
Detectable Effect (MDE)** by simulating placebo experiments on clean pre-period
history.

Motivation
----------
Every estimator in GeoCausality is a *post*-experiment tool — it measures the
lift of a test that has already run. Designing a trustworthy test needs the
*pre*-experiment question answered first: a study that is underpowered for the
effect sizes you care about will produce an inconclusive result regardless of how
the analysis is done.

Methodology
-----------
The simulation operates on clean pre-period history, where no real treatment
effect is present:

1. For a grid of candidate **effect sizes** and **test durations**, repeatedly
   carve a placebo experiment out of the history: a sliding window becomes a fake
   post-period, earlier dates the fake pre-period, and the candidate effect is
   injected into the test geos.
2. Run the chosen estimator on each placebo and record whether its p-value
   clears ``alpha``.
3. **Power** is the fraction of placebos detected; the **MDE** is the smallest
   effect reaching a target power (default 0.8), interpolated between tested
   effects.

Running the grid at an effect of zero also calibrates the false-positive rate
against ``alpha`` — a free validation of the underlying inference.

Usage
-----

.. code-block:: python

   from GeoCausality import power
   from GeoCausality.augmented_synthetic_control import AugmentedSyntheticControl

   pa = power.PowerAnalysis(
       df,
       geo_variable="geo",
       treatment_variable="is_treatment",
       date_variable="date",
       pre_period="2022-06-30",   # last date of clean history
       y_variable="orders",
       estimator=AugmentedSyntheticControl,
   )
   pa.simulate(effect_sizes=[0.0, 0.05, 0.10, 0.15], durations=[14, 28], n_sims=200)
   pa.mde(target_power=0.8)
   pa.summarize()   # power curve + MDE table
   pa.plot()        # power-vs-effect curve, one line per duration

The estimator is pluggable, so power is measured with the same method that will
analyse the real experiment. Effect injection is multiplicative by default (a
fractional lift) and the placebo sampler is seeded for reproducibility.

References
----------
* GeoLift (Meta). ``GeoLiftPower``.
  https://facebookincubator.github.io/GeoLift/
