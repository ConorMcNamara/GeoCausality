=======
GeoLift
=======

.. contents:: Table of Contents
   :depth: 2

Introduction
------------
``GeoLift`` is a single, high-level entry point that reproduces the core of
Meta's `GeoLift <https://facebookincubator.github.io/GeoLift/>`_ pipeline. Rather
than asking the user to pick an estimator, it encodes GeoLift's specific
two-step design: estimate the lift with one method, and quantify the uncertainty
with another.

The two-step design
-------------------
Meta's GeoLift depends on both the ``augsynth`` and ``gsynth`` R packages and
describes its methodology as using

   "ASCM to estimate and de-bias the synthetic control estimate and then ... GSC's
   robustness on small samples and on heterogeneous effects across units to
   perform inference."

GeoCausality follows this exactly:

1. **Point estimate — Augmented Synthetic Control.** The de-biased lift,
   incrementality, and counterfactual come from
   :doc:`augmented_synthetic_control`.
2. **Inference — Generalized Synthetic Control.** The p-value and confidence
   interval come from :doc:`generalized_synthetic_control` run with
   :doc:`parametric-bootstrap inference <inference>`, which is robust on small
   samples and heterogeneous effects.

The reported interval is the ASC point estimate with the GSC bootstrap interval
recentred onto it, so the summary and plot reflect the ASC counterfactual while
the uncertainty comes from GSC. The raw GSC inference is retained under
``results["gsc_inference"]`` for transparency.

Usage
-----

.. code-block:: python

   from GeoCausality import geolift

   model = geolift.GeoLift(
       df,
       geo_variable="geo",
       treatment_variable="is_treatment",
       date_variable="date",
       pre_period="2022-06-30",
       post_period="2022-07-01",
       y_variable="orders",
       spend=500_000,
   )
   model.pre_process().generate().summarize(lift="incremental")
   model.plot()

Estimator-specific options are forwarded with ``asc_kwargs`` and ``gsc_kwargs``;
the bootstrap is controlled with ``n_boot`` and ``bootstrap_seed``.

Validation against GeoLift
--------------------------
GeoCausality ships a parity test that runs ``GeoLift`` on Meta's published
``GeoLift_Test`` example (a 15-day campaign in Chicago and Portland over 90
pre-period days) and checks the result against the documented numbers. The point
estimate lands within roughly one percentage point of GeoLift's published lift.

When to reach for it
--------------------
Use ``GeoLift`` when you want the opinionated, GeoLift-faithful pipeline in one
call. Use a specific estimator directly (see :doc:`augmented_synthetic_control`
and the other estimator pages) when you want a particular method or full control
over its inference path.
