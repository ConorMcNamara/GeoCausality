==========
Quickstart
==========

.. contents:: Table of Contents
   :depth: 2

Installation
------------

.. code-block:: bash

   pip install geocausality

GeoCausality requires Python ≥ 3.13.

Data format
-----------

Estimators accept a long-format panel (one row per geo per date) as either a
``pandas`` or ``polars`` DataFrame, with columns identifying the geo, the date,
and the outcome metric. Treatment is specified either with explicit
``test_geos`` / ``control_geos`` lists or a binary ``treatment_variable``
column (``1`` = treatment, ``0`` = control).

.. code-block:: text

   geo      date        orders   is_treatment
   chicago  2022-06-01  3300     1
   chicago  2022-06-02  3202     1
   boston   2022-06-01  2980     0
   ...      ...         ...      ...

The one-call interface
----------------------

``GeoLift`` reproduces Meta's GeoLift pipeline behind a single call: it uses
Augmented Synthetic Control for the de-biased point estimate and Generalized
Synthetic Control with parametric-bootstrap inference for the uncertainty.

.. code-block:: python

   import pandas as pd
   from GeoCausality import geolift

   df = pd.read_csv("geo_data.csv", parse_dates=["date"])

   model = geolift.GeoLift(
       df,
       geo_variable="geo",
       treatment_variable="is_treatment",
       date_variable="date",
       pre_period="2022-06-30",   # last date of the pre-treatment window
       post_period="2022-07-01",  # first date of the post-treatment window
       y_variable="orders",
       spend=500_000,
   )
   model.pre_process().generate().summarize(lift="incremental")
   model.plot()

   model.results["incrementality"]          # ASC de-biased point estimate
   model.results["p_value"]                 # GSC bootstrap inference
   model.results["incrementality_ci_lower"]

The chainable estimator interface
---------------------------------

Every estimator shares the same three-step, chainable interface
``pre_process() → generate() → summarize()``. Pick a specific estimator when you
want its exact method rather than the GeoLift pipeline:

.. code-block:: python

   from GeoCausality import synthetic_control

   model = synthetic_control.SyntheticControl(
       df,
       test_geos=["chicago", "portland"],
       date_variable="date",
       pre_period="2022-06-30",
       post_period="2022-07-01",
       y_variable="orders",
   )
   model.pre_process().generate().summarize(lift="relative")

``summarize`` lift options
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Description
   * - ``incremental``
     - Total absolute lift over the post-period
   * - ``absolute``
     - Per-period absolute lift
   * - ``relative``
     - Percentage lift vs. the counterfactual
   * - ``revenue``
     - Incremental revenue (requires ``msrp``)
   * - ``roas``
     - Return on ad spend (requires ``spend``)
   * - ``cost-per``
     - Cost per incremental unit (requires ``spend``)

Choosing the inference path
---------------------------

Synthetic-control estimators report a distribution-free p-value and confidence
interval, with the method recorded in ``results["method"]``. By default
(``"auto"``) inference uses conformal prediction and falls back to jackknife+ on
short pre-periods. Force a path before ``generate()``:

.. code-block:: python

   model.inference_method = "bootstrap"   # "auto" | "conformal" | "jackknife" | "bootstrap"
   model.pre_process().generate()

See :doc:`inference` for the details of each path.

Designing an experiment
-----------------------

Before running a test, use :doc:`power` to size the Minimum Detectable Effect
and :doc:`market_selection` to choose which geos to treat.
