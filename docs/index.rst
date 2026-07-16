GeoCausality
============

GeoCausality is a Python library for measuring the causal impact of geo-level
A/B experiments. It provides a consistent, chainable API across a family of
estimators — from difference-in-differences to augmented and generalized
synthetic control — together with the pre-experiment design and inference tools
needed to plan and read a geo test, mirroring Meta's GeoLift workflow.

Every estimator shares the same three-step interface:

.. code-block:: python

   model.pre_process().generate().summarize(lift="incremental")

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   about
   quickstart

.. toctree::
   :maxdepth: 1
   :caption: Estimators

   geo_x
   diff_in_diff
   fixed_effects
   synthetic_control
   penalized_synthetic_control
   robust_synthetic_control
   matrix_completion
   augmented_synthetic_control
   elastic_net_synthetic_control
   generalized_synthetic_control
   nonlinear_synthetic_control
   kernel_synthetic_control
   synthetic_diff_in_diff
   causal_impact

.. toctree::
   :maxdepth: 1
   :caption: GeoLift Workflow

   geolift
   power
   market_selection
   inference

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api
   references

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
