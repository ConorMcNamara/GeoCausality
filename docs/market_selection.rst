================
Market Selection
================

.. contents:: Table of Contents
   :depth: 2

Introduction
------------
``MarketSelection`` is GeoCausality's pre-experiment test-market chooser,
analogous to GeoLift's ``GeoLiftMarketSelection``. Where :doc:`power` answers
"how detectable is *this* split?", market selection answers the prior question:
"*which* geos should we treat?" It enumerates candidate test-geo sets, scores
each, and returns a ranked recommendation.

Motivation
----------
The detectability and credibility of a geo experiment depend heavily on which
markets are treated. A poorly chosen test set may be impossible to reconstruct
from the donors (untrustworthy counterfactual) or too small to power the effect
of interest. Choosing the test markets by hand is error-prone; market selection
turns it into a data-driven search.

Methodology
-----------
Each candidate test set is scored on two axes, both computed on clean pre-period
history:

* **Power** — via :doc:`power`, the fraction of placebo experiments at the target
  effect and duration that are detected. This is the scoring engine; market
  selection is a search loop around it.
* **Pre-period fit** — how tightly the chosen estimator reconstructs the
  candidate test geos before any treatment (the estimator's split-conformal band,
  scaled by the test geos' mean outcome). A candidate the donors cannot reproduce
  pre-period gives an untrustworthy lift regardless of its power.

The two are normalised across candidates and combined into a single best-first
score (``fit_weight`` trades them off). When the full enumeration would exceed
``max_candidates``, a seeded sample is drawn and a warning is issued, so the
search is never silently truncated.

Usage
-----

.. code-block:: python

   from GeoCausality import market_selection
   from GeoCausality.augmented_synthetic_control import AugmentedSyntheticControl

   ms = market_selection.MarketSelection(
       df,
       geo_variable="geo",
       date_variable="date",
       pre_period="2022-06-30",
       y_variable="orders",
       estimator=AugmentedSyntheticControl,
   )
   ms.search(n_test_geos=[1, 2, 3], effect_size=0.10, duration=28, n_sims=200)
   ms.summarize()   # ranked test markets: power, pre-fit, score
   ms.plot()        # top candidates by score

Use ``include`` to force geos into every candidate set and ``exclude`` to bar
geos from being treated (they remain available as controls).

References
----------
* GeoLift (Meta). ``GeoLiftMarketSelection``.
  https://facebookincubator.github.io/GeoLift/
