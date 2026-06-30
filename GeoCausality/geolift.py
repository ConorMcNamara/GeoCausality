"""A faithful GeoLift entry point: ASC estimate + GSC bootstrap inference.

Meta's GeoLift is a specific two-step pipeline, not a free choice of estimator.
Its `DESCRIPTION` imports both ``augsynth`` and ``gsynth``, and the methodology
states: "GeoLift uses ASCM to estimate and de-bias the synthetic control estimate
and then uses GSC's robustness on small samples and on heterogeneous effects
across units to perform inference… GSC provides powerful parametric bootstrapping
approaches to provide valid and reliable inference and uncertainty estimates."

``GeoLift`` reproduces that here: it fits an :class:`AugmentedSyntheticControl`
for the de-biased point estimate (lift / incrementality / counterfactual) and a
:class:`GeneralizedSyntheticControl` with parametric-bootstrap inference for the
uncertainty (p-value and confidence intervals). The reported result is the ASC
point estimate with the GSC bootstrap interval recentred onto it, so the summary
and plot reflect the ASC counterfactual while the inference comes from GSC.
"""

from typing import Any

import numpy as np
from narwhals.typing import IntoDataFrame

from GeoCausality.augmented_synthetic_control import AugmentedSyntheticControl
from GeoCausality.generalized_synthetic_control import GeneralizedSyntheticControl


class GeoLift:
    """Run a geo experiment GeoLift-style: ASC point estimate, GSC bootstrap inference."""

    def __init__(
        self,
        data: IntoDataFrame,
        geo_variable: str = "geo",
        test_geos: list[str] | None = None,
        control_geos: list[str] | None = None,
        treatment_variable: str | None = "is_treatment",
        date_variable: str = "date",
        pre_period: str = "2021-01-01",
        post_period: str = "2021-01-02",
        y_variable: str = "y",
        alpha: float = 0.1,
        msrp: float = 0.0,
        spend: float = 0.0,
        n_boot: int = 1000,
        bootstrap_seed: int = 0,
        asc_kwargs: dict[str, Any] | None = None,
        gsc_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the GeoLift pipeline and build its two estimators.

        Parameters
        ----------
        data : pandas or polars data frame
            Our geo-based time-series data
        geo_variable : str
            The name of the variable representing our geo-data
        test_geos : list, optional
            The geos that were assigned treatment. If not provided, rely on treatment variable
        control_geos : list, optional
            The geos that were withheld from treatment. If not provided, rely on treatment variable
        treatment_variable : str, optional
            If test and control geos are not provided, the column denoting which is test and control. Assumes that
            1 is coded as "treatment" and 0 is coded as "control"
        date_variable : str
            The name of the variable representing our dates
        pre_period : str
            The time period used to train our models. Starts from the first date in our data to pre_period.
        post_period : str
            The time period used to evaluate our performance. Starts from post_period to the last date in our data
        y_variable : str
            The name of the variable representing the results of our data
        alpha : float, default=0.1
            The alpha level for our experiment
        msrp : float, default=0.0
            The average MSRP of our sale. Used to calculate incremental revenue.
        spend : float, default=0.0
            The amount we spent on our treatment. Used to calculate ROAS (return on ad spend)
             or cost-per-acquisition.
        n_boot : int, default=1000
            Number of parametric-bootstrap replicates for the GSC inference.
        bootstrap_seed : int, default=0
            Seed for the parametric bootstrap, so inference is reproducible.
        asc_kwargs : dict, optional
            Extra keyword arguments for the Augmented Synthetic Control estimator
            (e.g. ``lambda_``, ``conformal_q``).
        gsc_kwargs : dict, optional
            Extra keyword arguments for the Generalized Synthetic Control estimator
            (e.g. ``n_factors``, ``max_factors``, ``holdout_len``).
        """
        self.estimator = AugmentedSyntheticControl(
            data,
            geo_variable=geo_variable,
            test_geos=test_geos,
            control_geos=control_geos,
            treatment_variable=treatment_variable,
            date_variable=date_variable,
            pre_period=pre_period,
            post_period=post_period,
            y_variable=y_variable,
            alpha=alpha,
            msrp=msrp,
            spend=spend,
            **(asc_kwargs or {}),
        )
        self.inference = GeneralizedSyntheticControl(
            data,
            geo_variable=geo_variable,
            test_geos=test_geos,
            control_geos=control_geos,
            treatment_variable=treatment_variable,
            date_variable=date_variable,
            pre_period=pre_period,
            post_period=post_period,
            y_variable=y_variable,
            alpha=alpha,
            msrp=msrp,
            spend=spend,
            **(gsc_kwargs or {}),
        )
        self.inference.inference_method = "bootstrap"
        self.inference.n_boot = n_boot
        self.inference.bootstrap_seed = bootstrap_seed
        self.results: dict[str, Any] | None = None

    def pre_process(self) -> "GeoLift":
        """Pre-process both the estimation and inference models.

        Returns
        -------
        GeoLift
            Itself, so it can be chained with generate().
        """
        self.estimator.pre_process()
        self.inference.pre_process()
        return self

    def generate(self) -> "GeoLift":
        """Fit both models and combine the ASC estimate with GSC bootstrap inference.

        Returns
        -------
        GeoLift
            Itself, so it can be chained with summarize().
        """
        self.estimator.generate()
        self.inference.generate()

        estimate = self.estimator.results
        inference = self.inference.results
        if estimate is None or inference is None:
            raise ValueError("estimator and inference results must not be None")

        # Recentre the GSC bootstrap interval onto the ASC point estimate so the
        # reported interval brackets the reported lift, then graft the GSC p-value.
        t1 = np.asarray(estimate["lift"], dtype=float).ravel().shape[0]
        incr = estimate["incrementality"]
        lower_width = inference["incrementality"] - inference["incrementality_ci_lower"]
        upper_width = inference["incrementality_ci_upper"] - inference["incrementality"]
        estimate["incrementality_ci_lower"] = incr - lower_width
        estimate["incrementality_ci_upper"] = incr + upper_width
        estimate["lift_ci_lower"] = estimate["incrementality_ci_lower"] / t1
        estimate["lift_ci_upper"] = estimate["incrementality_ci_upper"] / t1
        estimate["p_value"] = inference["p_value"]
        estimate["conformal_band"] = inference.get("conformal_band")
        estimate["method"] = "asc + gsc-bootstrap"
        # Keep the raw GSC inference for transparency.
        estimate["gsc_inference"] = {
            "p_value": inference["p_value"],
            "incrementality_ci_lower": inference["incrementality_ci_lower"],
            "incrementality_ci_upper": inference["incrementality_ci_upper"],
            "method": inference["method"],
        }
        self.results = estimate
        return self

    def summarize(self, lift: str = "incremental") -> None:
        """Print a tabulated summary (ASC estimate with GSC bootstrap inference).

        Parameters
        ----------
        lift : str, default="incremental"
            The kind of lift to report. One of ``"absolute"``, ``"relative"``,
            ``"incremental"``, ``"cost-per"``, ``"revenue"`` or ``"roas"``.
        """
        return self.estimator.summarize(lift)

    def plot(self) -> None:
        """Plot the ASC counterfactual, pointwise and cumulative differences."""
        return self.estimator.plot()
