"""Fixed-effects panel model for geo-experiment causal inference."""

import narwhals as nw
import numpy as np
import plotly.graph_objects as go
from linearmodels.panel import PanelOLS
from narwhals.typing import IntoDataFrame
from tabulate import tabulate  # type: ignore

from GeoCausality._base import EconometricEstimator


class FixedEffects(EconometricEstimator):
    """Run a two-way fixed-effects panel model for our geo-test."""

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
        inference: str = "ols",
    ) -> None:
        """Initialize the fixed-effects estimator.

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
        inference : str, default="ols"
            How to compute p-values and confidence intervals. ``"ols"`` uses the
            clustered standard errors of the ``campaign_treatment`` coefficient.
            The other options attach distribution-free inference to a
            parallel-trends counterfactual built on the treated/control group
            means via the shared conformal engine: ``"conformal"`` runs the
            Chernozhukov-Wuthrich-Zhu moving-block permutation test,
            ``"jackknife"`` uses the jackknife+ residual interval, and ``"auto"``
            picks conformal for long pre-periods and falls back to jackknife+ for
            short ones. In every case the point estimate stays the fixed-effects
            coefficient.

        Notes
        -----
        Based on https://matheusfacure.github.io/python-causality-handbook/14-Panel-Data-and-Fixed-Effects.html
        """
        valid_inference = {"ols", "conformal", "jackknife", "auto"}
        if inference not in valid_inference:
            raise ValueError(f"inference must be one of {sorted(valid_inference)}, got {inference!r}")
        super().__init__(
            data,
            geo_variable,
            test_geos,
            control_geos,
            treatment_variable,
            date_variable,
            pre_period,
            post_period,
            y_variable,
            alpha,
            msrp,
            spend,
        )
        self.inference = inference
        # Route the conformal-family options through the shared engine's dispatcher.
        if inference != "ols":
            self.inference_method = inference
        self.n_dates: int | None = None
        self.n_geos: int | None = None

    def pre_process(self) -> "FixedEffects":
        """Add the campaign-treatment interaction column and count treated dates and geos.

        Returns
        -------
        FixedEffects
            Itself, so it can be chained with generate().
        """
        super().pre_process()
        if self.treatment_variable is None:
            raise ValueError("treatment_variable must not be None")
        self.data: nw.DataFrame = self.data.with_columns(
            (nw.col("treatment_period") * nw.col(self.treatment_variable)).alias("campaign_treatment")
        )
        campaign_sum = (
            self.data.filter(nw.col("campaign_treatment") == 1).select(nw.col("campaign_treatment").sum()).item()
        )
        n_unique_geos = (
            self.data.filter(nw.col("campaign_treatment") == 1).select(nw.col(self.geo_variable).n_unique()).item()
        )
        self.n_dates = int(campaign_sum / n_unique_geos)
        self.n_geos = n_unique_geos
        return self

    def generate(self) -> "FixedEffects":
        """Fit the two-way fixed-effects model and store the lift estimates.

        The point estimate is always the ``campaign_treatment`` coefficient. The
        p-value and confidence intervals come from ``self.inference``: ``"ols"``
        uses the model's clustered standard errors, while the conformal-family
        options attach distribution-free inference to a parallel-trends
        counterfactual built on the treated/control group means.

        Returns
        -------
        FixedEffects
            Itself, so it can be chained with summarize().
        """
        if self.n_dates is None or self.n_geos is None:
            raise ValueError("n_dates and n_geos must not be None")
        data_pd = self.data.to_pandas().set_index([self.geo_variable, self.date_variable])
        model = PanelOLS.from_formula(
            f"{self.y_variable} ~ campaign_treatment + EntityEffects + TimeEffects",
            data=data_pd,
        )
        self.model = model.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
        lift = float(self.model.params.iloc[0])
        scale = self.n_dates * self.n_geos
        self.results = {
            "test": lift,
            "control": 0.0,
            "lift": lift,
            "incrementality": lift * scale,
        }
        if self.inference == "ols":
            cis = self.model.conf_int(1 - self.alpha)
            self.results["lift_ci_lower"] = float(cis["lower"].iloc[0])
            self.results["lift_ci_upper"] = float(cis["upper"].iloc[0])
            self.results["incrementality_ci_lower"] = self.results["lift_ci_lower"] * scale
            self.results["incrementality_ci_upper"] = self.results["lift_ci_upper"] * scale
            self.results["p_value"] = float(self.model.pvalues.iloc[0])
        else:
            actual_pre, prediction_pre, actual_post, prediction_post = self._counterfactual_from_parallel_trends()
            inference = self._conformal_inference(actual_pre, prediction_pre, actual_post, prediction_post)
            # The conformal ``lift`` CI is per-geo per-day (the coefficient scale),
            # so rescale incrementality by n_dates * n_geos to match the point
            # estimate rather than the engine's default per-period (* t1) scaling.
            self.results["lift_ci_lower"] = inference["lift_ci_lower"]
            self.results["lift_ci_upper"] = inference["lift_ci_upper"]
            self.results["incrementality_ci_lower"] = inference["lift_ci_lower"] * scale
            self.results["incrementality_ci_upper"] = inference["lift_ci_upper"] * scale
            self.results["p_value"] = inference["p_value"]
            self.results["conformal_band"] = inference["conformal_band"]
            self.results["method"] = inference["method"]
        return self

    def _counterfactual_from_parallel_trends(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build the treated group mean and its parallel-trends counterfactual.

        Averages the outcome over the treated geos and over the control geos per
        date, then takes the control mean level-shifted by the pre-period gap as
        the treated group's counterfactual. Working on group means (rather than
        geo-summed series) keeps the per-period residuals on the same per-geo,
        per-day scale as the fixed-effects coefficient, so the conformal p-value
        and confidence intervals are centred on the same effect.

        Returns
        -------
        A tuple of (actual_pre, prediction_pre, actual_post, prediction_post) as
        numpy arrays.
        """
        if self.treatment_variable is None:
            raise ValueError("treatment_variable must not be None")

        def group_mean(treatment_value: int, period: int) -> np.ndarray:
            return (
                self.data.filter(
                    (nw.col(self.treatment_variable) == treatment_value) & (nw.col("treatment_period") == period)
                )
                .group_by(self.date_variable)
                .agg(nw.col(self.y_variable).mean())
                .sort(self.date_variable)[self.y_variable]
                .to_numpy()
            )

        treated_pre, control_pre = group_mean(1, 0), group_mean(0, 0)
        treated_post, control_post = group_mean(1, 1), group_mean(0, 1)
        offset = treated_pre.mean() - control_pre.mean()
        return treated_pre, control_pre + offset, treated_post, control_post + offset

    def summarize(self, lift: str) -> None:
        """Print a tabulated summary of the estimated campaign lift.

        Parameters
        ----------
        lift : str
            The kind of lift to report. One of ``"absolute"``, ``"incremental"``,
            ``"cost-per"``, ``"revenue"`` or ``"roas"``.
        """
        if self.results is None:
            raise ValueError("self.results must not be None")
        lift = self._validate_lift(lift, allowed=("absolute", "incremental", "cost-per", "revenue", "roas"))
        ci_alpha = self._get_ci_print()
        lo_key, hi_key = f"{ci_alpha} Lower CI", f"{ci_alpha} Upper CI"
        incrementality = (
            self.results["incrementality"],
            self.results["incrementality_ci_lower"],
            self.results["incrementality_ci_upper"],
        )
        table_dict = {}
        if lift == "incremental":
            table_dict["Metric"] = [self.y_variable]
            table_dict["Lift Type"] = ["Incremental"]
            cells = self._format_lift_cells(lift, *incrementality)
        elif lift == "absolute":
            table_dict["Metric"] = [self.y_variable]
            table_dict["Lift Type"] = ["Absolute"]
            cells = self._format_lift_cells(
                lift, self.results["lift"], self.results["lift_ci_lower"], self.results["lift_ci_upper"]
            )
        elif lift == "revenue":
            table_dict["Metric"] = ["Revenue"]
            table_dict["Lift Type"] = ["Incremental"]
            cells = self._format_lift_cells(lift, *incrementality)
        else:
            table_dict["Metric"] = ["ROAS"]
            table_dict["Lift Type"] = ["Incremental"]
            cells = self._format_lift_cells(lift, *self._get_roas())
        table_dict["Lift"], table_dict[lo_key], table_dict[hi_key] = cells
        table_dict["p_value"] = [self.results["p_value"]]
        print(tabulate(table_dict, headers="keys", tablefmt="grid"))

    def plot(self) -> None:
        """Plot the event-study dynamic treatment effects for our fixed-effects model.

        Fits an auxiliary two-way fixed-effects model that interacts the treated
        indicator with each period relative to treatment onset (the period just
        before onset is the omitted reference), then plots the estimated
        coefficient path with confidence intervals. Pre-onset coefficients near
        zero support the parallel-trends assumption, while post-onset coefficients
        trace the dynamic treatment effect. This is fit independently of
        ``generate()`` and does not alter the headline single-coefficient results.

        Returns
        -------
        The event-study plot summarizing the dynamic treatment effect.
        """
        if self.data is None:
            raise ValueError("data must not be None")
        if self.treatment_variable is None:
            raise ValueError("treatment_variable must not be None")
        reference = -1
        data_pd = self.data.to_pandas()
        unique_dates = sorted(data_pd[self.date_variable].unique())
        date_rank = {d: i for i, d in enumerate(unique_dates)}
        post_dates = data_pd.loc[data_pd["treatment_period"] == 1, self.date_variable]
        start_rank = date_rank[min(post_dates)]
        data_pd["relative_period"] = data_pd[self.date_variable].map(lambda d: date_rank[d]) - start_rank
        event_columns = []
        for k in sorted(data_pd["relative_period"].unique()):
            if k == reference:
                continue
            column = f"evt_{k}".replace("-", "m")
            data_pd[column] = ((data_pd["relative_period"] == k) & (data_pd[self.treatment_variable] == 1)).astype(
                float
            )
            event_columns.append((k, column))
        indexed = data_pd.set_index([self.geo_variable, self.date_variable])
        formula = (
            f"{self.y_variable} ~ "
            + " + ".join(column for _, column in event_columns)
            + " + EntityEffects + TimeEffects"
        )
        event_model = PanelOLS.from_formula(formula, data=indexed).fit(cov_type="clustered", cluster_entity=True)
        cis = event_model.conf_int(1 - self.alpha)
        periods = [reference]
        estimates = [0.0]
        lowers = [0.0]
        uppers = [0.0]
        for k, column in event_columns:
            periods.append(k)
            estimates.append(float(event_model.params[column]))
            lowers.append(float(cis.loc[column, "lower"]))
            uppers.append(float(cis.loc[column, "upper"]))
        order = np.argsort(periods)
        periods = np.asarray(periods)[order]
        estimates = np.asarray(estimates)[order]
        lowers = np.asarray(lowers)[order]
        uppers = np.asarray(uppers)[order]
        fig = go.Figure(
            go.Scatter(
                x=periods,
                y=estimates,
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": uppers - estimates,
                    "arrayminus": estimates - lowers,
                },
                marker={"color": "purple"},
                mode="markers+lines",
                name="Effect",
            )
        )
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray")
        fig.add_vline(x=-0.5, line_width=1, line_dash="dash", line_color="black")
        fig.update_layout(
            title="Event Study (Two-Way Fixed Effects)",
            xaxis_title="Periods relative to treatment onset",
            yaxis_title=f"Effect on {self.y_variable}",
        )
        fig.show()
