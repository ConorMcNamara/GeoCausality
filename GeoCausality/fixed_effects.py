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

        Notes
        -----
        Based on https://matheusfacure.github.io/python-causality-handbook/14-Panel-Data-and-Fixed-Effects.html
        """
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

        Returns
        -------
        FixedEffects
            Itself, so it can be chained with summarize().
        """
        data_pd = self.data.to_pandas().set_index([self.geo_variable, self.date_variable])
        model = PanelOLS.from_formula(
            f"{self.y_variable} ~ campaign_treatment + EntityEffects + TimeEffects",
            data=data_pd,
        )
        self.model = model.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
        cis = self.model.conf_int(1 - self.alpha)
        self.results = {
            "test": float(self.model.params.iloc[0]),
            "control": 0.0,
            "lift": float(self.model.params.iloc[0]),
            "lift_ci_lower": float(cis["lower"].iloc[0]),
            "lift_ci_upper": float(cis["upper"].iloc[0]),
            "incrementality": float(self.model.params.iloc[0] * self.n_dates * self.n_geos),
            "incrementality_ci_lower": float(cis["lower"].iloc[0] * self.n_dates * self.n_geos),
            "incrementality_ci_upper": float(cis["upper"].iloc[0] * self.n_dates * self.n_geos),
            "p_value": float(self.model.pvalues.iloc[0]),
        }
        return self

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
