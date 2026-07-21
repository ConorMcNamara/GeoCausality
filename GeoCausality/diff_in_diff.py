"""Difference-in-differences method for geo-experiment causal inference."""

import narwhals as nw
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from narwhals.typing import IntoDataFrame
from tabulate import tabulate  # type: ignore

from GeoCausality._base import EconometricEstimator


class DiffinDiff(EconometricEstimator):
    """Run difference-in-differences for our geo-test."""

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
        """Initialize the difference-in-differences estimator.

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
            closed-form standard errors of the interaction coefficient. The other
            options attach distribution-free inference to the parallel-trends
            counterfactual via the shared conformal engine: ``"conformal"`` runs
            the Chernozhukov-Wuthrich-Zhu moving-block permutation test,
            ``"jackknife"`` uses the jackknife+ residual interval, and ``"auto"``
            picks conformal for long pre-periods and falls back to jackknife+ for
            short ones. In every case the point estimate stays the OLS
            difference-in-differences coefficient.

        Notes
        -----
        Based on https://matheusfacure.github.io/python-causality-handbook/13-Difference-in-Differences.html
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
        self.groupby_data: nw.DataFrame | None = None
        self.n_dates: int | None = None

    def pre_process(self) -> "DiffinDiff":
        """Aggregate the data by treatment, period and date and count post-period dates.

        Returns
        -------
        DiffinDiff
            Itself, so it can be chained with generate().
        """
        super().pre_process()
        if self.treatment_variable is None:
            raise ValueError("treatment_variable must not be None")
        self.groupby_data = (
            self.data.group_by(
                [self.treatment_variable, "treatment_period", self.date_variable],
            )
            .agg(nw.col(self.y_variable).sum())
            .sort([self.treatment_variable, "treatment_period", self.date_variable])
        )
        self.n_dates = self.groupby_data.filter(
            (nw.col(self.treatment_variable) == 1) & (nw.col("treatment_period") == 1)
        ).shape[0]
        return self

    def generate(self) -> "DiffinDiff":
        """Fit the OLS model and compute lift, confidence intervals and incrementality.

        The point estimate is always the interaction coefficient of the
        difference-in-differences OLS model. The p-value and confidence intervals
        come from ``self.inference``: ``"ols"`` uses the model's closed-form
        standard errors, while the conformal-family options attach
        distribution-free inference to the parallel-trends counterfactual.

        Returns
        -------
        DiffinDiff
            Itself, so it can be chained with summarize().
        """
        if self.groupby_data is None:
            raise ValueError("groupby_data must not be None")
        if self.n_dates is None:
            raise ValueError("n_dates must not be None")
        self.model = smf.ols(
            f"{self.y_variable} ~ {self.treatment_variable} * treatment_period",
            data=self.groupby_data.to_pandas(),
        ).fit()
        self.results = dict(
            test=float(self.model.params["treatment_period"]),
            control=float(self.model.params["Intercept"]),
            lift=float(self.model.params[f"{self.treatment_variable}:treatment_period"]),
        )
        self.results["incrementality"] = self.results["lift"] * self.n_dates
        if self.inference == "ols":
            cis = self.model.conf_int(alpha=self.alpha, cols=None).iloc[-1]
            self.results["lift_ci_lower"] = float(cis[0])
            self.results["lift_ci_upper"] = float(cis[1])
            self.results["incrementality_ci_lower"] = self.results["lift_ci_lower"] * self.n_dates
            self.results["incrementality_ci_upper"] = self.results["lift_ci_upper"] * self.n_dates
            self.results["p_value"] = float(self.model.pvalues[f"{self.treatment_variable}:treatment_period"])
        else:
            actual_pre, prediction_pre, actual_post, prediction_post = self._counterfactual_from_parallel_trends()
            self.results.update(
                self._conformal_inference(actual_pre, prediction_pre, actual_post, prediction_post)
            )
        return self

    def _counterfactual_from_parallel_trends(
        self,
    ) -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
        """Build the treated series and its parallel-trends counterfactual.

        Mirrors the counterfactual drawn in ``plot()``: the treated group's
        per-date outcome is the observed series, and the counterfactual is the
        control group's per-date outcome level-shifted by the pre-period gap
        between the two groups. Feeding these four series to the conformal engine
        gives per-period residuals whose mean equals the difference-in-differences
        estimand, so the conformal p-value and confidence intervals are centred on
        the same effect as the OLS coefficient.

        Returns
        -------
        A tuple of (actual_pre, prediction_pre, actual_post, prediction_post) as
        numpy arrays.
        """
        if self.groupby_data is None:
            raise ValueError("groupby_data must not be None")
        if self.treatment_variable is None:
            raise ValueError("treatment_variable must not be None")
        treated = self.groupby_data.filter(nw.col(self.treatment_variable) == 1).sort(self.date_variable)
        control = self.groupby_data.filter(nw.col(self.treatment_variable) == 0).sort(self.date_variable)
        treated_pre = treated.filter(nw.col("treatment_period") == 0)[self.y_variable].to_numpy()
        control_pre = control.filter(nw.col("treatment_period") == 0)[self.y_variable].to_numpy()
        treated_post = treated.filter(nw.col("treatment_period") == 1)[self.y_variable].to_numpy()
        control_post = control.filter(nw.col("treatment_period") == 1)[self.y_variable].to_numpy()
        offset = treated_pre.mean() - control_pre.mean()
        return treated_pre, control_pre + offset, treated_post, control_post + offset

    def summarize(self, lift: str) -> None:
        """Print a tabulated summary of the difference-in-differences results.

        Parameters
        ----------
        lift : str
            The kind of lift to report. One of ``"absolute"``, ``"relative"``,
            ``"incremental"``, ``"cost-per"``, ``"revenue"`` or ``"roas"``.
        """
        if self.results is None:
            raise ValueError("results must not be None")
        if self.n_dates is None:
            raise ValueError("n_dates must not be None")
        lift = self._validate_lift(lift)
        ci_alpha = self._get_ci_print()
        lo_key, hi_key = f"{ci_alpha} Lower CI", f"{ci_alpha} Upper CI"
        base_level = self.results["control"] * self.n_dates
        incrementality = (
            self.results["incrementality"],
            self.results["incrementality_ci_lower"],
            self.results["incrementality_ci_upper"],
        )
        table_dict = {
            "Variant": [base_level + self.results["incrementality"]],
            "Baseline": [base_level],
        }
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
        elif lift == "relative":
            table_dict["Metric"] = [self.y_variable]
            table_dict["Lift Type"] = ["Relative"]
            cells = self._format_lift_cells(lift, *incrementality, relative_divisor=base_level)
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
        """Plot the parallel-trends diagnostic for our difference-in-differences.

        Plots the treated and control group averages over time alongside the
        parallel-trends counterfactual for the treated group. The counterfactual
        is the control series level-shifted by the pre-period gap between the two
        groups; in the pre-period it overlays the treated series when trends are
        parallel, and the post-period gap between the treated series and the
        counterfactual is the difference-in-differences estimand.

        Returns
        -------
        The parallel-trends plot summarizing the results.
        """
        if self.groupby_data is None:
            raise ValueError("groupby_data must not be None")
        if self.treatment_variable is None:
            raise ValueError("treatment_variable must not be None")
        treated = self.groupby_data.filter(nw.col(self.treatment_variable) == 1).sort(self.date_variable)
        control = self.groupby_data.filter(nw.col(self.treatment_variable) == 0).sort(self.date_variable)
        treated_dates = treated[self.date_variable].to_list()
        control_dates = control[self.date_variable].to_list()
        treated_y = treated[self.y_variable].to_numpy()
        control_y = control[self.y_variable].to_numpy()
        treated_pre = treated.filter(nw.col("treatment_period") == 0)[self.y_variable].to_numpy()
        control_pre = control.filter(nw.col("treatment_period") == 0)[self.y_variable].to_numpy()
        offset = treated_pre.mean() - control_pre.mean()
        counterfactual = control_y + offset
        fig = go.Figure(
            [
                go.Scatter(
                    x=treated_dates,
                    y=treated_y,
                    marker={"color": "blue"},
                    mode="lines",
                    name="Treated",
                ),
                go.Scatter(
                    x=control_dates,
                    y=control_y,
                    marker={"color": "green"},
                    mode="lines",
                    name="Control",
                ),
                go.Scatter(
                    x=control_dates,
                    y=counterfactual,
                    marker={"color": "red"},
                    mode="lines",
                    line={"dash": "dash"},
                    name="Counterfactual",
                ),
            ]
        )
        fig.add_vline(
            x=self.post_period,
            line_width=1,
            line_dash="dash",
            line_color="black",
        )
        fig.update_layout(
            title="Parallel Trends",
            xaxis_title=self.date_variable,
            yaxis_title=self.y_variable,
        )
        fig.show()
