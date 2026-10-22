"""Difference-in-differences method for geo-experiment causal inference."""

from typing import Any

import narwhals as nw
import numpy as np
import statsmodels.formula.api as smf
from narwhals.typing import IntoDataFrame
from scipy.linalg import toeplitz
from scipy.stats import t as t_dist
from tabulate import tabulate  # type: ignore

from GeoCausality._base import EconometricEstimator
from GeoCausality._plotting import ParallelTrendsPlotData, PlotBackend, check_backend, get_theme, render


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
            How to compute p-values and confidence intervals. ``"ols"`` uses
            standard errors of the interaction coefficient that allow AR(1)
            autocorrelation within each group's daily series (see
            ``_ar1_standard_error``). The other
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
        self._pt_control: np.ndarray | None = None

    def pre_process(self) -> "DiffinDiff":
        """Average the data by treatment, period and date and count post-period dates.

        Each group is collapsed to its per-geo *mean* outcome on each date. The
        2x2 difference-in-differences compares group means: with per-date sums,
        a common time shock enters the treated and control series scaled by
        their geo counts and only cancels when the two groups are the same size.
        The per-geo effect is scaled back up by the number of treated geos in
        ``generate``.

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
            .agg(nw.col(self.y_variable).mean())
            .sort([self.treatment_variable, "treatment_period", self.date_variable])
        )
        self.n_dates = self.groupby_data.filter(
            (nw.col(self.treatment_variable) == 1) & (nw.col("treatment_period") == 1)
        ).shape[0]
        return self

    def generate(self) -> "DiffinDiff":
        """Fit the OLS model and compute lift, confidence intervals and incrementality.

        The point estimate is always the interaction coefficient of the
        difference-in-differences OLS model on the group-mean series, i.e. the
        per-geo, per-date effect (``att``). ``lift`` and its CIs scale it by the
        number of treated geos, giving the per-date effect on the treated
        group's total like the other estimators. The p-value and confidence intervals
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
        frame = self.groupby_data.to_pandas()
        self.model = smf.ols(f"{self.y_variable} ~ {self.treatment_variable} * treatment_period", data=frame).fit()
        params = self.model.params
        n_treated = self._n_treated
        att = float(params[f"{self.treatment_variable}:treatment_period"])
        # ``test`` and ``control`` are each group's fitted post-period per-date
        # per-geo outcome; the bare ``treatment_period`` coefficient is only the
        # control group's pre-to-post change, and ``Intercept`` its pre-period level.
        control_post = float(params["Intercept"] + params["treatment_period"])
        self.results = dict(
            test=control_post + float(params[self.treatment_variable]) + att,
            control=control_post,
            att=att,
            lift=att * n_treated,
        )
        self.results["incrementality"] = self.results["lift"] * self.n_dates
        if self.inference == "ols":
            interaction = f"{self.treatment_variable}:treatment_period"
            se, df_resid = self._ar1_standard_error(frame, interaction)
            se *= n_treated
            if df_resid > 0 and np.isfinite(se) and se > 0:
                crit = float(t_dist.ppf(1 - self.alpha / 2, df_resid))
                lower = self.results["lift"] - crit * se
                upper = self.results["lift"] + crit * se
                p_value = float(2.0 * t_dist.sf(abs(self.results["lift"]) / se, df_resid))
            else:
                # e.g. a saturated 2x2: no residual degrees of freedom.
                lower = upper = p_value = float("nan")
            self.results["standard_error"] = se
            self.results["lift_ci_lower"] = lower
            self.results["lift_ci_upper"] = upper
            self.results["incrementality_ci_lower"] = lower * self.n_dates
            self.results["incrementality_ci_upper"] = upper * self.n_dates
            self.results["p_value"] = p_value
        else:
            actual_pre, prediction_pre, actual_post, prediction_post = self._counterfactual_from_parallel_trends()
            self.results.update(self._conformal_inference(actual_pre, prediction_pre, actual_post, prediction_post))
        return self

    def _ar1_standard_error(self, frame: Any, term: str) -> tuple[float, int]:
        """Compute the standard error of an OLS coefficient under AR(1) errors within each group.

        The regression runs on one per-date series per group, whose errors are
        typically autocorrelated, so iid standard errors are far too small
        (Bertrand, Duflo & Mullainathan, 2004). This keeps the OLS estimate and
        uses the sandwich ``(X'X)^-1 X' Sigma X (X'X)^-1``, where ``Sigma`` is
        block-diagonal over groups with an AR(1) correlation in date order. The
        AR(1) coefficient is pooled from the within-group residual
        autocorrelation, with the ``(1 + 3 rho) / T`` small-sample bias
        correction. Very persistent series can still be under-covered; use
        ``inference="conformal"`` for those.

        Parameters
        ----------
        frame : pandas DataFrame
            The regression data, sorted by group and then date.
        term : str
            The coefficient name.

        Returns
        -------
        A tuple of (standard error, residual degrees of freedom).
        """
        if self.treatment_variable is None:
            raise ValueError("treatment_variable must not be None")
        exog = np.asarray(self.model.model.exog, dtype=float)
        resid = np.asarray(self.model.resid, dtype=float)
        n_obs, n_params = exog.shape
        df_resid = n_obs - n_params
        if df_resid <= 0:
            return float("nan"), df_resid
        groups = frame[self.treatment_variable].to_numpy()
        blocks = [np.flatnonzero(groups == g) for g in np.unique(groups)]
        num = float(np.sum([resid[idx][1:] @ resid[idx][:-1] for idx in blocks]))
        den = float(np.sum([resid[idx] @ resid[idx] for idx in blocks]))
        rho = num / den if den > 0 else 0.0
        mean_len = n_obs / len(blocks)
        rho = float(np.clip(rho + (1 + 3 * rho) / mean_len, -0.99, 0.99))
        sigma2 = float(resid @ resid) / df_resid
        meat = np.zeros((n_params, n_params))
        for idx in blocks:
            corr = toeplitz(rho ** np.arange(idx.shape[0]))
            meat += exog[idx].T @ corr @ exog[idx]
        bread = np.linalg.pinv(exog.T @ exog)
        cov = sigma2 * bread @ meat @ bread
        j = list(self.model.model.exog_names).index(term)
        return float(np.sqrt(max(cov[j, j], 0.0))), df_resid

    def _counterfactual_from_parallel_trends(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build the treated series and its parallel-trends counterfactual.

        Mirrors the counterfactual drawn in ``plot()``: the treated group's
        per-geo mean outcome is the observed series, and the counterfactual is
        the control group's per-geo mean level-shifted by the pre-period gap
        between the two groups. Both are scaled by the number of treated geos so
        the residuals are on the treated total's scale, like ``lift``. Feeding
        these four series to the conformal engine gives per-period residuals
        whose mean equals the difference-in-differences estimand, so the
        conformal p-value and confidence intervals are centred on the same
        effect as the OLS coefficient.

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
        n = self._n_treated
        self._pt_control = n * np.concatenate([control_pre, control_post])
        return n * treated_pre, n * (control_pre + offset), n * treated_post, n * (control_post + offset)

    def _refit_counterfactual(self, y_full: np.ndarray, train: np.ndarray) -> np.ndarray | None:
        """Refit the parallel-trends level shift on a subset of periods.

        The counterfactual is the control series plus a constant gap, so the
        refit is the mean treated-minus-control gap over the training periods.

        Parameters
        ----------
        y_full : numpy array, shape (T,)
            The treated series over every period.
        train : numpy array of bool, shape (T,)
            The periods to fit on.

        Returns
        -------
        The counterfactual over every period, or None before it is built.
        """
        control = self._pt_control
        if control is None or control.shape[0] != y_full.shape[0] or not train.any():
            return None
        return control + float(np.mean((y_full - control)[train]))

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
        # The baseline is the treated group's counterfactual total over the
        # post-period: its per-geo pre-period level plus the control group's
        # pre-to-post change (Intercept + treated + period coefficients), not the
        # control group's level, scaled to all treated geos and post dates.
        params = self.model.params
        counterfactual_level = float(params["Intercept"] + params[self.treatment_variable] + params["treatment_period"])
        base_level = counterfactual_level * self._n_treated * self.n_dates
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
        elif lift == "roas":
            table_dict["Metric"] = ["ROAS"]
            table_dict["Lift Type"] = ["Incremental"]
            cells = self._format_lift_cells(lift, *self._get_roas())
        else:
            table_dict["Metric"] = ["Cost-per"]
            table_dict["Lift Type"] = ["Incremental"]
            cells = self._format_lift_cells(lift, *self._get_cost_per())
        table_dict["Lift"], table_dict[lo_key], table_dict[hi_key] = cells
        table_dict["p_value"] = [self.results["p_value"]]
        print(tabulate(table_dict, headers="keys", tablefmt="grid"))

    def plot(self, dark_mode: bool = False, backend: PlotBackend = "plotly") -> Any:
        """Plot the parallel-trends diagnostic for our difference-in-differences.

        Plots the treated and control group averages over time alongside the
        parallel-trends counterfactual for the treated group. The counterfactual
        is the control series level-shifted by the pre-period gap between the two
        groups; in the pre-period it overlays the treated series when trends are
        parallel, and the post-period gap between the treated series and the
        counterfactual is the difference-in-differences estimand.

        Parameters
        ----------
        dark_mode : bool, default=False
            Render with a dark background and lighter series colours.
        backend : {"plotly", "altair"}, default="plotly"
            Library to draw the figure with. ``"altair"`` needs the optional
            dependency (``pip install geocausality[altair]``).

        Returns
        -------
        plotly Figure or altair LayerChart
            The parallel-trends plot summarizing the results. Notebooks render
            it when it is the cell's last expression; elsewhere call its
            ``.show()`` (Plotly) or ``.save()`` (Altair).
        """
        check_backend(backend)
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
        data = ParallelTrendsPlotData(
            treated_dates=treated_dates,
            treated=treated_y,
            control_dates=control_dates,
            control=control_y,
            counterfactual=counterfactual,
            post_period=self.post_period,
            x_title=self.date_variable,
            y_title=self.y_variable,
        )
        return render(data, get_theme(dark_mode), backend)
