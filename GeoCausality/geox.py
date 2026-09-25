"""GeoX (time-based regression) method for geo-experiment causal inference."""

import warnings
from datetime import date as date_cls
from itertools import combinations
from math import ceil, comb
from typing import Any

import narwhals as nw
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
from narwhals.typing import IntoDataFrame
from plotly.subplots import make_subplots
from scipy.stats import t as t_dist
from tabulate import tabulate  # type: ignore

from GeoCausality._base import MLEstimator


class GeoX(MLEstimator):
    """Run a GeoX time-based regression experiment for our geo-test."""

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
        non_negative: bool = True,
        alternative: str = "two-sided",
    ) -> None:
        """Initialize the GeoX estimator.

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
        non_negative : bool, default=True
            When True, the regression slope is clamped to zero if the OLS
            estimate is negative. Matches Meridian GeoX's guardrail against
            nonsensical fits.
        alternative : str, default="two-sided"
            The sidedness of the hypothesis test. One of ``"two-sided"``,
            ``"greater"`` (treatment > control) or ``"less"`` (treatment <
            control).

        Notes
        -----
        Based on https://github.com/google/matched_markets?tab=readme-ov-file
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
        self.non_negative = non_negative
        alternative = alternative.casefold()
        if alternative not in ("two-sided", "greater", "less"):
            raise ValueError(f"alternative must be 'two-sided', 'greater' or 'less', got {alternative!r}")
        self.alternative = alternative
        self.intercept_test: Any = None
        self.prediction_pre: nw.DataFrame | None = None
        self.prediction_post: nw.DataFrame | None = None
        self.dates: list[Any] | None = None

    def generate(self, rescale: float = 1.0) -> "GeoX":
        """Fit the control-on-test regression and compute the counterfactual and lift.

        Parameters
        ----------
        rescale : float, default=1.0
            An additional scaling factor for the cumulative delta distribution.

        Returns
        -------
        GeoX
            Itself, so it can be chained with summarize().
        """
        if self.pre_control is None:
            raise ValueError("pre_control must not be None")
        if self.pre_test is None:
            raise ValueError("pre_test must not be None")
        if self.post_control is None:
            raise ValueError("post_control must not be None")
        if self.post_test is None:
            raise ValueError("post_test must not be None")
        pre_control_np = self.pre_control[self.y_variable].to_numpy()
        pre_test_np = self.pre_test[self.y_variable].to_numpy().flatten()
        intercept_train = sm.add_constant(pre_control_np)

        # --- Validation split: hold out the last `post_len` pre-period rows ---
        post_len = len(self.post_control)
        val_rmse: float | None = None
        if len(pre_control_np) > post_len + 2:
            train_x = intercept_train[:-post_len]
            train_y = pre_test_np[:-post_len]
            val_x = intercept_train[-post_len:]
            val_y = pre_test_np[-post_len:]
            val_model = sm.OLS(train_y, train_x).fit()
            if self.non_negative and val_model.params[1] < 0:
                val_model.params[1] = 0.0
            val_pred = val_model.predict(val_x)
            val_rmse = float(np.sqrt(np.mean((val_y - val_pred) ** 2)))

        # --- Full pre-period fit ---
        self.model = sm.OLS(pre_test_np, intercept_train).fit()
        if self.non_negative and self.model.params[1] < 0:
            self.model.params[1] = 0.0

        self.intercept_test = sm.add_constant(self.post_control[self.y_variable].to_numpy())
        model_summary = self.model.get_prediction(self.intercept_test).summary_frame(alpha=self.alpha)
        # Add counterfactual column via pandas (we're at a statsmodels boundary already)
        post_test_pd = self.post_test.to_pandas()
        post_test_pd = post_test_pd.assign(counterfactual=model_summary["mean"].values)
        self.post_test = nw.from_native(post_test_pd, eager_only=True)
        actual = self.post_test[self.y_variable].to_numpy()
        counterfactual = self.post_test["counterfactual"].to_numpy()
        incrementality = actual - counterfactual
        ci_lower_series = actual - model_summary["obs_ci_upper"].values
        ci_upper_series = actual - model_summary["obs_ci_lower"].values
        self.results = {
            "date": self.test_dates,
            "test": actual,
            "control": self.post_control[self.y_variable].to_numpy(),
            "counterfactual": counterfactual,
            "counterfactual_ci_lower": model_summary["obs_ci_lower"],
            "counterfactual_ci_upper": model_summary["obs_ci_upper"],
            "incrementality": incrementality,
            "incrementality_ci_lower": ci_lower_series,
            "incrementality_ci_upper": ci_upper_series,
        }
        self.results["cumulative_incrementality"] = np.cumsum(self.results["incrementality"])
        ci_dict = self._get_cumulative_cis(rescale)
        self.results["cumulative_incrementality_ci_lower"] = ci_dict["cumulative_ci_lower"]
        self.results["cumulative_incrementality_ci_upper"] = ci_dict["cumulative_ci_upper"]
        self.results["p_value"] = ci_dict["p_value"]
        if val_rmse is not None:
            self.results["validation_rmse"] = val_rmse

        # --- Log-ratio percent lift CIs ---
        safe_cf = np.where(counterfactual > 0, counterfactual, np.nan)
        log_ratio = np.log(actual / safe_cf)
        log_rmse = float(np.sqrt(self.model.scale)) / float(np.nanmean(safe_cf))
        q = t_dist.ppf(1 - self.alpha / 2, self.model.df_resid)
        self.results["percent_lift"] = np.where(np.isnan(log_ratio), np.nan, np.exp(log_ratio) - 1)
        self.results["percent_lift_ci_lower"] = np.where(
            np.isnan(log_ratio), np.nan, np.exp(log_ratio - q * log_rmse) - 1
        )
        self.results["percent_lift_ci_upper"] = np.where(
            np.isnan(log_ratio), np.nan, np.exp(log_ratio + q * log_rmse) - 1
        )
        return self

    def summarize(self, lift: str) -> None:
        """Print a tabulated summary of the GeoX results.

        Parameters
        ----------
        lift : str
            The kind of lift to report. One of ``"absolute"``, ``"relative"``,
            ``"incremental"``, ``"cost-per"``, ``"revenue"`` or ``"roas"``.
        """
        if self.results is None:
            raise ValueError("results must not be None")
        lift = self._validate_lift(lift)
        ci_alpha = self._get_ci_print()
        lo_key, hi_key = f"{ci_alpha} Lower CI", f"{ci_alpha} Upper CI"
        baseline = np.sum(self.results["counterfactual"])
        cumulative = (
            self.results["cumulative_incrementality"][-1],
            self.results["cumulative_incrementality_ci_lower"][-1],
            self.results["cumulative_incrementality_ci_upper"][-1],
        )
        table_dict: dict[str, list[Any]] = {
            "Variant": [np.sum(self.results["test"])],
            "Baseline": [baseline],
        }
        if lift in ("incremental", "absolute"):
            table_dict["Metric"] = [self.y_variable]
            table_dict["Lift Type"] = ["Incremental"]
            cells = self._format_lift_cells(lift, *cumulative)
        elif lift == "relative":
            table_dict["Metric"] = [self.y_variable]
            table_dict["Lift Type"] = ["Relative"]
            cells = self._format_lift_cells(lift, *cumulative, relative_divisor=baseline)
        elif lift == "revenue":
            table_dict["Metric"] = ["Revenue"]
            table_dict["Lift Type"] = ["Incremental"]
            cells = self._format_lift_cells(lift, *cumulative)
        elif lift == "roas":
            table_dict["Metric"] = ["ROAS"]
            table_dict["Lift Type"] = ["Incremental"]
            cells = self._format_lift_cells(lift, *self._get_roas())
        else:
            table_dict["Metric"] = ["Cost-per"]
            table_dict["Lift Type"] = ["Incremental"]
            cells = self._format_lift_cells(lift, *self._get_cost_per())
        table_dict["Lift"], table_dict[lo_key], table_dict[hi_key] = cells
        table_dict["p_value"] = [self.results["p_value"][-1]]
        print(tabulate(table_dict, headers="keys", tablefmt="grid"))

    def _get_roas(self) -> tuple[float, float, float]:
        if self.results is None:
            raise ValueError("results must not be None")
        incr = self.results["cumulative_incrementality"][-1]
        roas_lift = self._safe_ratio(incr * self.msrp, self.spend)
        ci_lower = self.results["cumulative_incrementality_ci_lower"][-1]
        roas_ci_lower = self._safe_ratio(ci_lower * self.msrp, self.spend)
        ci_upper = self.results["cumulative_incrementality_ci_upper"][-1]
        roas_ci_upper = self._safe_ratio(ci_upper * self.msrp, self.spend)
        return roas_lift, roas_ci_lower, roas_ci_upper

    def _get_cost_per(self) -> tuple[float, float, float]:
        if self.results is None:
            raise ValueError("results must not be None")
        lift = ceil(self.results["cumulative_incrementality"][-1])
        cost_per = self.spend / lift if lift > 0 else np.inf
        ci_upper = ceil(self.results["cumulative_incrementality_ci_upper"][-1])
        cost_per_lower = self.spend / ci_upper if ci_upper > 0 else np.inf
        ci_lower = ceil(self.results["cumulative_incrementality_ci_lower"][-1])
        cost_per_upper = self.spend / ci_lower if ci_lower > 0 else np.inf
        return cost_per, cost_per_lower, cost_per_upper

    def _cumulative_distribution(self, rescale: float = 1.0) -> Any:
        """Calculate the shifted distribution of our cumulative data.

        Parameters
        ----------
        rescale : float, default=1.0
            An additional scaling factor for our delta

        Returns
        -------
        Our shifted t-distribution, as explained in Section 9.1 of https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45950.pdf

        Notes
        -----
        Taken from https://github.com/google/matched_markets/blob/master/matched_markets/methodology/tbr.py
        """
        if self.post_control is None:
            raise ValueError("post_control must not be None")
        if self.results is None:
            raise ValueError("results must not be None")
        test_len = len(self.post_control)
        one_to_t = np.arange(1, test_len + 1)
        one_to_t = one_to_t.reshape(test_len, 1)
        control_arr = np.array(self.results["control"])
        control_matrix = sm.add_constant(control_arr)
        cumulative_control_t = np.cumsum(control_matrix, axis=0) / one_to_t
        param_covariance = np.array(self.model.cov_params())
        var_params_list: list[Any] = []
        for t in range(test_len):
            # Sum of parameter variance terms from eqn 5 of Kerman 2017.
            var_t = cumulative_control_t[t] @ param_covariance @ cumulative_control_t[t].T
            var_params_list.append(var_t)
        var_params = np.array(var_params_list).reshape(test_len, 1)
        var_from_params = var_params * pow(one_to_t, 2)
        sigma_square = self.model.scale
        var_from_observations = one_to_t * sigma_square
        delta_mean = rescale * np.array(self.results["cumulative_incrementality"]).flatten()
        delta_var = var_from_params + var_from_observations
        delta_scale = rescale * np.sqrt(delta_var).flatten()
        delta_df = self.model.df_resid
        return t_dist(delta_df, loc=delta_mean, scale=delta_scale)

    def _get_cumulative_cis(self, rescale: float = 1.0) -> dict[str, np.ndarray]:
        """Calculate the confidence intervals and p-values from our shifted T-distribution.

        Parameters
        ----------
        rescale : float, default=1.0
            An additional scaling factor for our delta

        Returns
        -------
        ci_dict : dict
            A dictionary containing our confidence intervals as well as p-values.
        """
        if self.post_control is None:
            raise ValueError("post_control must not be None")
        delta = self._cumulative_distribution(rescale=rescale)
        test_len = len(self.post_control)
        one_sided = delta.cdf(0.0).reshape(test_len)
        if self.alternative == "greater":
            ci_lower = delta.ppf(self.alpha).reshape(test_len)
            ci_upper = np.full(test_len, np.inf)
            p_value = one_sided
        elif self.alternative == "less":
            ci_lower = np.full(test_len, -np.inf)
            ci_upper = delta.ppf(1 - self.alpha).reshape(test_len)
            p_value = 1.0 - one_sided
        else:
            ci_lower = delta.ppf(self.alpha / 2).reshape(test_len)
            ci_upper = delta.ppf(1 - self.alpha / 2).reshape(test_len)
            p_value = 2.0 * np.minimum(one_sided, 1.0 - one_sided)
        ci_dict = {
            "cumulative_ci_lower": ci_lower,
            "cumulative_ci_upper": ci_upper,
            "p_value": p_value,
        }
        return ci_dict

    def placebo_test(
        self,
        n_placebos: int = 500,
        min_placebo_r2: float = 0.6,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Run a placebo permutation test using only the control geos.

        Splits the control geos into pseudo-treatment and pseudo-control groups,
        fits TBR on each split, and compares the real experiment's t-statistic
        against the empirical null distribution. This validates that the model
        is well-calibrated and provides a non-parametric p-value.

        Parameters
        ----------
        n_placebos : int, default=500
            Maximum number of placebo splits to evaluate. When the full
            enumeration of same-sized splits is smaller, all are used.
        min_placebo_r2 : float, default=0.6
            Minimum pre-period R² for a placebo to be included. Splits with
            poor fit are uninformative and are discarded.
        seed : int, default=0
            Seed for the placebo sampler when the full enumeration exceeds
            ``n_placebos``.

        Returns
        -------
        dict
            ``placebo_t_stats`` (array of placebo t-statistics that passed the
            R² filter), ``real_t_stat`` (the real experiment's t-statistic),
            ``empirical_p_value`` (fraction of placebos at least as extreme),
            ``n_placebos_passed`` (how many passed the R² filter), and
            ``n_placebos_total`` (how many were evaluated).
        """
        if self.results is None:
            raise ValueError("Call generate() before placebo_test()")

        treatment_var: str = self.treatment_variable or "is_test"
        if self.test_geos is not None:
            test_geo_list = list(self.test_geos)
        else:
            test_geo_list = self.data.filter(nw.col(treatment_var) == 1)[self.geo_variable].unique().to_list()
        if self.control_geos is not None:
            control_geos = list(self.control_geos)
        else:
            control_geos = self.data.filter(nw.col(treatment_var) == 0)[self.geo_variable].unique().to_list()

        n_test = len(test_geo_list)
        if len(control_geos) < 2:
            raise ValueError("Need at least 2 control geos for a placebo test")

        split_size = min(n_test, len(control_geos) - 1)
        total_splits = comb(len(control_geos), split_size)
        rng = np.random.default_rng(seed)

        if total_splits <= n_placebos:
            splits = list(combinations(control_geos, split_size))
        else:
            seen: set[tuple[str, ...]] = set()
            while len(seen) < n_placebos:
                pick = tuple(sorted(rng.choice(control_geos, size=split_size, replace=False)))
                seen.add(pick)
            splits = list(seen)

        date_str = nw.col(self.date_variable).cast(nw.String)
        control_data = self.data.filter(nw.col(self.geo_variable).is_in(control_geos))

        placebo_stats: list[float] = []
        for pseudo_test in splits:
            pseudo_test_set = set(pseudo_test)
            is_pseudo_test = nw.col(self.geo_variable).is_in(list(pseudo_test_set))

            pre_pseudo_test = (
                control_data.filter((date_str <= self.pre_period) & is_pseudo_test)
                .group_by(self.date_variable)
                .agg(nw.col(self.y_variable).sum())
                .sort(self.date_variable)
            )
            pre_pseudo_ctrl = (
                control_data.filter((date_str <= self.pre_period) & ~is_pseudo_test)
                .group_by(self.date_variable)
                .agg(nw.col(self.y_variable).sum())
                .sort(self.date_variable)
            )
            post_pseudo_test = (
                control_data.filter((date_str >= self.post_period) & is_pseudo_test)
                .group_by(self.date_variable)
                .agg(nw.col(self.y_variable).sum())
                .sort(self.date_variable)
            )
            post_pseudo_ctrl = (
                control_data.filter((date_str >= self.post_period) & ~is_pseudo_test)
                .group_by(self.date_variable)
                .agg(nw.col(self.y_variable).sum())
                .sort(self.date_variable)
            )

            x_pre = pre_pseudo_ctrl[self.y_variable].to_numpy()
            y_pre = pre_pseudo_test[self.y_variable].to_numpy().flatten()
            x_post = post_pseudo_ctrl[self.y_variable].to_numpy()
            y_post = post_pseudo_test[self.y_variable].to_numpy().flatten()

            model = sm.OLS(y_pre, sm.add_constant(x_pre)).fit()
            if self.non_negative and model.params[1] < 0:
                model.params[1] = 0.0

            if model.rsquared < min_placebo_r2:
                continue

            pred_post = model.predict(sm.add_constant(x_post))
            cum_delta = float(np.sum(y_post - pred_post))
            post_len = len(x_post)
            one_to_t = np.arange(1, post_len + 1).reshape(post_len, 1)
            ctrl_matrix = sm.add_constant(np.array(x_post.flatten()))
            cum_ctrl = np.cumsum(ctrl_matrix, axis=0) / one_to_t
            var_t = cum_ctrl[-1] @ np.array(model.cov_params()) @ cum_ctrl[-1].T
            var_from_params = var_t * post_len**2
            var_from_obs = post_len * model.scale
            scale = float(np.sqrt(var_from_params + var_from_obs))
            if scale > 0:
                placebo_stats.append(cum_delta / scale)

        # Real experiment's t-statistic at the final post-period date.
        real_cum = float(self.results["cumulative_incrementality"][-1])
        delta_dist = self._cumulative_distribution()
        real_scale = float(delta_dist.kwds["scale"][-1])
        real_t = real_cum / real_scale if real_scale > 0 else 0.0

        placebo_arr = np.asarray(placebo_stats, dtype=float)
        if len(placebo_arr) > 0:
            n_extreme = np.sum(np.abs(placebo_arr) >= abs(real_t))
            empirical_p = float((n_extreme + 1) / (len(placebo_arr) + 1))
        else:
            empirical_p = np.nan

        result = {
            "placebo_t_stats": placebo_arr,
            "real_t_stat": real_t,
            "empirical_p_value": empirical_p,
            "n_placebos_passed": len(placebo_arr),
            "n_placebos_total": len(splits),
        }
        if self.results is not None:
            self.results["placebo"] = result
        return result

    def randomization_test(
        self,
        *,
        statistic: str = "avg_lift",
        seed: int | None = None,
        max_placebos: int | None = None,
        n_jobs: int = 1,
    ) -> dict[str, Any]:
        """Abadie-style placebo permutation test for GeoX.

        For each control geo, temporarily treat it as the single treated unit
        while the remaining control geos form the donor pool.  The full
        ``pre_process().generate()`` pipeline is re-run for every placebo geo,
        producing a null distribution of treatment effect statistics.

        Parameters
        ----------
        statistic : {"avg_lift", "sum_lift"}, default "avg_lift"
            The test statistic.

            * ``"avg_lift"`` — mean of the per-period incrementality.
            * ``"sum_lift"`` — total cumulative incrementality.
        seed : int, optional
            Seed controlling the random subset when ``max_placebos`` is set.
        max_placebos : int, optional
            Cap the number of placebo geos to run.  When the control pool is
            very large, this draws a random subset.
        n_jobs : int, default 1
            Number of parallel workers for the placebo loop.  ``-1`` uses all
            available cores.

        Returns
        -------
        dict with keys ``observed_statistic``, ``placebo_statistics``,
        ``placebo_geos``, ``p_value``, ``n_placebos``, ``n_failed``, and
        ``statistic``.
        """
        current_results = self.results
        if current_results is None:
            raise ValueError("Call generate() before randomization_test()")
        valid_stats = ("avg_lift", "sum_lift")
        if statistic not in valid_stats:
            raise ValueError(f"statistic must be one of {valid_stats}, got {statistic!r}")

        observed = self._ri_statistic(current_results, statistic)

        treatment_var: str = self.treatment_variable or "is_test"
        if self.test_geos is not None:
            test_geo_set: set[str] = set(self.test_geos)
        else:
            test_geo_set = set(self.data.filter(nw.col(treatment_var) == 1)[self.geo_variable].unique().to_list())
        if self.control_geos is not None:
            donor_geos: list[str] = list(self.control_geos)
        else:
            donor_geos = [g for g in sorted(self.data[self.geo_variable].unique().to_list()) if g not in test_geo_set]

        if not donor_geos:
            raise ValueError("No control geos available for placebo permutation")

        if max_placebos is not None and max_placebos < len(donor_geos):
            rng = np.random.default_rng(seed)
            donor_geos = list(rng.choice(donor_geos, size=max_placebos, replace=False))

        native_data = self.data.to_native()
        all_geos = sorted(self.data[self.geo_variable].unique().to_list())
        geo_var = self.geo_variable
        date_var = self.date_variable
        pre_per = self.pre_period
        post_per = self.post_period
        y_var = self.y_variable
        alpha_val = self.alpha
        non_neg = self.non_negative
        alt = self.alternative

        def _fit_placebo(placebo_geo: str) -> tuple[str, float] | None:
            placebo_control = [g for g in all_geos if g != placebo_geo and g not in test_geo_set]
            try:
                m = GeoX(
                    native_data,
                    geo_variable=geo_var,
                    test_geos=[placebo_geo],
                    control_geos=placebo_control,
                    date_variable=date_var,
                    pre_period=pre_per,
                    post_period=post_per,
                    y_variable=y_var,
                    alpha=alpha_val,
                    non_negative=non_neg,
                    alternative=alt,
                )
                m.pre_process().generate()
                r = m.results
                if r is None:
                    return None
                return (placebo_geo, self._ri_statistic(r, statistic))
            except Exception:
                return None

        if n_jobs == 1:
            raw = [_fit_placebo(g) for g in donor_geos]
        else:
            from joblib import Parallel, delayed

            raw = Parallel(n_jobs=n_jobs)(delayed(_fit_placebo)(g) for g in donor_geos)

        placebo_stats: list[float] = []
        placebo_geos: list[str] = []
        n_failed = 0
        for r in raw:
            if r is None:
                n_failed += 1
            else:
                placebo_geos.append(r[0])
                placebo_stats.append(r[1])

        if not placebo_stats:
            warnings.warn("All placebo runs failed; cannot compute a p-value", stacklevel=2)
            result: dict[str, Any] = {
                "observed_statistic": observed,
                "placebo_statistics": [],
                "placebo_geos": [],
                "p_value": float("nan"),
                "n_placebos": 0,
                "n_failed": n_failed,
                "statistic": statistic,
            }
            current_results["randomization_test"] = result
            return result

        n_placebos = len(placebo_stats)
        n_extreme = sum(1 for s in placebo_stats if abs(s) >= abs(observed))
        p_value = (n_extreme + 1) / (n_placebos + 1)

        result = {
            "observed_statistic": observed,
            "placebo_statistics": placebo_stats,
            "placebo_geos": placebo_geos,
            "p_value": p_value,
            "n_placebos": n_placebos,
            "n_failed": n_failed,
            "statistic": statistic,
        }
        current_results["randomization_test"] = result
        return result

    @staticmethod
    def _ri_statistic(results: dict[str, Any], statistic: str) -> float:
        """Extract the test statistic from a GeoX results dict."""
        incr = np.asarray(results["incrementality"], dtype=float).ravel()
        if statistic == "avg_lift":
            return float(np.mean(incr))
        if statistic == "sum_lift":
            return float(np.sum(incr))
        raise ValueError(f"Unknown statistic: {statistic!r}")

    def plot(self) -> None:
        """Plot our actual results, our counterfactual, the pointwise difference and cumulative difference.

        Returns
        -------
        Our three plots determining the results
        """
        if self.pre_control is None:
            raise ValueError("pre_control must not be None")
        if self.pre_test is None:
            raise ValueError("pre_test must not be None")
        if self.post_control is None:
            raise ValueError("post_control must not be None")
        if self.post_test is None:
            raise ValueError("post_test must not be None")
        if self.results is None:
            raise ValueError("results must not be None")
        self.dates = sorted(self.data[self.date_variable].unique().to_list())
        # Compare on ``datetime.date`` so the filter is robust to the date column's
        # backend type (str, ``datetime.date``, ``datetime``, or pandas ``Timestamp``).
        post_period_date = date_cls.fromisoformat(self.post_period)

        def _as_date(value: Any) -> Any:
            if isinstance(value, str):
                return date_cls.fromisoformat(value[:10])
            return value.date() if hasattr(value, "date") else value

        marketing_start = [date for date in self.dates if _as_date(date) >= post_period_date]
        control_data = nw.concat([self.pre_control, self.post_control])
        counterfactual = self.model.predict(sm.add_constant(control_data[self.y_variable].to_numpy()))
        total_fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(
                "Expected vs Counterfactual",
                "Pointwise Difference",
                "Cumulative Difference",
            ),
        )
        top_fig = go.Figure(
            [
                go.Scatter(
                    x=self.dates,
                    y=np.concatenate(
                        [self.pre_test[self.y_variable].to_numpy(), self.post_test[self.y_variable].to_numpy()]
                    ),
                    marker={"color": "blue"},
                    mode="lines",
                    name="Actual",
                ),
                go.Scatter(
                    x=self.dates,
                    y=counterfactual,
                    marker={"color": "red"},
                    mode="lines",
                    name="Counterfactual",
                ),
                go.Scatter(
                    name="Counterfactual Upper Bound",
                    x=marketing_start,
                    y=self.results["counterfactual_ci_upper"],
                    mode="lines",
                    marker=dict(color="red"),
                    line=dict(width=0),
                    showlegend=False,
                ),
                go.Scatter(
                    name="Counterfactual Lower Bound",
                    x=marketing_start,
                    y=self.results["counterfactual_ci_lower"],
                    marker=dict(color="red"),
                    line=dict(width=0),
                    mode="lines",
                    fillcolor="rgba(68, 68, 68, 0.3)",
                    fill="tonexty",
                    showlegend=False,
                ),
            ]
        )
        residuals = (
            np.concatenate([self.pre_test[self.y_variable].to_numpy(), self.post_test[self.y_variable].to_numpy()])
            - counterfactual
        )
        middle_fig = go.Figure(
            [
                go.Scatter(
                    x=self.dates,
                    y=residuals,
                    marker={"color": "purple"},
                    mode="lines",
                    name="Residuals",
                ),
                go.Scatter(
                    name="Pointwise Difference Upper Bound",
                    x=marketing_start,
                    y=self.results["incrementality_ci_upper"],
                    mode="lines",
                    marker=dict(color="purple"),
                    line=dict(width=0),
                    showlegend=False,
                ),
                go.Scatter(
                    name="Pointwise Difference Lower Bound",
                    x=marketing_start,
                    y=self.results["incrementality_ci_lower"],
                    marker=dict(color="purple"),
                    line=dict(width=0),
                    mode="lines",
                    fillcolor="rgba(68, 68, 68, 0.3)",
                    fill="tonexty",
                    showlegend=False,
                ),
            ]
        )
        cum_resids = self.results["cumulative_incrementality"]
        bottom_fig = go.Figure(
            [
                go.Scatter(
                    x=marketing_start,
                    y=cum_resids,
                    marker={"color": "orange"},
                    mode="lines",
                    name="Cumulative Incrementality",
                ),
                go.Scatter(
                    name="Cumulative Difference Upper Bound",
                    x=marketing_start,
                    y=self.results["cumulative_incrementality_ci_upper"],
                    mode="lines",
                    marker=dict(color="orange"),
                    line=dict(width=0),
                    showlegend=False,
                ),
                go.Scatter(
                    name="Cumulative Difference Lower Bound",
                    x=marketing_start,
                    y=self.results["cumulative_incrementality_ci_lower"],
                    marker=dict(color="orange"),
                    line=dict(width=0),
                    mode="lines",
                    fillcolor="rgba(68, 68, 68, 0.3)",
                    fill="tonexty",
                    showlegend=False,
                ),
            ]
        )
        figures = [top_fig, middle_fig, bottom_fig]
        for i, figure in enumerate(figures):
            for trace_data in figure.data:
                total_fig.add_trace(trace_data, row=i + 1, col=1)
            total_fig.add_vline(
                x=self.post_period,
                line_width=1,
                line_dash="dash",
                line_color="black",
                row=i + 1,
                col=1,
            )
        total_fig.show()
