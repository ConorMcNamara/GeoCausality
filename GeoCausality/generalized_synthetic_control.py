"""Generalized Synthetic Control method for geo-experiment causal inference."""

from datetime import date as date_cls
from math import ceil
from typing import Any

import narwhals as nw
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from narwhals.typing import IntoDataFrame
from plotly.subplots import make_subplots
from tabulate import tabulate  # type: ignore

from GeoCausality._base import EconometricEstimator
from GeoCausality.utils import HoldoutSplitter


class GeneralizedSyntheticControl(EconometricEstimator):
    """Run generalized synthetic control (interactive fixed effects) for our geo-test.

    Where classic synthetic control builds the counterfactual as a convex blend of
    donor geos, the generalized method of Xu (2017) models the outcome with an
    *interactive fixed effects* (latent factor) structure
    ``Y_it = lambda_i' f_t + eps_it``: a small number of latent time factors
    ``f_t`` shared across geos, each weighted by a geo-specific loading
    ``lambda_i``. The factors are learned from the control geos (which are never
    treated), the treated unit's loadings are recovered from its pre-period fit to
    those factors, and the counterfactual is the loadings projected through the
    factors over the post-period. This relaxes the parallel-trends assumption of
    two-way fixed effects, to which it reduces when the factor structure is
    trivial.
    """

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
        n_factors: int | None = None,
        max_factors: int = 5,
        holdout_len: int = 1,
        conformal_q: float = 1.0,
    ) -> None:
        """Initialize the generalized synthetic control estimator.

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
        n_factors : int, optional
            The number of latent factors to use. If None, it is selected by
            cross-validation on the pre-period over ``range(0, max_factors + 1)``.
        max_factors : int, default=5
            The largest number of factors considered during cross-validation.
        holdout_len : int, default=1
            The block length held out on each cross-validation iteration.
        conformal_q : float, default=1.0
            The exponent of the moving-block test statistic used for conformal
            inference (p-values and confidence intervals).

        Notes
        -----
        Based on Xu, Yiqing. "Generalized Synthetic Control Method: Causal
        Inference with Interactive Fixed Effects Models." Political Analysis 25.1
        (2017): 57-76.
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
        if n_factors is not None and n_factors < 0:
            raise ValueError(f"n_factors must be non-negative, got {n_factors}")
        self.n_factors = n_factors
        self.max_factors = max_factors
        self.holdout_len = holdout_len
        self.conformal_q = conformal_q
        self.dates: list[Any] | None = None
        self.n_factors_selected: int | None = None
        self.prediction_pre: nw.DataFrame | None = None
        self.prediction_post: nw.DataFrame | None = None
        self.actual_pre: nw.DataFrame | None = None
        self.actual_post: nw.DataFrame | None = None

    def pre_process(self) -> "GeneralizedSyntheticControl":
        """Aggregate the treated series and record the sorted date axis.

        Returns
        -------
        GeneralizedSyntheticControl
            Itself, so it can be chained with generate().
        """
        super().pre_process()
        self.dates = sorted(self.data[self.date_variable].unique().to_list())
        return self

    def generate(self) -> "GeneralizedSyntheticControl":
        """Fit the factor model and build the counterfactual, lift and inference.

        Returns
        -------
        GeneralizedSyntheticControl
            Itself, so it can be chained with summarize().
        """
        assert self.treatment_variable is not None
        # Treated series, aggregated across test geos, split into pre / post.
        self.actual_pre = (
            self.data.filter((nw.col(self.treatment_variable) == 1) & (nw.col("treatment_period") == 0))
            .select([self.y_variable, self.date_variable])
            .group_by(self.date_variable)
            .agg(nw.col(self.y_variable).sum())
            .sort(self.date_variable)
        )
        self.actual_post = (
            self.data.filter((nw.col(self.treatment_variable) == 1) & (nw.col("treatment_period") == 1))
            .select([self.y_variable, self.date_variable])
            .group_by(self.date_variable)
            .agg(nw.col(self.y_variable).sum())
            .sort(self.date_variable)
        )
        # Control outcome matrix over all periods: rows = dates, cols = geos.
        control_all = (
            self.data.filter(nw.col(self.treatment_variable) == 0)
            .select([self.y_variable, self.date_variable, self.geo_variable])
            .group_by([self.date_variable, self.geo_variable])
            .agg(nw.col(self.y_variable).sum())
            .sort([self.date_variable, self.geo_variable])
        )
        control_pivot = nw.from_native(
            control_all.to_native().pivot(on=self.geo_variable, index=self.date_variable, values=self.y_variable),
            eager_only=True,
        ).sort(self.date_variable)
        y0 = control_pivot.drop(self.date_variable).to_numpy()  # T x N
        n_pre = self.actual_pre.shape[0]
        y1_all = np.concatenate(
            [self.actual_pre[self.y_variable].to_numpy(), self.actual_post[self.y_variable].to_numpy()]
        )

        prediction_all = self._fit_factor_model(y0, y1_all, n_pre)

        self.prediction_pre = nw.from_native(
            pl.DataFrame(
                {
                    self.date_variable: self.actual_pre[self.date_variable].to_native(),
                    self.y_variable: prediction_all[:n_pre],
                }
            ),
            eager_only=True,
        )
        self.prediction_post = nw.from_native(
            pl.DataFrame(
                {
                    self.date_variable: self.actual_post[self.date_variable].to_native(),
                    self.y_variable: prediction_all[n_pre:],
                }
            ),
            eager_only=True,
        )
        self.results = {
            "test": self.actual_post,
            "counterfactual": self.prediction_post,
            "lift": self.actual_post[self.y_variable].to_numpy() - self.prediction_post[self.y_variable].to_numpy(),
            "n_factors": self.n_factors_selected,
        }
        self.results["incrementality"] = float(np.sum(self.results["lift"]))
        self.results.update(
            self._conformal_inference(
                self.actual_pre[self.y_variable].to_numpy(),
                self.prediction_pre[self.y_variable].to_numpy(),
                self.actual_post[self.y_variable].to_numpy(),
                self.prediction_post[self.y_variable].to_numpy(),
                q=self.conformal_q,
            )
        )
        return self

    def _fit_factor_model(self, y0: np.ndarray, y1_all: np.ndarray, n_pre: int) -> np.ndarray:
        """Fit the interactive-fixed-effects counterfactual for the treated series.

        Factors are the leading left singular vectors of the column-centred
        control matrix (estimated from controls over all periods, since controls
        are never treated). An intercept plus the chosen number of factors are fit
        to the treated unit's pre-period by least squares, and the resulting
        loadings are projected through the factors over every period.

        Parameters
        ----------
        y0 : numpy array, shape (T, N)
            Control outcome matrix, rows = dates, cols = control geos.
        y1_all : numpy array, shape (T,)
            Treated series over all periods (pre then post).
        n_pre : int
            The number of pre-period dates.

        Returns
        -------
        The counterfactual treated series over all periods, shape (T,).
        """
        # Column-centre so the factors capture common temporal dynamics; the
        # intercept added below carries the treated unit's level.
        y0_centered = y0 - y0.mean(axis=0, keepdims=True)
        u, _, _ = np.linalg.svd(y0_centered, full_matrices=False)

        max_r = max(0, min(self.max_factors, u.shape[1], n_pre - self.holdout_len - 1))
        if self.n_factors is not None:
            r = min(self.n_factors, max_r)
        else:
            r = self._select_n_factors(u[:n_pre], y1_all[:n_pre], max_r)
        self.n_factors_selected = r

        factors = self._design(u, r)
        beta = self._ols(factors[:n_pre], y1_all[:n_pre])
        return factors @ beta

    @staticmethod
    def _design(u: np.ndarray, r: int) -> np.ndarray:
        """Assemble the design matrix: an intercept plus ``r`` leading factors.

        Parameters
        ----------
        u : numpy array, shape (T, k)
            Left singular vectors of the centred control matrix.
        r : int
            The number of leading factors to include.

        Returns
        -------
        The design matrix of shape (T, r + 1).
        """
        intercept = np.ones((u.shape[0], 1))
        if r == 0:
            return intercept
        return np.column_stack([intercept, u[:, :r]])

    @staticmethod
    def _ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Least-squares solution, robust to rank deficiency.

        Parameters
        ----------
        x : numpy array, shape (n, p)
            Design matrix.
        y : numpy array, shape (n,)
            Target vector.

        Returns
        -------
        The coefficient vector of shape (p,).
        """
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        return beta

    def _select_n_factors(self, u_pre: np.ndarray, y1_pre: np.ndarray, max_r: int) -> int:
        """Choose the factor count by cross-validation on the pre-period.

        For each candidate factor count, a moving block of the pre-period is held
        out (via :class:`~GeoCausality.utils.HoldoutSplitter`), the loadings are
        fit on the remainder and scored on the block; the count with the smallest
        mean held-out error wins.

        Parameters
        ----------
        u_pre : numpy array, shape (n_pre, k)
            Pre-period rows of the control left singular vectors.
        y1_pre : numpy array, shape (n_pre,)
            Treated series over the pre-period.
        max_r : int
            The largest factor count to consider.

        Returns
        -------
        The selected factor count.
        """
        best_r, best_err = 0, np.inf
        for r in range(max_r + 1):
            design = self._design(u_pre, r)
            df = pd.DataFrame(design)
            ser = pd.Series(y1_pre)
            errors, folds = 0.0, 0
            for x_train, x_holdout, y_train, y_holdout in HoldoutSplitter(df, ser, self.holdout_len):
                if x_train.shape[0] <= x_train.shape[1]:
                    continue  # underdetermined fold; skip
                beta = self._ols(x_train.to_numpy(), y_train.to_numpy())
                pred = x_holdout.to_numpy() @ beta
                errors += float(np.sum((y_holdout.to_numpy() - pred) ** 2))
                folds += 1
            if folds == 0:
                continue
            mse = errors / folds
            if mse < best_err:
                best_err, best_r = mse, r
        return best_r

    def summarize(self, lift: str) -> None:
        """Print a tabulated summary of the generalized synthetic control results.

        Parameters
        ----------
        lift : str
            The kind of lift to report. One of ``"absolute"``, ``"relative"``,
            ``"incremental"``, ``"cost-per"``, ``"revenue"`` or ``"roas"``.
        """
        if self.results is None:
            raise ValueError("results must not be None")
        lift = lift.casefold()
        if lift not in [
            "absolute",
            "relative",
            "incremental",
            "cost-per",
            "revenue",
            "roas",
        ]:
            raise ValueError(
                f"Cannot measure {lift}. Choose one of `absolute`, `relative`,  `incremental`, `cost-per`, `revenue` "
                f"or `roas`"
            )
        ci_alpha = self._get_ci_print()
        baseline = self.results["counterfactual"][self.y_variable].sum()
        if lift in ["incremental", "absolute"]:
            table_dict = {
                "Variant": [self.results["test"][self.y_variable].sum()],
                "Baseline": [baseline],
                "Metric": [self.y_variable],
                "Lift Type ": ["Incremental"],
                "Lift": [f"""{ceil(self.results["incrementality"]):,}"""],
                f"{ci_alpha} Lower CI": [f"""{ceil(self.results["incrementality_ci_lower"]):,}"""],
                f"{ci_alpha} Upper CI": [f"""{ceil(self.results["incrementality_ci_upper"]):,}"""],
            }
        elif lift == "relative":
            table_dict = {
                "Variant": [self.results["test"][self.y_variable].sum()],
                "Baseline": [baseline],
                "Metric": [self.y_variable],
                "Lift Type": ["Relative"],
                "Lift": [f"""{round(float(self.results["incrementality"]) * 100 / baseline, 2)}%"""],
                f"{ci_alpha} Lower CI": [f"""{round(self.results["incrementality_ci_lower"] * 100 / baseline, 2)}%"""],
                f"{ci_alpha} Upper CI": [f"""{round(self.results["incrementality_ci_upper"] * 100 / baseline, 2)}%"""],
            }
        elif lift == "revenue":
            table_dict = {
                "Variant": [f"""${round(self.results["test"][self.y_variable].sum() * self.msrp, 2):,}"""],
                "Baseline": [f"""${round(baseline * self.msrp, 2):,}"""],
                "Metric": ["Revenue"],
                "Lift Type ": ["Incremental"],
                "Lift": [f"""${round(self.results["incrementality"] * self.msrp, 2):,}"""],
                f"{ci_alpha} Lower CI": [f"""${round(self.results["incrementality_ci_lower"] * self.msrp, 2):,}"""],
                f"{ci_alpha} Upper CI": [f"""${round(self.results["incrementality_ci_upper"] * self.msrp, 2):,}"""],
            }
        else:
            roas_lift, roas_ci_lower, roas_ci_upper = self._get_roas()
            table_dict = {
                "Variant": [f"""${round(self.spend / self.results["test"][self.y_variable].sum(), 2)}"""],
                "Baseline": [f"""${round(self.spend / baseline, 2)}"""],
                "Metric": ["ROAS"],
                "Lift Type": ["Incremental"],
                "Lift": [f"${round(roas_lift, 2)}"],
                f"{ci_alpha} Lower CI": [f"${round(roas_ci_lower, 2)}"],
                f"{ci_alpha} Upper CI": [f"${round(roas_ci_upper, 2)}"],
            }
        table_dict["p_value"] = [self.results["p_value"]]
        print(tabulate(table_dict, headers="keys", tablefmt="grid"))

    def _get_roas(self) -> tuple[float, float, float]:
        if self.results is None:
            raise ValueError("results must not be None")
        lift = ceil(self.results["incrementality"])
        roas_lift = self.spend / lift if lift > 0 else np.inf
        ci_upper = ceil(self.results["incrementality_ci_upper"])
        roas_ci_lower = self.spend / ci_upper if ci_upper > 0 else np.inf
        ci_lower = ceil(self.results["incrementality_ci_lower"])
        roas_ci_upper = self.spend / ci_lower if ci_lower > 0 else np.inf
        return roas_lift, roas_ci_lower, roas_ci_upper

    def plot(self) -> None:
        """Plot our actual results, our counterfactual, the pointwise difference and cumulative difference.

        Returns
        -------
        Our three plots determining the results
        """
        if self.actual_pre is None:
            raise ValueError("actual_pre must not be None")
        if self.actual_post is None:
            raise ValueError("actual_post must not be None")
        if self.prediction_pre is None:
            raise ValueError("prediction_pre must not be None")
        if self.prediction_post is None:
            raise ValueError("prediction_post must not be None")
        if self.dates is None:
            raise ValueError("dates must not be None")
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
                        [
                            self.actual_pre[self.y_variable].to_numpy(),
                            self.actual_post[self.y_variable].to_numpy(),
                        ]
                    ),
                    marker={"color": "blue"},
                    mode="lines",
                    name="Actual",
                ),
                go.Scatter(
                    x=self.dates,
                    y=np.concatenate(
                        [
                            self.prediction_pre[self.y_variable].to_numpy(),
                            self.prediction_post[self.y_variable].to_numpy(),
                        ]
                    ),
                    marker={"color": "red"},
                    mode="lines",
                    name="Counterfactual",
                ),
            ]
        )
        residuals = np.concatenate(
            [self.actual_pre[self.y_variable].to_numpy(), self.actual_post[self.y_variable].to_numpy()]
        ) - np.concatenate(
            [
                self.prediction_pre[self.y_variable].to_numpy(),
                self.prediction_post[self.y_variable].to_numpy(),
            ]
        )
        middle_fig = go.Figure(
            [
                go.Scatter(
                    x=self.dates,
                    y=residuals,
                    marker={"color": "purple"},
                    mode="lines",
                    name="Residuals",
                )
            ]
        )
        cum_resids = self.actual_post[self.y_variable].to_numpy() - self.prediction_post[self.y_variable].to_numpy()
        post_period_date = date_cls.fromisoformat(self.post_period)
        marketing_start = [d for d in self.dates if d >= post_period_date]
        bottom_fig = go.Figure(
            [
                go.Scatter(
                    x=marketing_start,
                    y=cum_resids.cumsum(),
                    marker={"color": "orange"},
                    mode="lines",
                    name="Cumulative Incrementality",
                )
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
