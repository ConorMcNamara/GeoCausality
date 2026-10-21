import abc
import inspect
import warnings
from abc import ABC
from collections.abc import Callable
from datetime import date as date_cls
from datetime import datetime as datetime_cls
from itertools import combinations
from math import comb
from typing import Any, Self

import narwhals as nw
import numpy as np
import plotly.graph_objects as go
import polars as pl
from narwhals.typing import IntoDataFrame
from plotly.subplots import make_subplots
from scipy.optimize import Bounds, LinearConstraint, minimize, nnls
from tabulate import tabulate  # type: ignore

from GeoCausality._plotting import apply_theme, get_theme

# The lift types every ``summarize`` accepts. ``FixedEffects`` allows a subset.
_LIFT_TYPES: tuple[str, ...] = ("absolute", "relative", "incremental", "cost-per", "revenue", "roas")


def date_key(frame: nw.DataFrame, date_variable: str) -> nw.Expr:
    """Build a backend-agnostic, comparable string key for the date column.

    Date and datetime columns are formatted as ``YYYY-MM-DD`` so they compare
    against ``pre_period`` / ``post_period`` by calendar date. A plain string
    cast would render a datetime as ``"2021-01-01 00:00:00"``, which sorts
    after ``"2021-01-01"`` and silently drops the boundary date from ``<=``
    filters. Numeric columns (integer years, week indices) are left numeric so
    they compare numerically (a string cast would put ``"10"`` before ``"5"``).
    Other columns (ISO strings) are cast to string.

    Parameters
    ----------
    frame : narwhals DataFrame
        The frame holding the date column (used for its dtype).
    date_variable : str
        The name of the date column.

    Returns
    -------
    The narwhals expression for the comparison key.
    """
    column = nw.col(date_variable)
    if frame.schema[date_variable] in (nw.Datetime, nw.Date):
        return column.dt.to_string("%Y-%m-%d")
    if frame.schema[date_variable].is_numeric():
        return column
    return column.cast(nw.String)


def period_key(value: Any, frame: nw.DataFrame | None = None, date_variable: str | None = None) -> Any:
    """Normalise a date value to the form produced by ``date_key``.

    Parameters
    ----------
    value : str, date, datetime, number or other
        A period boundary or a value taken from the date column.
    frame : narwhals DataFrame, optional
        The frame holding the date column. When the column is numeric, the
        value is coerced to a number so the comparison is numeric.
    date_variable : str, optional
        The name of the date column (required with ``frame``).

    Returns
    -------
    ``YYYY-MM-DD`` for date and datetime values, a number for a numeric date
    column, otherwise ``str(value)``.
    """
    if frame is not None and date_variable is not None and frame.schema[date_variable].is_numeric():
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Column '{date_variable}' is numeric, so period boundary {value!r} must be a number."
            ) from exc
        return int(number) if number.is_integer() else number
    if isinstance(value, datetime_cls):
        return value.date().isoformat()
    if isinstance(value, date_cls):
        return value.isoformat()
    return str(value)


class Estimator(abc.ABC):
    # Constructor arguments, recorded by ``__new__`` so placebo refits
    # (randomization inference, placebo-in-time) can rebuild the estimator with
    # the same hyperparameters (penalties, ranks, bandwidths, ...) rather than
    # the defaults.
    _init_kwargs: dict[str, Any]

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        obj = super().__new__(cls)
        init_kwargs: dict[str, Any] = {}
        signature = inspect.signature(cls.__init__)
        try:
            bound = signature.bind(obj, *args, **kwargs)
        except TypeError:
            bound = None
        if bound is not None:
            for name, value in list(bound.arguments.items())[1:]:
                if signature.parameters[name].kind is inspect.Parameter.VAR_KEYWORD:
                    init_kwargs.update(value)
                else:
                    init_kwargs[name] = value
        obj._init_kwargs = init_kwargs
        return obj

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
        """Initialize the geo-causality estimator (abstract base class).

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
        treatment_variable : str, optional, default="is_treatment"
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
        """
        self.data: nw.DataFrame = nw.from_native(data, eager_only=True)
        self.test_geos = test_geos
        self.control_geos = control_geos
        self.pre_period = pre_period
        self.post_period = post_period
        self.geo_variable = geo_variable
        self.date_variable = date_variable
        self.y_variable = y_variable
        if (self.control_geos is None) | (self.test_geos is None):
            self.treatment_variable = treatment_variable
        else:
            self.treatment_variable = "is_test"
        self.alpha = alpha
        self.msrp = msrp
        self.spend = spend
        self.results: dict[str, Any] | None = None

    def _clone(self, **overrides: Any) -> Self:
        """Build a fresh estimator of this class with the same constructor arguments.

        Parameters
        ----------
        **overrides : Any
            Constructor arguments to replace (e.g. ``data``, ``test_geos``).

        Returns
        -------
        A new, un-fitted estimator.
        """
        return type(self)(**{**self._init_kwargs, **overrides})

    def _restrict_to_listed_geos(self) -> None:
        """Drop geos outside ``test_geos`` and ``control_geos`` when both are given.

        Otherwise every non-test geo would silently enter the donor pool,
        including geos the caller deliberately left out (e.g. holdouts or
        markets exposed to spillover).
        """
        if self.test_geos is not None and self.control_geos is not None:
            listed = list(self.test_geos) + list(self.control_geos)
            self.data = self.data.filter(nw.col(self.geo_variable).is_in(listed))

    @staticmethod
    def _placebo_sets(
        donor_geos: list[Any],
        n_treated: int,
        max_placebos: int | None,
        seed: int | None,
    ) -> list[tuple[Any, ...]]:
        """Draw the placebo treatment assignments for randomization inference.

        Each placebo is a set of ``n_treated`` donor geos, so a placebo's
        statistic is on the same scale as the observed one (which aggregates
        ``n_treated`` geos). At least one donor is always left as a control.

        Parameters
        ----------
        donor_geos : list of str
            The control geos available for placebo assignment.
        n_treated : int
            The number of geos in the real treated group.
        max_placebos : int, optional
            Cap on the number of placebos. Defaults to every donor when
            ``n_treated == 1`` and to 200 subsets otherwise.
        seed : int, optional
            Seed for sampling placebos when they are not enumerated.

        Returns
        -------
        A list of placebo geo tuples.

        Raises
        ------
        ValueError
            If there are not enough donors to form a placebo and its controls.
        """
        k = max(int(n_treated), 1)
        if len(donor_geos) <= k:
            raise ValueError(
                f"Need more than {k} control geos to form placebos of {k} geos with a remaining donor pool"
            )
        limit = max_placebos if max_placebos is not None else (len(donor_geos) if k == 1 else 200)
        n_candidates = comb(len(donor_geos), k)
        if n_candidates <= limit:
            return list(combinations(donor_geos, k))
        # Sample donor indices (not labels) so geo labels keep their own type.
        rng = np.random.default_rng(seed)
        if k == 1:
            return [(donor_geos[i],) for i in rng.choice(len(donor_geos), size=limit, replace=False)]
        seen: set[tuple[int, ...]] = set()
        while len(seen) < limit:
            seen.add(tuple(sorted(int(i) for i in rng.choice(len(donor_geos), size=k, replace=False))))
        return [tuple(donor_geos[i] for i in idx) for idx in sorted(seen)]

    @abc.abstractmethod
    def pre_process(self) -> "Estimator":
        """Pre-process our data to make it usable for the estimator.

        Returns
        -------
        Itself, so it can be chained with generate()
        """
        pass

    @abc.abstractmethod
    def generate(self) -> "Estimator":
        """Run our models on pre-processed data to estimate causality.

        Returns
        -------
        Itself, so it can be chained with summarize()
        """
        pass

    @abc.abstractmethod
    def summarize(self, lift: str) -> None:
        """Summarize the results of generated models on our pre-processed data.

        Parameters
        ----------
        lift : str
            The kind of uplift we are measuring for geo-causality

        Returns
        -------
        The lift of our campaign
        """
        pass

    def _get_ci_print(self) -> str:
        return f"{(1 - self.alpha) * 100:g}%"

    def _validate_lift(self, lift: str, allowed: tuple[str, ...] = _LIFT_TYPES) -> str:
        """Normalise and validate the requested lift type.

        Parameters
        ----------
        lift : str
            The requested lift type (case-insensitive).
        allowed : tuple of str, optional
            The permitted lift types; defaults to every type in ``_LIFT_TYPES``.

        Returns
        -------
        The normalised (case-folded) lift type.

        Raises
        ------
        ValueError
            If ``lift`` is not one of ``allowed``.
        """
        lift = lift.casefold()
        if lift not in allowed:
            choices = ", ".join(f"`{option}`" for option in allowed)
            raise ValueError(f"Cannot measure {lift}. Choose one of {choices}")
        return lift

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        """Divide, returning a signed ``inf`` (or ``nan``) for a zero denominator.

        Matches the guard in ``_get_roas`` / ``_get_cost_per`` so a degenerate
        summary (e.g. a counterfactual or treated post-period that sums to zero)
        reports ``inf`` rather than raising ``ZeroDivisionError``. The infinity
        carries the sign of the numerator, and ``0 / 0`` is ``nan``.
        """
        if denominator:
            return numerator / denominator
        if numerator == 0 or np.isnan(numerator):
            return float(np.nan)
        return float(np.inf) if numerator > 0 else float(-np.inf)

    def _cost_per_value(self, incrementality: float) -> float:
        """Spend per incremental unit, tolerating unbounded CIs.

        Parameters
        ----------
        incrementality : float
            An incrementality estimate or CI bound; may be infinite or NaN.

        Returns
        -------
        ``spend / incrementality``; ``inf`` when it is not positive, ``0`` when
        it is ``+inf`` and ``nan`` when it is NaN. The incrementality is not
        rounded first: rounding up (e.g. 0.3 -> 1) biased the cost downward.
        """
        incrementality = float(incrementality)
        if np.isnan(incrementality):
            return float("nan")
        if incrementality == float("inf"):
            return 0.0
        return self.spend / incrementality if incrementality > 0 else float("inf")

    def _format_lift_cells(
        self,
        lift: str,
        point: float,
        lower: float,
        upper: float,
        relative_divisor: float | None = None,
    ) -> tuple[list[str], list[str], list[str]]:
        """Format the Lift / lower-CI / upper-CI cells for one lift type.

        Parameters
        ----------
        lift : str
            The lift type. ``"relative"`` renders a percentage of
            ``relative_divisor``; ``"revenue"`` scales by ``msrp`` as currency;
            ``"roas"`` / ``"cost-per"`` render currency as-is; anything else
            renders a thousands-separated integer count.
        point, lower, upper : float
            The point estimate and its confidence-interval bounds.
        relative_divisor : float, optional
            The denominator for the ``"relative"`` percentage.

        Returns
        -------
        A tuple of single-element lists ``(Lift, lower CI, upper CI)`` ready to
        drop into the tabulate column dict.
        """
        if lift == "relative":
            if relative_divisor is None:
                raise ValueError("relative_divisor is required for relative lift")

            def fmt(value: float) -> str:
                return f"{round(self._safe_ratio(float(value) * 100, relative_divisor), 2)}%"
        elif lift == "revenue":

            def fmt(value: float) -> str:
                return f"${round(value * self.msrp, 2):,}"
        elif lift in ("roas", "cost-per"):

            def fmt(value: float) -> str:
                return f"${round(value, 2)}"
        else:

            def fmt(value: float) -> str:
                return f"{round(value):,}" if np.isfinite(value) else str(float(value))

        return [fmt(point)], [fmt(lower)], [fmt(upper)]

    @abc.abstractmethod
    def _get_roas(self) -> tuple[float, float, float]:
        """Return our Return on Ad Spend (ROAS) and CIs.

        Returns
        -------
        A tuple containing our ROAS as well as Confidence Intervals
        """
        pass


class EconometricEstimator(Estimator, ABC):
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
        """Initialize the econometric estimator (abstract base class for FixedEffects and Diff-in-Diff).

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
        self.model: Any = None
        # Inference path: "auto" picks conformal for long pre-periods and falls
        # back to jackknife+ for short ones; "conformal" / "jackknife" force one;
        # "bootstrap" uses the parametric bootstrap (needs ``_bootstrap_refit``).
        self.inference_method: str = "auto"
        # Parametric-bootstrap settings (used when inference_method == "bootstrap").
        self.n_boot: int = 1000
        self.bootstrap_seed: int = 0
        # Donor matrices cached by synthetic-control estimators for faithful
        # jackknife+ (the shared leave-one-out loop in ``_block_loo``).
        self._jk_x_pre: np.ndarray | None = None
        self._jk_y_pre: np.ndarray | None = None
        self._jk_x_post: np.ndarray | None = None
        # Number of treated geos summed into the treated series. Estimators whose
        # donor weights are constrained to sum to one fit the per-geo average
        # (treated series / n) and scale the counterfactual back up by n, since
        # sum-to-one weights cannot reproduce a sum of several geos.
        self._n_treated: int = 1
        # Zero-centred null replicates of the incrementality from the last
        # parametric bootstrap (used by GeoLift to test the ASC estimate).
        self._bootstrap_null: np.ndarray | None = None

    def pre_process(self) -> "EconometricEstimator":
        self._restrict_to_listed_geos()
        if self.test_geos is not None:
            if self.treatment_variable is not None:
                self.data: nw.DataFrame = self.data.with_columns(
                    nw.col(self.geo_variable).is_in(self.test_geos).cast(nw.Int64).alias(self.treatment_variable)
                )
        elif self.control_geos is not None:
            if self.treatment_variable is not None:
                self.data: nw.DataFrame = self.data.with_columns(
                    (1 - nw.col(self.geo_variable).is_in(self.control_geos).cast(nw.Int64)).alias(
                        self.treatment_variable
                    )
                )
        else:
            pass
        # Compare dates through an ISO ``YYYY-MM-DD`` key so the comparison is
        # backend agnostic: polars `Date` columns cannot be compared to a string
        # literal directly, while pandas `datetime` columns cannot be compared
        # to a `date` object. Dates strictly between ``pre_period`` and
        # ``post_period`` (e.g. a washout or ramp-up window) belong to neither
        # period and are dropped, so they are neither fitted on nor counted as
        # post-period.
        key = date_key(self.data, self.date_variable)
        in_pre = key <= period_key(self.pre_period, self.data, self.date_variable)
        in_post = key >= period_key(self.post_period, self.data, self.date_variable)
        self.data: nw.DataFrame = self.data.filter(in_pre | in_post).with_columns(
            nw.when(in_pre).then(0).otherwise(1).alias("treatment_period")
        )
        if self.treatment_variable is not None:
            n_treated = self.data.filter(nw.col(self.treatment_variable) == 1)[self.geo_variable].n_unique()
            self._n_treated = max(int(n_treated), 1)

        return self

    # ------------------------------------------------------------------
    # Shared donor-matrix construction
    #
    # Every synthetic-control style estimator builds the treated outcome as a
    # per-date sum and the donor pool as a date-by-geo matrix, filtered to the
    # pre-period, post-period or the full window. These two helpers build both,
    # so each estimator's ``pre_process`` / ``generate`` just names the pieces it
    # needs. ``pre_aggregate`` toggles the ``group_by([date, geo]).sum()`` some
    # estimators run before pivoting (others assume one row per date-geo);
    # ``sort_pivot`` orders the pivot by date for estimators that split
    # pre/post by row index rather than joining on date.
    # ------------------------------------------------------------------
    def _period_filter(self, treatment_value: int, period: str) -> "nw.Expr":
        """Build the ``(treatment, period)`` row filter for the aggregation helpers.

        Parameters
        ----------
        treatment_value : int
            ``1`` for treated rows, ``0`` for control rows.
        period : {"pre", "post", "all"}
            Restrict to the pre-period, the post-period, or keep every date.

        Returns
        -------
        The narwhals boolean expression selecting the requested rows.
        """
        if self.treatment_variable is None:
            raise ValueError("treatment_variable must not be None")
        row_filter = nw.col(self.treatment_variable) == treatment_value
        if period == "pre":
            row_filter = row_filter & (nw.col("treatment_period") == 0)
        elif period == "post":
            row_filter = row_filter & (nw.col("treatment_period") == 1)
        return row_filter

    def _treated_series(self, period: str) -> nw.DataFrame:
        """Treated outcome summed by date over the given period.

        Parameters
        ----------
        period : {"pre", "post", "all"}
            Which window to aggregate.

        Returns
        -------
        A narwhals frame of ``(date, y)`` sorted by date.
        """
        return (
            self.data.filter(self._period_filter(1, period))
            .select([self.y_variable, self.date_variable])
            .group_by(self.date_variable)
            .agg(nw.col(self.y_variable).sum())
            .sort(self.date_variable)
        )

    def _control_matrix(self, period: str, *, pre_aggregate: bool = True, sort_pivot: bool = False) -> nw.DataFrame:
        """Control donor matrix (rows = date, cols = geo) over the given period.

        Parameters
        ----------
        period : {"pre", "post", "all"}
            Which window to aggregate.
        pre_aggregate : bool, default=True
            Sum the outcome per ``(date, geo)`` before pivoting. Set ``False`` when
            the panel already has one row per date-geo (the pivot then assumes
            uniqueness).
        sort_pivot : bool, default=False
            Sort the pivoted matrix by date. Needed by estimators that split
            pre/post by row index rather than joining on the date column.

        Returns
        -------
        The pivoted donor matrix as a narwhals frame. Donor columns are always in
        geo-sorted order (independent of the input row order), so a matrix used to
        fit weights and one used to predict share the same column order.
        """
        frame = self.data.filter(self._period_filter(0, period)).select(
            [self.y_variable, self.date_variable, self.geo_variable]
        )
        if pre_aggregate:
            frame = frame.group_by([self.date_variable, self.geo_variable]).agg(nw.col(self.y_variable).sum())
        # Sort by (date, geo) before pivoting so the donor columns come out in a
        # stable geo-sorted order regardless of the source row order. Without this,
        # estimators that fit on one _control_matrix pivot and predict on another
        # would map the weight vector onto mis-ordered donor columns.
        frame = frame.sort([self.date_variable, self.geo_variable])
        pivot = nw.from_native(
            frame.to_native().pivot(on=self.geo_variable, index=self.date_variable, values=self.y_variable),
            eager_only=True,
        )
        if sort_pivot:
            pivot = pivot.sort(self.date_variable)
        return pivot

    # ------------------------------------------------------------------
    # Shared results surface
    #
    # Every synthetic-control style estimator reports the same lift table and
    # return-on-ad-spend from an identical ``results`` contract, so ``summarize``
    # and ``_get_roas`` / ``_get_cost_per`` are defined once here. They read ``results["test"]`` and
    # ``results["counterfactual"]`` as arrays of the post-period outcome, so each
    # estimator's ``generate`` stores those two keys as numpy arrays. Estimators
    # with a different estimand (e.g. ``DiffinDiff``, ``FixedEffects``) override
    # ``summarize``.
    # ------------------------------------------------------------------
    def summarize(self, lift: str) -> None:
        """Print a tabulated summary of the counterfactual results.

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
        variant = float(np.sum(self.results["test"]))
        baseline = float(np.sum(self.results["counterfactual"]))
        incrementality = (
            self.results["incrementality"],
            self.results["incrementality_ci_lower"],
            self.results["incrementality_ci_upper"],
        )
        table_dict: dict[str, list[Any]]
        if lift in ("incremental", "absolute"):
            table_dict = {
                "Variant": [variant],
                "Baseline": [baseline],
                "Metric": [self.y_variable],
                "Lift Type": ["Incremental"],
            }
            cells = self._format_lift_cells(lift, *incrementality)
        elif lift == "relative":
            table_dict = {
                "Variant": [variant],
                "Baseline": [baseline],
                "Metric": [self.y_variable],
                "Lift Type": ["Relative"],
            }
            cells = self._format_lift_cells(lift, *incrementality, relative_divisor=baseline)
        elif lift == "revenue":
            table_dict = {
                "Variant": [f"${round(variant * self.msrp, 2):,}"],
                "Baseline": [f"${round(baseline * self.msrp, 2):,}"],
                "Metric": ["Revenue"],
                "Lift Type": ["Incremental"],
            }
            cells = self._format_lift_cells(lift, *incrementality)
        elif lift == "roas":
            table_dict = {
                "Variant": [f"${round(variant * self.msrp, 2):,}"],
                "Baseline": [f"${round(baseline * self.msrp, 2):,}"],
                "Metric": ["ROAS"],
                "Lift Type": ["Incremental"],
            }
            cells = self._format_lift_cells(lift, *self._get_roas())
        else:
            table_dict = {
                "Variant": [f"${round(self._safe_ratio(self.spend, variant), 2)}"],
                "Baseline": [f"${round(self._safe_ratio(self.spend, baseline), 2)}"],
                "Metric": ["Cost-per"],
                "Lift Type": ["Incremental"],
            }
            cells = self._format_lift_cells(lift, *self._get_cost_per())
        table_dict["Lift"], table_dict[lo_key], table_dict[hi_key] = cells
        table_dict["p_value"] = [self.results["p_value"]]
        print(tabulate(table_dict, headers="keys", tablefmt="grid"))

    def _get_roas(self) -> tuple[float, float, float]:
        """Return the return-on-ad-spend point estimate and confidence interval.

        ROAS is ``(incrementality * msrp) / spend``: the incremental revenue
        generated per dollar of ad spend.

        Returns
        -------
        A tuple of (ROAS, lower CI, upper CI). With zero spend, returns a
        signed ``inf`` (``nan`` when the revenue is zero too).
        """
        if self.results is None:
            raise ValueError("results must not be None")
        incr = self.results["incrementality"]

        def revenue(incrementality: float) -> float:
            # ``inf * 0`` is nan; zero revenue per unit makes revenue zero.
            return incrementality * self.msrp if self.msrp else 0.0

        roas_lift = self._safe_ratio(revenue(incr), self.spend)
        roas_ci_lower = self._safe_ratio(revenue(self.results["incrementality_ci_lower"]), self.spend)
        roas_ci_upper = self._safe_ratio(revenue(self.results["incrementality_ci_upper"]), self.spend)
        return roas_lift, roas_ci_lower, roas_ci_upper

    def _get_cost_per(self) -> tuple[float, float, float]:
        """Return the cost-per-acquisition point estimate and confidence interval.

        CPA is ``spend / incrementality``: the ad spend required per
        incremental unit.

        Returns
        -------
        A tuple of (CPA, lower CI, upper CI). Returns ``inf`` when the
        incrementality (or its CI bound) is not positive.
        """
        if self.results is None:
            raise ValueError("results must not be None")
        cost_per = self._cost_per_value(self.results["incrementality"])
        cost_per_lower = self._cost_per_value(self.results["incrementality_ci_upper"])
        cost_per_upper = self._cost_per_value(self.results["incrementality_ci_lower"])
        return cost_per, cost_per_lower, cost_per_upper

    def _finalize_counterfactual_results(
        self,
        actual_pre: Any,
        prediction_pre: Any,
        actual_post: Any,
        prediction_post: Any,
        q: float = 1.0,
        **extra: Any,
    ) -> dict[str, Any]:
        """Assemble the shared counterfactual ``results`` dict with conformal inference.

        Every synthetic-control style ``generate`` ends the same way: normalise the
        four actual/counterfactual series to arrays, record the ``test`` /
        ``counterfactual`` / ``lift`` / ``incrementality`` fields, then merge the
        conformal p-value, confidence intervals and prediction band. This builds
        that dict once. Estimators with extra fields (e.g. ``n_factors``, ``att``)
        pass them as keyword arguments in ``extra``.

        Parameters
        ----------
        actual_pre, prediction_pre : numpy array or narwhals frame
            Observed and counterfactual values over the pre-period.
        actual_post, prediction_post : numpy array or narwhals frame
            Observed and counterfactual values over the post-period.
        q : float, default=1.0
            The exponent of the moving-block test statistic.
        **extra : Any
            Estimator-specific fields to add to the results dict.

        Returns
        -------
        The populated ``results`` dictionary.
        """
        actual_pre = self._counterfactual_series(actual_pre)
        prediction_pre = self._counterfactual_series(prediction_pre)
        actual_post = self._counterfactual_series(actual_post)
        prediction_post = self._counterfactual_series(prediction_post)
        results: dict[str, Any] = {
            "test": actual_post,
            "counterfactual": prediction_post,
            "lift": actual_post - prediction_post,
            "_actual_pre": actual_pre,
            "_prediction_pre": prediction_pre,
            **extra,
        }
        results["incrementality"] = float(np.sum(results["lift"]))
        results.update(self._conformal_inference(actual_pre, prediction_pre, actual_post, prediction_post, q=q))
        return results

    # ------------------------------------------------------------------
    # Conformal inference
    #
    # Synthetic-control style estimators produce a fitted counterfactual but
    # no closed-form standard errors, so we attach distribution-free p-values
    # and confidence intervals via conformal inference. Two complementary
    # tools are provided, both operating purely on the pre-/post-period
    # residuals (actual - counterfactual):
    #   * The Chernozhukov, Wuthrich & Zhu (2021) moving-block permutation
    #     test, which gives an exact p-value for the sharp null of no effect
    #     and confidence intervals for a constant per-period effect via test
    #     inversion. Exactness needs the pre- and post-period residuals to be
    #     exchangeable, so under each null ``theta`` the counterfactual is refit
    #     on all T periods with the post-period outcome set to ``y - theta``
    #     (``_refit_counterfactual``). Fitting on the pre-period only and
    #     shifting the post-period residuals makes the in-sample pre residuals
    #     systematically smaller than the out-of-sample post residuals, and the
    #     test over-rejects badly as donors are added.
    #   * Conformal prediction bands, calibrated on out-of-sample (cross-fitted)
    #     pre-period residuals, which give a pointwise band around the
    #     counterfactual.
    # Estimators that cannot refit fall back to the fitted residuals, which is
    # only approximate.
    # ------------------------------------------------------------------
    @staticmethod
    def _conformal_residuals(
        actual_pre: Any,
        prediction_pre: Any,
        actual_post: Any,
        prediction_post: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute pre- and post-period residuals as flat numpy arrays.

        Parameters
        ----------
        actual_pre, prediction_pre : array-like
            Observed and counterfactual values over the pre-period.
        actual_post, prediction_post : array-like
            Observed and counterfactual values over the post-period.

        Returns
        -------
        A tuple of (pre_period_residuals, post_period_residuals).
        """
        pre = np.asarray(actual_pre, dtype=float).ravel() - np.asarray(prediction_pre, dtype=float).ravel()
        post = np.asarray(actual_post, dtype=float).ravel() - np.asarray(prediction_post, dtype=float).ravel()
        return pre, post

    @staticmethod
    def _block_statistics(residuals: np.ndarray, t1: int, q: float = 1.0) -> np.ndarray:
        """Compute the moving-block test statistic over every cyclic block.

        For each of the ``len(residuals)`` cyclically-contiguous blocks of
        length ``t1`` we compute ``S = mean(|u|^q) ** (1 / q)``.

        Parameters
        ----------
        residuals : numpy array
            The full (pre-period then post-period) residual series.
        t1 : int
            The length of the post-period block.
        q : float, default=1.0
            The exponent of the test statistic.

        Returns
        -------
        An array of block statistics, indexed by the block's start position.
        """
        n = residuals.shape[0]
        offsets = np.arange(t1)
        stats = np.empty(n)
        for start in range(n):
            block = residuals[(start + offsets) % n]
            stats[start] = np.mean(np.abs(block) ** q) ** (1.0 / q)
        return stats

    def _conformal_p_value(self, pre_resid: np.ndarray, post_resid: np.ndarray, q: float = 1.0) -> float:
        """Permutation p-value for the sharp null of no treatment effect.

        Parameters
        ----------
        pre_resid, post_resid : numpy array
            Pre- and post-period residuals.
        q : float, default=1.0
            The exponent of the moving-block test statistic.

        Returns
        -------
        The fraction of cyclic blocks whose statistic is at least as large as
        the observed post-period block.
        """
        full = np.concatenate([pre_resid, post_resid])
        t1 = post_resid.shape[0]
        stats = self._block_statistics(full, t1, q)
        observed = stats[full.shape[0] - t1]
        return float(np.mean(stats >= observed))

    @staticmethod
    def _bisect_threshold(
        p_func: Callable[[float], float],
        x_in: float,
        x_out: float,
        alpha: float,
        iters: int = 64,
    ) -> float:
        """Find the effect boundary where the test inverts at level ``alpha``.

        Assumes ``p_func(x_in) >= alpha`` (inside the acceptance region) and
        searches towards ``x_out`` for the furthest point still accepted.

        Parameters
        ----------
        p_func : callable
            Maps a candidate effect to its permutation p-value.
        x_in : float
            A point known to be inside the acceptance region.
        x_out : float
            A point assumed to be outside the acceptance region.
        alpha : float
            The significance level.
        iters : int, default=64
            Number of bisection iterations.

        Returns
        -------
        The boundary effect value.
        """
        if p_func(x_out) >= alpha:
            return x_out
        tol = 1e-6 * abs(x_out - x_in)
        for _ in range(iters):
            if abs(x_out - x_in) <= tol:
                break
            mid = 0.5 * (x_in + x_out)
            if p_func(mid) >= alpha:
                x_in = mid
            else:
                x_out = mid
        return x_in

    def _conformal_interval(
        self,
        pre_resid: np.ndarray,
        post_resid: np.ndarray,
        q: float = 1.0,
        alpha: float | None = None,
        resid_fn: Callable[[float], tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> tuple[float, float]:
        """Confidence interval for a constant per-period effect by test inversion.

        Inverts the moving-block permutation test: the interval is the set of
        constant per-period effects ``theta`` for which the sharp null is not
        rejected at level ``alpha``. Under each ``theta``, ``resid_fn`` returns
        the residuals of the counterfactual refit on all periods with the
        post-period outcome reduced by ``theta`` (Chernozhukov, Wuthrich & Zhu,
        2021). Without it, ``theta`` is subtracted from the fixed post-period
        residuals, which is only an approximation.

        Parameters
        ----------
        pre_resid, post_resid : numpy array
            Pre- and post-period residuals of the original fit; they set the
            search's starting point and scale.
        q : float, default=1.0
            The exponent of the moving-block test statistic.
        alpha : float, optional
            The significance level. Defaults to ``self.alpha``.
        resid_fn : callable, optional
            Maps ``theta`` to the null-imposed ``(pre, post)`` residuals.

        Returns
        -------
        A tuple of (lower, upper) bounds on the per-period effect.
        """
        if alpha is None:
            alpha = self.alpha
        center = float(np.mean(post_resid))
        spread = float(np.std(np.concatenate([pre_resid, post_resid])))
        if spread == 0.0:
            spread = abs(center) if center != 0.0 else 1.0
        if resid_fn is None:

            def resid_fn(theta: float) -> tuple[np.ndarray, np.ndarray]:
                return pre_resid, post_resid - theta

        def p_func(theta: float) -> float:
            return self._conformal_p_value(*resid_fn(theta), q)

        # Bisection needs a point inside the acceptance region. The mean
        # post-period residual usually is, but not always (e.g. when the
        # pre-period fit is much tighter than the post-period noise), so fall
        # back to a grid search for the accepted points nearest each end.
        lo_in = hi_in = center
        if p_func(center) < alpha:
            grid = center + spread * np.linspace(-10.0, 10.0, 81)
            accepted = [theta for theta in grid if p_func(theta) >= alpha]
            if not accepted:
                warnings.warn(
                    "The conformal test rejects every constant effect, so the confidence "
                    "interval is empty (reported as NaN); the counterfactual likely fits "
                    "the pre-period much better than the post-period.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return float("nan"), float("nan")
            lo_in, hi_in = float(min(accepted)), float(max(accepted))

        upper = self._expand_and_bisect(p_func, hi_in, spread, alpha)
        lower = self._expand_and_bisect(p_func, lo_in, -spread, alpha)
        return lower, upper

    def _expand_and_bisect(
        self,
        p_func: Callable[[float], float],
        x_in: float,
        step: float,
        alpha: float,
        max_doublings: int = 40,
    ) -> float:
        """Locate the acceptance-region boundary in the direction of ``step``.

        Doubles the distance from ``x_in`` until a rejected point is found, then
        bisects. Returns ``+/- inf`` when no rejected point is found, i.e. the
        region is unbounded in that direction (possible when the residual series
        is too short for the test to reject at level ``alpha``).

        Parameters
        ----------
        p_func : callable
            Maps a candidate effect to its permutation p-value.
        x_in : float
            A point inside the acceptance region.
        step : float
            Initial signed step; its sign sets the search direction.
        alpha : float
            The significance level.
        max_doublings : int, default=40
            Cap on the outward expansion.

        Returns
        -------
        The boundary effect value.
        """
        distance = 10.0 * step
        for _ in range(max_doublings):
            x_out = x_in + distance
            if p_func(x_out) < alpha:
                return self._bisect_threshold(p_func, x_in, x_out, alpha)
            x_in = x_out
            distance *= 2.0
        return float("inf") if step > 0 else float("-inf")

    def _split_conformal_band(
        self,
        pre_resid: np.ndarray,
        alpha: float | None = None,
    ) -> float:
        """Half-width of a conformal prediction band around the counterfactual.

        Returns the conformalized ``(1 - alpha)`` quantile of the absolute
        calibration residuals, so a pointwise ``(1 - alpha)`` prediction interval
        is ``counterfactual +/- band``. The residuals must be out-of-sample
        (``_cross_fit_residuals``): in-sample residuals of the fitted model are
        too small, and the band under-covers.

        Parameters
        ----------
        pre_resid : numpy array
            Out-of-sample pre-period residuals used for calibration.
        alpha : float, optional
            The significance level. Defaults to ``self.alpha``.

        Returns
        -------
        The band half-width.
        """
        if alpha is None:
            alpha = self.alpha
        # The conformalized band is the ceil((n + 1)(1 - alpha))-th smallest
        # absolute residual (widening to the max when that rank exceeds n) --
        # identical to the jackknife+ quantile, so reuse it. (A previous
        # np.quantile(scores, k / n) computed one order statistic too high.)
        return self._jackknife_quantile(pre_resid, alpha)

    def _conformal_inference(
        self,
        actual_pre: Any,
        prediction_pre: Any,
        actual_post: Any,
        prediction_post: Any,
        q: float = 1.0,
    ) -> dict[str, Any]:
        """Bundles conformal p-value, effect CIs and prediction band.

        Parameters
        ----------
        actual_pre, prediction_pre : array-like
            Observed and counterfactual values over the pre-period.
        actual_post, prediction_post : array-like
            Observed and counterfactual values over the post-period.
        q : float, default=1.0
            The exponent of the moving-block test statistic.

        Returns
        -------
        A dict suitable for ``self.results.update(...)`` containing the
        p-value, per-period ``lift`` CIs, total ``incrementality`` CIs, the
        prediction band half-width, and the ``method`` used for the intervals
        (``"conformal"``, ``"jackknife+"`` or ``"jackknife+ (residual)"``).

        Notes
        -----
        The moving-block permutation test and split-conformal band both need a
        reasonable number of pre-period points: the band quantile saturates and
        the permutation p-value loses resolution when the pre-period is short.
        When that happens (see ``_pre_period_too_short``) the interval falls back
        to jackknife+ (Barber et al., 2021), which reuses every pre-period point
        for both fitting and calibration. An estimator that overrides
        ``_loo_counterfactuals`` gets the faithful refit-based jackknife+;
        otherwise a residual-only approximation is used.
        """
        pre_resid, post_resid = self._conformal_residuals(actual_pre, prediction_pre, actual_post, prediction_post)
        t1 = post_resid.shape[0]
        resid_fn = self._null_residual_fn(actual_pre, actual_post)
        if resid_fn is None:
            p_value = self._conformal_p_value(pre_resid, post_resid, q)
        else:
            p_value = self._conformal_p_value(*resid_fn(0.0), q)

        boot = None
        if self.inference_method == "bootstrap":
            fitted_pre = np.asarray(prediction_pre, dtype=float).ravel()
            fitted_post = np.asarray(prediction_post, dtype=float).ravel()
            actual_post_arr = np.asarray(actual_post, dtype=float).ravel()
            boot = self._parametric_bootstrap_interval(fitted_pre, fitted_post, actual_post_arr, pre_resid)
        if boot is not None:
            method = "bootstrap"
            lift_lower, lift_upper, band, p_value = boot
        elif self._pre_period_too_short(pre_resid.shape[0]):
            loo = self._loo_counterfactuals()
            if loo is not None:
                method = "jackknife+"
                lift_lower, lift_upper, band = self._jackknife_plus_interval(
                    loo[0], loo[1], np.asarray(actual_post, dtype=float).ravel(), t1
                )
            else:
                method = "jackknife+ (residual)"
                lift_lower, lift_upper, band = self._jackknife_residual_interval(pre_resid, post_resid)
            if not (np.isfinite(lift_lower) and np.isfinite(lift_upper)):
                warnings.warn(
                    f"The pre-period is too short for a finite {self._get_ci_print()} interval; "
                    "the reported interval is unbounded. Use a longer pre-period, a larger alpha, "
                    'or inference_method="bootstrap".',
                    RuntimeWarning,
                    stacklevel=3,
                )
        else:
            method = "conformal"
            lift_lower, lift_upper = self._conformal_interval(pre_resid, post_resid, q, resid_fn=resid_fn)
            calibration = self._cross_fit_residuals(actual_pre, t1)
            band = self._split_conformal_band(pre_resid if calibration is None else calibration)

        return {
            "p_value": p_value,
            "lift_ci_lower": lift_lower,
            "lift_ci_upper": lift_upper,
            "incrementality_ci_lower": lift_lower * t1,
            "incrementality_ci_upper": lift_upper * t1,
            "conformal_band": band,
            "method": method,
            "conformal_refit": resid_fn is not None,
        }

    def _refit_counterfactual(self, y_full: np.ndarray, train: np.ndarray) -> np.ndarray | None:
        """Refit the counterfactual on a subset of periods and predict every period.

        The hook behind the null-imposed conformal refits and the cross-fitted
        band. The default refits the donor weights through
        ``_fit_predict_weights`` on the cached pre- and post-period donor
        matrices (``_jk_x_pre`` / ``_jk_x_post``); an estimator with a different
        design overrides it. Returns ``None`` when refitting is unsupported.

        Parameters
        ----------
        y_full : numpy array, shape (T,)
            The treated series over every period (pre then post). Post-period
            values may be null-adjusted, or arbitrary when ``train`` excludes them.
        train : numpy array of bool, shape (T,)
            The periods to fit on.

        Returns
        -------
        The counterfactual over every period, shape (T,), or None.
        """
        if self._jk_x_pre is None or self._jk_x_post is None:
            return None
        x_full = np.vstack([np.asarray(self._jk_x_pre, dtype=float), np.asarray(self._jk_x_post, dtype=float)])
        if x_full.shape[0] != y_full.shape[0]:
            return None
        pred = self._fit_predict_weights(x_full[train], y_full[train], x_full)
        return None if pred is None else np.asarray(pred, dtype=float).ravel()

    def _null_residual_fn(
        self, actual_pre: Any, actual_post: Any
    ) -> Callable[[float], tuple[np.ndarray, np.ndarray]] | None:
        """Build the null-imposed residual function for the conformal test.

        Under the null of a constant per-period effect ``theta``, the post-period
        outcome without treatment is ``y - theta``, so the counterfactual is refit
        on all T periods of that series and the residuals are split back into
        pre- and post-period parts (Chernozhukov, Wuthrich & Zhu, 2021, Algorithm
        1). Results are cached per ``theta``.

        Parameters
        ----------
        actual_pre, actual_post : array-like
            Observed pre- and post-period treated series.

        Returns
        -------
        A function mapping ``theta`` to ``(pre_residuals, post_residuals)``, or
        None when the estimator cannot refit.
        """
        y_pre = np.asarray(self._counterfactual_series(actual_pre), dtype=float).ravel()
        y_post = np.asarray(self._counterfactual_series(actual_post), dtype=float).ravel()
        n_pre = y_pre.shape[0]
        train = np.ones(n_pre + y_post.shape[0], dtype=bool)
        cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

        def resid_fn(theta: float) -> tuple[np.ndarray, np.ndarray]:
            if theta not in cache:
                y_full = np.concatenate([y_pre, y_post - theta])
                cf = self._refit_counterfactual(y_full, train)
                if cf is None:
                    raise RuntimeError("counterfactual refit failed")
                resid = y_full - cf
                cache[theta] = (resid[:n_pre], resid[n_pre:])
            return cache[theta]

        try:
            resid_fn(0.0)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            return None
        return resid_fn

    def _cross_fit_residuals(self, actual_pre: Any, t1: int, n_folds: int = 10) -> np.ndarray | None:
        """Out-of-sample pre-period residuals from blocked cross-fitting.

        Splits the pre-period into up to ``n_folds`` contiguous blocks, refits
        the counterfactual on the other pre-period blocks and predicts the
        held-out one, so every residual comes from a model that did not see it.
        These calibrate the conformal prediction band.

        Parameters
        ----------
        actual_pre : array-like
            Observed pre-period treated series.
        t1 : int
            The number of post-period periods.
        n_folds : int, default=10
            The maximum number of blocks.

        Returns
        -------
        The held-out residuals, shape (n_pre,), or None when the estimator cannot
        refit.
        """
        y_pre = np.asarray(self._counterfactual_series(actual_pre), dtype=float).ravel()
        n_pre = y_pre.shape[0]
        if n_pre < 2:
            return None
        y_full = np.concatenate([y_pre, np.zeros(t1)])
        resid = np.empty(n_pre)
        for idx in np.array_split(np.arange(n_pre), min(n_folds, n_pre)):
            train = np.zeros(n_pre + t1, dtype=bool)
            train[:n_pre] = True
            train[idx] = False
            cf = self._refit_counterfactual(y_full, train)
            if cf is None:
                return None
            resid[idx] = y_pre[idx] - cf[idx]
        return resid

    # ------------------------------------------------------------------
    # Jackknife+ fallback
    #
    # When the pre-period is too short for the conformal tools above, we fall
    # back to jackknife+ (Barber, Candes, Ramdas & Tibshirani, 2021), which
    # reuses every pre-period point for both fitting and calibration via
    # leave-one-out and carries a distribution-free >= 1 - 2*alpha coverage
    # guarantee. The faithful version needs per-fold refits, supplied by an
    # estimator overriding ``_loo_counterfactuals``; without it we approximate
    # using the pre-period residuals with the counterfactual held fixed.
    # ------------------------------------------------------------------
    def _pre_period_too_short(self, n_pre: int, alpha: float | None = None) -> bool:
        """Whether the pre-period is too short for the conformal interval.

        Honours an explicit ``inference_method`` override; otherwise the
        pre-period is "too short" exactly when the split-conformal quantile level
        ``ceil((n + 1)(1 - alpha)) / n`` saturates at 1.0, the documented failure
        mode where the band collapses to the largest residual.

        Parameters
        ----------
        n_pre : int
            The number of pre-period points.
        alpha : float, optional
            The significance level. Defaults to ``self.alpha``.

        Returns
        -------
        True if the jackknife+ fallback should be used.
        """
        if self.inference_method == "jackknife":
            return True
        if self.inference_method == "conformal":
            return False
        if alpha is None:
            alpha = self.alpha
        return bool(np.ceil((n_pre + 1) * (1 - alpha)) > n_pre)

    def _loo_counterfactuals(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Leave-one-out pre-period residuals and post-period counterfactuals.

        Returns ``(loo_residuals, loo_post_predictions)`` where ``loo_residuals``
        has shape ``(n_folds,)`` (the absolute held-out pre-period residual of
        each fold) and ``loo_post_predictions`` has shape ``(n_folds, t1)`` (each
        leave-one-out model's counterfactual over the post-period). Returns
        ``None`` -- triggering the residual-only jackknife+ approximation -- when
        the estimator has not cached the donor matrices (``_jk_x_pre`` etc.) and
        implemented ``_fit_predict_weights``.

        Synthetic-control estimators all predict ``donor_matrix @ weights`` (with
        an intercept for the augmented method), so the leave-one-out loop is
        shared in ``_block_loo`` and each estimator only supplies its weight fit.

        Returns
        -------
        The leave-one-out arrays, or None if unsupported.
        """
        x_pre = self._jk_x_pre
        if x_pre is None or self._jk_y_pre is None or self._jk_x_post is None:
            return None
        return self._block_loo(x_pre, self._jk_y_pre, self._jk_x_post)

    def _fit_predict_weights(self, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray | None:
        """Fit this estimator's donor weights on a pre-period subset and predict.

        Default returns ``None`` (no faithful jackknife+). A synthetic-control
        estimator overrides this to fit its weights on ``(x_train, y_train)`` --
        a subset of the pre-period rows -- and return the counterfactual for the
        rows of ``x_eval`` (a held-out pre-period row stacked on the post-period
        donor matrix). Donor-column order matches the cached ``_jk_x_*`` matrices.

        Parameters
        ----------
        x_train : numpy array, shape (n_train, n_donors)
            Pre-period donor rows used to refit the weights.
        y_train : numpy array, shape (n_train,)
            Treated pre-period series on the same rows.
        x_eval : numpy array, shape (n_eval, n_donors)
            Donor rows to predict the counterfactual for.

        Returns
        -------
        The counterfactual for each ``x_eval`` row, or None if unsupported.
        """
        return None

    def _block_loo(
        self,
        x_pre: np.ndarray,
        y_pre: np.ndarray,
        x_post: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Run the leave-one-out refit loop over the pre-period via ``_fit_predict_weights``.

        For each pre-period row, refit the weights on the remaining rows and
        predict both the held-out row (its residual) and the post-period
        counterfactual. Folds the fit declines (returns None) are skipped. The
        synthetic-control weight fits are regularised or simplex-constrained, so
        they remain well posed when donors outnumber the training rows.

        Parameters
        ----------
        x_pre : numpy array, shape (n_pre, n_donors)
            Pre-period donor matrix.
        y_pre : numpy array, shape (n_pre,)
            Treated pre-period series.
        x_post : numpy array, shape (t1, n_donors)
            Post-period donor matrix.

        Returns
        -------
        ``(loo_residuals, loo_post_predictions)``, or None if no fold succeeded.
        """
        n = x_pre.shape[0]
        loo_resid: list[float] = []
        loo_post: list[np.ndarray] = []
        for i in range(n):
            mask = np.arange(n) != i
            eval_mat = np.vstack([x_pre[i][None, :], x_post])
            pred = self._fit_predict_weights(x_pre[mask], y_pre[mask], eval_mat)
            if pred is None:
                continue
            loo_resid.append(abs(float(y_pre[i] - pred[0])))
            loo_post.append(np.asarray(pred[1:], dtype=float))
        if not loo_resid:
            return None
        return np.array(loo_resid), np.array(loo_post)

    @staticmethod
    def _jackknife_quantile(abs_residuals: np.ndarray, alpha: float) -> float:
        """Return the jackknife+ ``(1 - alpha)`` quantile of absolute residuals.

        Uses the ``ceil((1 - alpha)(n + 1))``-th smallest absolute residual.
        When that rank exceeds ``n`` there are too few residuals for a
        ``(1 - alpha)`` guarantee and the quantile is infinite (Barber et al.,
        2021); truncating to the largest residual would under-cover.

        Parameters
        ----------
        abs_residuals : numpy array
            Absolute residuals.
        alpha : float
            The significance level.

        Returns
        -------
        The band half-width.
        """
        scores = np.sort(np.abs(abs_residuals))
        n = scores.shape[0]
        k = int(np.ceil((1 - alpha) * (n + 1)))
        if k > n:
            return float("inf")
        return float(scores[k - 1])

    def _jackknife_residual_interval(
        self,
        pre_resid: np.ndarray,
        post_resid: np.ndarray,
        alpha: float | None = None,
    ) -> tuple[float, float, float]:
        """Residual-only jackknife+ interval for a constant per-period effect.

        Holds the counterfactual fixed and centres the interval on the mean
        post-period residual. The interval is for that *mean*, so it is
        calibrated on the means of every cyclic block of ``t1`` consecutive
        pre-period residuals (as in the moving-block conformal test), which keeps
        their autocorrelation. A per-observation residual quantile would treat
        the mean of ``t1`` residuals as one residual and be about ``sqrt(t1)``
        times too wide under independence. The returned ``band`` is still the
        per-period quantile, for the pointwise prediction band.

        Parameters
        ----------
        pre_resid, post_resid : numpy array
            Pre- and post-period residuals.
        alpha : float, optional
            The significance level. Defaults to ``self.alpha``.

        Returns
        -------
        A tuple of (lift_lower, lift_upper, band_half_width).
        """
        if alpha is None:
            alpha = self.alpha
        band = self._jackknife_quantile(pre_resid, alpha)
        center = float(np.mean(post_resid))
        t1 = post_resid.shape[0]
        n = pre_resid.shape[0]
        offsets = np.arange(t1)
        block_means = np.array([np.mean(pre_resid[(start + offsets) % n]) for start in range(n)])
        half_width = self._jackknife_quantile(block_means, alpha)
        return center - half_width, center + half_width, band

    def _jackknife_plus_interval(
        self,
        loo_resid: np.ndarray,
        loo_post_pred: np.ndarray,
        actual_post: np.ndarray,
        t1: int,
        alpha: float | None = None,
    ) -> tuple[float, float, float]:
        """Faithful jackknife+ interval from leave-one-out refits.

        For each post-period point the counterfactual bounds combine every
        leave-one-out model's prediction at that point with that model's held-out
        residual; the per-period lift bounds are aggregated into a per-period
        effect CI (so ``lift_ci * t1`` recovers the incrementality CI).

        Each fold holds out a single pre-period point, so only per-point
        residuals are available for calibration. Summing per-period bounds is
        therefore deliberately conservative for the mean effect (roughly
        ``sqrt(t1)`` too wide under independent errors); narrowing it would need
        an assumption about the residuals' dependence that the per-point folds
        cannot check. The residual-only fallback, which has the whole residual
        series, calibrates on block means instead.

        Parameters
        ----------
        loo_resid : numpy array, shape (n_folds,)
            Absolute held-out pre-period residual of each fold.
        loo_post_pred : numpy array, shape (n_folds, t1)
            Each leave-one-out model's post-period counterfactual.
        actual_post : numpy array, shape (t1,)
            Observed post-period outcome.
        t1 : int
            The number of post-period points.
        alpha : float, optional
            The significance level. Defaults to ``self.alpha``.

        Returns
        -------
        A tuple of (lift_lower, lift_upper, band_half_width).
        """
        if alpha is None:
            alpha = self.alpha
        n = loo_resid.shape[0]
        lo_rank = int(np.floor(alpha * (n + 1)))
        hi_rank = int(np.ceil((1 - alpha) * (n + 1)))
        band = self._jackknife_quantile(loo_resid, alpha)
        if lo_rank < 1 or hi_rank > n:
            # Too few folds for the jackknife+ quantiles: the interval is
            # unbounded (Barber et al., 2021). Clamping to the extreme folds
            # would drop the 1 - 2 * alpha coverage guarantee.
            return float("-inf"), float("inf"), band
        lo_idx, hi_idx = lo_rank - 1, hi_rank - 1
        incr_lower, incr_upper = 0.0, 0.0
        for t in range(t1):
            cf_lower = np.sort(loo_post_pred[:, t] - loo_resid)[lo_idx]
            cf_upper = np.sort(loo_post_pred[:, t] + loo_resid)[hi_idx]
            incr_lower += actual_post[t] - cf_upper
            incr_upper += actual_post[t] - cf_lower
        return incr_lower / t1, incr_upper / t1, band

    # ------------------------------------------------------------------
    # Parametric bootstrap
    #
    # Mirrors GeoLift's Generalized Synthetic Control inference: hold the
    # estimated counterfactual fixed, draw parametric noise at the pre-period
    # residual scale, rebuild pseudo treated pre-period series, refit the
    # counterfactual (via ``_bootstrap_refit``) and accumulate an incrementality
    # distribution. The CI is the percentile interval; the p-value is the
    # two-sided proportion of no-effect replicates at least as extreme as the
    # observed incrementality. Needs a refit hook, so estimators without one fall
    # back to the conformal / jackknife+ path.
    # ------------------------------------------------------------------
    def _bootstrap_refit(self, treated_pre: np.ndarray) -> np.ndarray | None:
        """Refit the counterfactual over the post-period from a resampled pre-period.

        Delegates to ``_fit_predict_weights``: refit the weights on the cached
        pre-period donor matrix against the resampled treated series and predict
        the post-period donor matrix. Synthetic-control estimators get the
        parametric bootstrap for free this way; an estimator that caches a
        different design (e.g. the factor design) overrides this directly. Returns
        ``None`` -- so the bootstrap falls back -- when the donor matrices are not
        cached or the weight fit is unsupported.

        Parameters
        ----------
        treated_pre : numpy array, shape (n_pre,)
            A resampled treated pre-period series.

        Returns
        -------
        The refit post-period counterfactual, or None if unsupported.
        """
        if self._jk_x_pre is None or self._jk_x_post is None:
            return None
        return self._fit_predict_weights(self._jk_x_pre, treated_pre, self._jk_x_post)

    def _parametric_bootstrap_interval(
        self,
        fitted_pre: np.ndarray,
        fitted_post: np.ndarray,
        actual_post: np.ndarray,
        pre_resid: np.ndarray,
        alpha: float | None = None,
    ) -> tuple[float, float, float, float] | None:
        """Parametric-bootstrap interval and p-value for the incrementality.

        Parameters
        ----------
        fitted_pre, fitted_post : numpy array
            The fitted counterfactual over the pre- and post-period.
        actual_post : numpy array
            Observed post-period outcome.
        pre_resid : numpy array
            Pre-period residuals, whose scale sets the parametric noise.
        alpha : float, optional
            The significance level. Defaults to ``self.alpha``.

        Returns
        -------
        A tuple of (lift_lower, lift_upper, band_half_width, p_value), or None if
        the estimator does not implement ``_bootstrap_refit``.
        """
        if alpha is None:
            alpha = self.alpha
        if self._bootstrap_refit(fitted_pre) is None:
            return None

        sigma = float(np.std(pre_resid, ddof=1)) if pre_resid.shape[0] > 1 else 0.0
        if sigma == 0.0:
            sigma = 1.0
        n_pre, t1 = fitted_pre.shape[0], actual_post.shape[0]
        rng = np.random.default_rng(self.bootstrap_seed)
        incr_obs = float(np.sum(actual_post - fitted_post))

        incr_samples = np.empty(self.n_boot)
        null_samples = np.empty(self.n_boot)
        for b in range(self.n_boot):
            cf_post = self._bootstrap_refit(fitted_pre + rng.normal(0.0, sigma, n_pre))
            eps_post = rng.normal(0.0, sigma, t1)
            # Replicate of the observed incrementality (counterfactual + post noise).
            incr_samples[b] = float(np.sum(actual_post - cf_post - eps_post))
            # Replicate under the sharp null of no effect (observed == counterfactual).
            null_samples[b] = float(np.sum(fitted_post + eps_post - cf_post))

        # Center the bootstrap spread on the reported incrementality. A refit on
        # the fitted counterfactual need not reproduce it exactly for penalised or
        # mean-based estimators, which would otherwise bias the percentile
        # interval off the point estimate; centering on incr_obs de-biases it and
        # guarantees the interval brackets the reported lift.
        boot_mean = float(np.mean(incr_samples))
        lower = incr_obs + float(np.percentile(incr_samples, 100 * alpha / 2)) - boot_mean
        upper = incr_obs + float(np.percentile(incr_samples, 100 * (1 - alpha / 2))) - boot_mean
        band = self._jackknife_quantile(pre_resid, alpha)
        # The null replicates carry the same refit bias (their mean is
        # sum(fitted_post - E[refit])), so center them on zero before comparing
        # with the observed incrementality, and count the observed value itself
        # (+1) so the p-value is never exactly zero.
        null_samples = null_samples - float(np.mean(null_samples))
        self._bootstrap_null = null_samples
        n_extreme = int(np.sum(np.abs(null_samples) >= abs(incr_obs)))
        p_value = (n_extreme + 1) / (self.n_boot + 1)
        return lower / t1, upper / t1, band, p_value

    # ------------------------------------------------------------------
    # Sum-to-one donor weights
    #
    # Synthetic-control weights constrained to sum to one cannot reproduce a
    # treated series that sums several geos (the donors' common dynamics would
    # need to be scaled up by the number of treated geos). These estimators
    # therefore fit the per-geo average treated series, as augsynth and
    # Arkhangelsky et al. do, and scale the weights back up by the treated
    # count. The same helper rescales the problem to unit magnitude so SLSQP's
    # absolute tolerances do not stop it early on outcomes in the thousands.
    # ------------------------------------------------------------------
    def _per_geo_problem(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Rescale a sum-to-one weight fit to the per-geo, unit-magnitude problem.

        Parameters
        ----------
        x : numpy array, shape (n_rows, n_donors)
            Donor outcomes.
        y : numpy array, shape (n_rows,)
            The treated series (a sum over the treated geos).

        Returns
        -------
        A tuple ``(x / s, y / (n * s), n)`` where ``n`` is the number of treated
        geos and ``s`` the largest absolute donor value. Weights fitted on the
        rescaled problem are multiplied by ``n`` to predict the summed series.
        """
        n = float(self._n_treated)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float) / n
        scale = float(np.max(np.abs(x))) if x.size else 1.0
        if not np.isfinite(scale) or scale == 0.0:
            scale = 1.0
        return x / scale, y / scale, n

    def _simplex_fit(self, x: np.ndarray, y: np.ndarray, name: str) -> np.ndarray:
        """Simplex-constrained least-squares donor weights for the summed treated series.

        Solves ``min ||y / n - x w||^2`` subject to ``w >= 0, sum(w) = 1`` on the
        rescaled per-geo problem (see ``_per_geo_problem``) and returns ``n * w``.

        Parameters
        ----------
        x : numpy array, shape (n_rows, n_donors)
            Donor outcomes.
        y : numpy array, shape (n_rows,)
            The treated series.
        name : str
            The estimator name for the convergence warning.

        Returns
        -------
        The donor weights, summing to the number of treated geos.
        """
        xs, ys, n_treated = self._per_geo_problem(x, y)
        weights = self._solve_simplex_lsq(xs, ys, name)
        return weights * n_treated

    # Weight on the extra NNLS row that enforces ``sum(w) = 1``, relative to the
    # largest entry of ``x``. The row is a penalty, not a hard constraint: before
    # renormalising, the sum misses 1 by roughly ``n_rows * |residual| / (1e8 *
    # max|x|)``, up to ~1e-3 when ``y`` is 1e3-1e4x the donor scale.
    # Renormalising restores the sum; the weights are then exact when the
    # optimum is a vertex of the simplex and otherwise drift slightly from the
    # constrained optimum. SC and ASC go through ``_per_geo_problem``,
    # which puts ``y`` on the donor scale, and there the fit matches a tightly
    # converged SLSQP to ~1e-14 in relative loss. A larger weight would tighten
    # this but ill-condition the augmented matrix.
    _SIMPLEX_SUM_WEIGHT = 1e4

    def _solve_simplex_lsq(self, x: np.ndarray, y: np.ndarray, name: str, ridge: float = 0.0) -> np.ndarray:
        """Least-squares weights on the probability simplex.

        Solves ``min_w ||y - x w||^2 + ridge * ||w||^2`` subject to ``w >= 0,
        sum(w) = 1`` as a non-negative least-squares problem: the ridge becomes
        ``sqrt(ridge) * I`` rows with zero targets, and the sum constraint a
        heavily weighted row of ones with target one. NNLS works on ``x`` itself,
        so it never squares the conditioning of (often highly collinear) donor
        series the way the Gram-matrix form does, and its active-set solve is
        orders of magnitude faster than SLSQP on large donor pools. If NNLS hits
        its iteration limit the problem falls back to ``_solve_simplex_qp``.
        Non-finite inputs raise a ``ValueError`` rather than returning
        meaningless weights.

        Parameters
        ----------
        x : numpy array, shape (n_rows, n)
            The design matrix (e.g. donor outcomes, one column per donor).
        y : numpy array, shape (n_rows,)
            The target series.
        name : str
            The estimator name for the fallback's convergence warning.
        ridge : float, default=0.0
            The L2 penalty on the weights.

        Returns
        -------
        The simplex weights.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError(
                f"{name}: the panel has missing or non-finite outcome values (e.g. a geo with no row "
                "for some date, which pivots to NaN). Fill or drop those geo-date cells so every geo "
                "has a finite value on every date."
            )
        n_c = x.shape[1]
        if ridge > 0.0:
            x = np.vstack([x, np.sqrt(ridge) * np.identity(n_c)])
            y = np.concatenate([y, np.zeros(n_c)])
        level = float(np.max(np.abs(x))) if x.size else 0.0
        weight = self._SIMPLEX_SUM_WEIGHT * (level if np.isfinite(level) and level > 0.0 else 1.0)
        try:
            weights, _ = nnls(np.vstack([x, np.full(n_c, weight)]), np.append(y, weight), maxiter=50 * n_c)
        except RuntimeError:
            gram = x.T @ x
            return self._solve_simplex_qp(gram, x.T @ y, float(y @ y), name)
        total = weights.sum()
        return weights / total if total > 0 else np.full(n_c, 1.0 / n_c)

    def _solve_simplex_qp(
        self,
        gram: np.ndarray,
        cross: np.ndarray,
        const: float,
        name: str,
        linear: np.ndarray | None = None,
    ) -> np.ndarray:
        """Minimise a convex quadratic over the probability simplex.

        Solves ``min_w w' G w - 2 c' w + const + l' w`` subject to ``w >= 0,
        sum(w) = 1``. ``const`` makes the objective a (non-negative) loss so it
        can be divided by its value at the uniform start: SLSQP's ``ftol`` is an
        absolute tolerance on the objective, so on a rescaled problem whose loss
        is ~1e-5 the default ``1e-6`` stops after a step or two, still near the
        start, while reporting success. Normalising to 1 at the start and using a
        tight tolerance with the analytic gradient makes the stop relative.

        Parameters
        ----------
        gram : numpy array, shape (n, n)
            The quadratic term ``G`` (e.g. ``X'X``).
        cross : numpy array, shape (n,)
            The linear term ``c`` (e.g. ``X'y``).
        const : float
            The constant term (e.g. ``y'y``).
        name : str
            The estimator name for the convergence warning.
        linear : numpy array, shape (n,), optional
            An additional linear penalty ``l`` (e.g. a pairwise-discrepancy
            penalty).

        Returns
        -------
        The simplex weights.
        """
        n_c = gram.shape[0]
        lin = -2.0 * cross if linear is None else linear - 2.0 * cross
        x0 = np.full(n_c, 1.0 / n_c)
        start = float(x0 @ gram @ x0 + lin @ x0 + const)
        scale = start if np.isfinite(start) and start > 0.0 else 1.0
        res = minimize(
            fun=lambda w: float(w @ gram @ w + lin @ w + const) / scale,
            x0=x0,
            jac=lambda w: (2.0 * gram @ w + lin) / scale,
            bounds=Bounds(lb=np.zeros(n_c), ub=np.ones(n_c)),
            constraints=LinearConstraint(A=np.ones(n_c), lb=1.0, ub=1.0),
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        self._warn_unconverged(res, name)
        # SLSQP can step marginally outside the bounds; project back.
        weights = np.clip(res["x"], 0.0, None)
        total = weights.sum()
        return weights / total if total > 0 else x0

    @staticmethod
    def _warn_unconverged(res: Any, name: str) -> None:
        """Warn when a scipy optimizer did not report success.

        Parameters
        ----------
        res : scipy.optimize.OptimizeResult
            The optimizer result.
        name : str
            The estimator name for the message.
        """
        if not res.success:
            warnings.warn(
                f"{name}: weight optimizer did not converge ({res.message}); the fitted weights may be unreliable",
                RuntimeWarning,
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    # Shared factor-model helpers
    #
    # The factor-based estimators (``GeneralizedSyntheticControl`` and
    # ``InteractiveFixedEffects``) both solve least squares against a fitted
    # factor design and both pick the factor count from the control panel's
    # spectrum, so those two operations live here rather than being duplicated.
    # ------------------------------------------------------------------
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

    @staticmethod
    def _eigenvalue_ratio_factors(singular_values: np.ndarray, max_r: int) -> int:
        """Choose the factor count by the eigenvalue-ratio criterion.

        Selects the number of latent factors from the control panel's own spectrum
        (Ahn & Horenstein, 2013): with eigenvalues ``mu_k`` (the squared singular
        values of the centred control matrix in descending order), the estimated
        factor count is the ``k`` in ``1..max_r`` that maximises the adjacent
        ratio ``mu_k / mu_{k+1}``. A genuine factor leaves a large gap before the
        noise eigenvalues, so the ratio spikes at the true count.

        This deliberately replaces a treated-pre-period cross-validation: minimising
        held-out *treated* error rewards ever-richer factor subspaces that fit the
        pre-period but overfit the counterfactual (they reconstruct the treated
        unit's post-period and wash out the effect). The factor *count* is a
        property of the donor panel's covariance, so we estimate it there instead.

        Parameters
        ----------
        singular_values : numpy array
            Singular values of the centred control matrix, descending.
        max_r : int
            The largest factor count to consider.

        Returns
        -------
        The selected factor count, in ``1..max_r``; ``0`` only when ``max_r < 1``.
        The ratio criterion cannot itself select a zero-factor model.
        """
        if max_r < 1:
            return 0
        eig = np.asarray(singular_values, dtype=float)[: max_r + 1] ** 2
        if eig.shape[0] < 2:
            return min(1, max_r)
        denom = np.where(eig[1:] <= 0.0, np.finfo(float).tiny, eig[1:])
        ratios = eig[:-1] / denom  # ratios[k-1] = mu_k / mu_{k+1} for k = 1..len
        return int(np.argmax(ratios)) + 1

    # ------------------------------------------------------------------
    # Shared counterfactual plot
    #
    # Every synthetic-control style estimator draws the same three panels
    # (actual vs counterfactual, pointwise difference, cumulative difference)
    # and carries the same inference outputs, so the figure -- including the
    # confidence bands -- is built once here and each estimator's ``plot()``
    # just supplies its series as arrays.
    # ------------------------------------------------------------------
    def _plot_counterfactual(
        self,
        dates: list[Any],
        actual_pre: np.ndarray,
        actual_post: np.ndarray,
        prediction_pre: np.ndarray,
        prediction_post: np.ndarray,
        dark_mode: bool = False,
    ) -> None:
        """Build and show the three-panel counterfactual figure with confidence bands.

        The top panel shows the actual and counterfactual series with the pointwise
        prediction band (``results["conformal_band"]``) shaded around the
        counterfactual; the middle panel shows the pointwise difference with the
        same band shaded around zero (points outside it are the significant
        per-period effects); the bottom panel shows the cumulative post-period
        difference with a band that grows linearly to the reported incrementality
        interval, so its endpoint matches ``summarize()``.

        Parameters
        ----------
        dates : list
            The full (pre- then post-period) date axis.
        actual_pre, actual_post : numpy array
            Observed outcome over the pre- and post-period.
        prediction_pre, prediction_post : numpy array
            Counterfactual outcome over the pre- and post-period.
        dark_mode : bool, default=False
            Render with a dark background and lighter series colours.
        """
        if self.results is None:
            raise ValueError("results must not be None")
        actual_pre = np.asarray(actual_pre, dtype=float).ravel()
        actual_post = np.asarray(actual_post, dtype=float).ravel()
        prediction_pre = np.asarray(prediction_pre, dtype=float).ravel()
        prediction_post = np.asarray(prediction_post, dtype=float).ravel()
        actual = np.concatenate([actual_pre, actual_post])
        prediction = np.concatenate([prediction_pre, prediction_post])
        residuals = actual - prediction
        t1 = actual_post.shape[0]

        band_value = self.results.get("conformal_band")
        band = float(band_value) if band_value is not None else 0.0
        has_band = bool(np.isfinite(band)) and band > 0
        theme = get_theme(dark_mode)
        band_fill = theme.band(0.2)
        ci_label = f"{self._get_ci_print()} CI"

        total_fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(
                "Expected vs Counterfactual",
                "Pointwise Difference",
                "Cumulative Difference",
            ),
        )

        # Top panel: band around the counterfactual, then the two series on top.
        if has_band:
            total_fig.add_trace(
                go.Scatter(x=dates, y=prediction + band, mode="lines", line={"width": 0}, showlegend=False),
                row=1,
                col=1,
            )
            total_fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=prediction - band,
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor=band_fill,
                    name=ci_label,
                ),
                row=1,
                col=1,
            )
        total_fig.add_trace(
            go.Scatter(x=dates, y=actual, marker={"color": theme.actual}, mode="lines", name="Actual"),
            row=1,
            col=1,
        )
        total_fig.add_trace(
            go.Scatter(
                x=dates, y=prediction, marker={"color": theme.counterfactual}, mode="lines", name="Counterfactual"
            ),
            row=1,
            col=1,
        )

        # Middle panel: the pointwise band around zero (the no-effect region).
        if has_band:
            total_fig.add_trace(
                go.Scatter(x=dates, y=np.full(len(dates), band), mode="lines", line={"width": 0}, showlegend=False),
                row=2,
                col=1,
            )
            total_fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=np.full(len(dates), -band),
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor=band_fill,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        total_fig.add_trace(
            go.Scatter(x=dates, y=residuals, marker={"color": theme.difference}, mode="lines", name="Difference"),
            row=2,
            col=1,
        )

        # Bottom panel: cumulative post-period difference with a band that grows
        # linearly to the reported incrementality interval.
        post_key = period_key(self.post_period, self.data, self.date_variable)
        marketing_start = [d for d in dates if period_key(d, self.data, self.date_variable) >= post_key]
        cum_resids = np.cumsum(actual_post - prediction_post)
        incr = self.results.get("incrementality")
        incr_lo = self.results.get("incrementality_ci_lower")
        incr_hi = self.results.get("incrementality_ci_upper")
        if incr is not None and incr_lo is not None and incr_hi is not None and t1 > 0:
            frac = np.arange(1, t1 + 1, dtype=float) / t1
            cum_upper = cum_resids + frac * (float(incr_hi) - float(incr))
            cum_lower = cum_resids + frac * (float(incr_lo) - float(incr))
            total_fig.add_trace(
                go.Scatter(x=marketing_start, y=cum_upper, mode="lines", line={"width": 0}, showlegend=False),
                row=3,
                col=1,
            )
            total_fig.add_trace(
                go.Scatter(
                    x=marketing_start,
                    y=cum_lower,
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor=theme.cumulative_band(0.2),
                    name=ci_label,
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
        total_fig.add_trace(
            go.Scatter(
                x=marketing_start,
                y=cum_resids,
                marker={"color": theme.cumulative},
                mode="lines",
                name="Cumulative Incrementality",
            ),
            row=3,
            col=1,
        )

        for row in (1, 2, 3):
            total_fig.add_vline(
                x=self.post_period,
                line_width=1,
                line_dash="dash",
                line_color=theme.reference,
                row=row,
                col=1,
            )
        apply_theme(total_fig, theme)
        total_fig.show()

    def _counterfactual_series(self, series: Any) -> np.ndarray:
        """Coerce a cached actual/counterfactual series to a flat numpy array.

        Synthetic-control style estimators cache these series either as numpy
        arrays (``SyntheticControl``, ``CausalImpact``, ``SyntheticDiffInDiff``)
        or as narwhals frames with a single outcome column (the V-weighted,
        penalised and factor estimators). This returns a 1-D array for either.

        Parameters
        ----------
        series : numpy array or narwhals frame
            A cached actual or counterfactual series.

        Returns
        -------
        The series as a flat numpy array.
        """
        if hasattr(series, "columns"):
            return series[self.y_variable].to_numpy()
        return np.asarray(series)

    def _series_frame(self, dates: list[Any], values: np.ndarray) -> nw.DataFrame:
        """Wrap a date axis and value array into a narwhals frame.

        Parameters
        ----------
        dates : list
            The date axis.
        values : numpy array
            The values aligned to ``dates``.

        Returns
        -------
        A narwhals data frame with the date and outcome columns.
        """
        return nw.from_native(  # type: ignore[call-overload]
            pl.DataFrame({self.date_variable: dates, self.y_variable: np.asarray(values, dtype=float)}),
            eager_only=True,
        )

    # ------------------------------------------------------------------
    # Randomization inference (Abadie-style placebo permutation test)
    # ------------------------------------------------------------------
    def randomization_test(
        self,
        *,
        statistic: str = "avg_lift",
        seed: int | None = None,
        max_placebos: int | None = None,
        n_jobs: int = 1,
    ) -> dict[str, Any]:
        """Placebo permutation test in the style of Abadie, Diamond & Hainmueller (2010).

        Each placebo reassigns treatment to a set of donor (control) geos the
        same size as the real treated group, while the remaining donors form
        the control pool. The originally treated geos are excluded from every
        placebo, since their post-period outcomes carry the real effect. With
        one treated geo every donor is a placebo in turn; with ``k > 1`` the
        placebos are ``k``-subsets of the donors (all of them when there are at
        most ``max_placebos``, otherwise a random sample), so the placebo and
        observed statistics are on the same scale. The full
        ``pre_process().generate()`` pipeline is re-run for every placebo with
        the estimator's own hyperparameters, producing a null distribution of
        treatment effect statistics.  The rank-based p-value is the fraction
        of placebos whose absolute statistic is at least as large as the
        observed one.

        Parameters
        ----------
        statistic : {"avg_lift", "sum_lift", "mspe_ratio"}, default "avg_lift"
            The test statistic.

            * ``"avg_lift"`` — mean of the per-period lift (matches the
              Proposition 99 literature).
            * ``"sum_lift"`` — total incrementality.
            * ``"mspe_ratio"`` — ratio of the post-period MSPE to the
              pre-period MSPE, as in Abadie, Diamond & Hainmueller (2015).
              Placebos with poor pre-period fit are down-weighted
              automatically by this ratio, so it does not need an explicit
              fit filter.
        seed : int, optional
            Seed controlling which placebos are drawn when they are sampled
            rather than enumerated.
        max_placebos : int, optional
            Cap the number of placebos to run. When there are more candidate
            placebos than this, a random subset is drawn. Defaults to every
            donor with one treated geo, and to 200 donor subsets otherwise.
        n_jobs : int, default 1
            Number of parallel workers for the placebo loop. ``-1`` uses all
            available cores.  ``1`` (the default) runs sequentially with no
            parallelism overhead.

        Returns
        -------
        dict with keys:

        * ``observed_statistic`` — the test statistic on the real treated unit.
        * ``placebo_statistics`` — list of per-geo placebo statistics.
        * ``placebo_geos`` — the corresponding geo labels (a tuple of labels
          per placebo when more than one geo is treated).
        * ``p_value`` — two-sided rank-based p-value.
        * ``n_placebos`` — number of successful placebos.
        * ``n_failed`` — number of placebos that raised during fitting.
        * ``statistic`` — name of the test statistic used.

        Raises
        ------
        ValueError
            If ``generate()`` has not been called, or no control geos are
            available in the data.
        """
        current_results = self.results
        if current_results is None:
            raise ValueError("Call generate() before randomization_test()")
        valid_stats = ("avg_lift", "sum_lift", "mspe_ratio")
        if statistic not in valid_stats:
            raise ValueError(f"statistic must be one of {valid_stats}, got {statistic!r}")

        results: dict[str, Any] = current_results
        observed = self._ri_statistic(results, statistic)

        all_geos: list[str] = sorted(self.data[self.geo_variable].unique().to_list())
        treat_var: str = self.treatment_variable or "is_treatment"
        test_geos_set: set[str] = set(
            self.test_geos
            if self.test_geos is not None
            else self.data.filter(nw.col(treat_var) == 1)[self.geo_variable].unique().to_list()
        )
        donor_geos = [g for g in all_geos if g not in test_geos_set]
        placebo_sets = self._placebo_sets(donor_geos, len(test_geos_set), max_placebos, seed)

        native_data = self.data.to_native()

        def _fit_placebo(placebo_set: tuple[str, ...]) -> tuple[Any, float] | None:
            placebo_control = [g for g in donor_geos if g not in placebo_set]
            label: Any = placebo_set[0] if len(placebo_set) == 1 else placebo_set
            try:
                m = self._clone(
                    data=native_data,
                    test_geos=list(placebo_set),
                    control_geos=placebo_control,
                )
                m.pre_process().generate()
                r = m.results
                if r is None:
                    return None
                return (label, self._ri_statistic(r, statistic))
            except Exception:
                return None

        if n_jobs == 1:
            raw = [_fit_placebo(s) for s in placebo_sets]
        else:
            from joblib import Parallel, delayed

            raw = Parallel(n_jobs=n_jobs)(delayed(_fit_placebo)(s) for s in placebo_sets)

        placebo_stats: list[float] = []
        placebo_geos: list[Any] = []
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
            results["randomization_test"] = result
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
        results["randomization_test"] = result
        return result

    @staticmethod
    def _ri_statistic(results: dict[str, Any], statistic: str) -> float:
        """Extract the test statistic from a results dict."""
        lift = np.asarray(results["lift"], dtype=float).ravel()
        if statistic == "avg_lift":
            return float(np.mean(lift))
        elif statistic == "sum_lift":
            return float(np.sum(lift))
        else:
            actual_pre = results.get("_actual_pre")
            prediction_pre = results.get("_prediction_pre")
            if actual_pre is not None and prediction_pre is not None:
                pre_resid = np.asarray(actual_pre, dtype=float) - np.asarray(prediction_pre, dtype=float)
                mspe_pre = float(np.mean(pre_resid**2))
            else:
                mspe_pre = 1.0
            actual_post = np.asarray(results["test"], dtype=float)
            counterfactual_post = np.asarray(results["counterfactual"], dtype=float)
            post_resid = actual_post - counterfactual_post
            mspe_post = float(np.mean(post_resid**2))
            if mspe_pre == 0:
                return float("inf") if mspe_post > 0 else 0.0
            return mspe_post / mspe_pre

    # ------------------------------------------------------------------
    # Model diagnostics
    # ------------------------------------------------------------------
    def diagnose(self) -> dict[str, Any]:
        """Run a battery of pre-period diagnostics on a fitted model.

        Returns a dictionary with the following keys:

        * ``pre_mape`` — mean absolute percentage error of the
          counterfactual fit over the pre-period.
        * ``pre_rmse`` — root mean squared error of the pre-period fit.
        * ``durbin_watson`` — Durbin-Watson statistic of the pre-period
          residuals (values near 2 indicate no autocorrelation;
          significantly below 2 suggests positive autocorrelation).
        * ``residual_stationarity`` — dict with ``adf_statistic``,
          ``adf_p_value`` and ``is_stationary`` (at the 5% level) from
          the augmented Dickey-Fuller test on the pre-period residuals.
        * ``placebo_in_time`` — dict from a placebo-in-time test that
          splits the pre-period at 70%, re-fits, and reports ``lift``
          and ``p_value`` on the held-out fake post-period. ``None``
          when the pre-period is too short to split.

        Raises
        ------
        ValueError
            If ``generate()`` has not been called.
        """
        current_results = self.results
        if current_results is None:
            raise ValueError("Call generate() before diagnose()")

        actual_pre = np.asarray(current_results["_actual_pre"], dtype=float).ravel()
        prediction_pre = np.asarray(current_results["_prediction_pre"], dtype=float).ravel()
        pre_resid = actual_pre - prediction_pre

        pre_rmse = float(np.sqrt(np.mean(pre_resid**2)))
        safe_actual = np.where(actual_pre != 0, actual_pre, np.nan)
        pre_mape = float(np.nanmean(np.abs(pre_resid / safe_actual)))

        dw_num = float(np.sum(np.diff(pre_resid) ** 2))
        dw_den = float(np.sum(pre_resid**2))
        durbin_watson = dw_num / dw_den if dw_den > 0 else float("nan")

        from statsmodels.tsa.stattools import adfuller

        n_pre = len(pre_resid)
        if n_pre >= 6:
            adf_result = adfuller(pre_resid, maxlag=max(1, n_pre // 4 - 1), autolag="AIC")
            residual_stationarity: dict[str, Any] = {
                "adf_statistic": float(adf_result[0]),
                "adf_p_value": float(adf_result[1]),
                "is_stationary": float(adf_result[1]) < 0.05,
            }
        else:
            residual_stationarity = {
                "adf_statistic": float("nan"),
                "adf_p_value": float("nan"),
                "is_stationary": None,
            }

        placebo_in_time = self._placebo_in_time()

        diag: dict[str, Any] = {
            "pre_mape": pre_mape,
            "pre_rmse": pre_rmse,
            "durbin_watson": durbin_watson,
            "residual_stationarity": residual_stationarity,
            "placebo_in_time": placebo_in_time,
        }
        current_results["diagnostics"] = diag
        return diag

    def _placebo_in_time(self) -> dict[str, Any] | None:
        """Split the pre-period and re-fit on the first 70% as a placebo test.

        The placebo sees only pre-period data, so its fake post-period is the
        last 30% of the real pre-period. Returns ``None`` when the pre-period is
        too short (< 10 dates) to produce a meaningful split.
        """
        # Only the real pre-period goes to the placebo fit: passing the full panel
        # would make the fake post-period run on through the real post-period and
        # pick up the real treatment effect.
        pre_data = self.data.filter(nw.col("treatment_period") == 0)
        pre_dates = sorted(pre_data[self.date_variable].unique().to_list())
        if len(pre_dates) < 10:
            return None

        split_idx = int(len(pre_dates) * 0.7)
        fake_pre = period_key(pre_dates[split_idx - 1])
        fake_post = period_key(pre_dates[split_idx])

        try:
            placebo = self._clone(
                data=pre_data.to_native(),
                pre_period=fake_pre,
                post_period=fake_post,
            )
            placebo.pre_process().generate()
            r = placebo.results
            if r is None:
                return None
            return {
                "lift": float(np.mean(np.asarray(r["lift"], dtype=float))),
                "p_value": float(r["p_value"]),
                "fake_pre": fake_pre,
                "fake_post": fake_post,
            }
        except Exception:
            return None

    def plot(self, dark_mode: bool = False) -> None:
        """Plot the actual series, counterfactual, and pointwise and cumulative differences.

        Draws the shared three-panel counterfactual figure (see
        ``_plot_counterfactual``) from the ``actual_*``/``prediction_*`` series
        each synthetic-control style estimator caches in ``generate``.

        Parameters
        ----------
        dark_mode : bool, default=False
            Render with a dark background and lighter series colours.
        """
        # Read via getattr: these series are declared and cached on the concrete
        # synthetic-control subclasses, not on this base.
        dates = getattr(self, "dates", None)
        actual_pre = getattr(self, "actual_pre", None)
        actual_post = getattr(self, "actual_post", None)
        prediction_pre = getattr(self, "prediction_pre", None)
        prediction_post = getattr(self, "prediction_post", None)
        if actual_pre is None:
            raise ValueError("actual_pre must not be None")
        if actual_post is None:
            raise ValueError("actual_post must not be None")
        if prediction_pre is None:
            raise ValueError("prediction_pre must not be None")
        if prediction_post is None:
            raise ValueError("prediction_post must not be None")
        if dates is None:
            raise ValueError("dates must not be None")
        self._plot_counterfactual(
            dates,
            self._counterfactual_series(actual_pre),
            self._counterfactual_series(actual_post),
            self._counterfactual_series(prediction_pre),
            self._counterfactual_series(prediction_post),
            dark_mode=dark_mode,
        )


class NonLinearEstimator(EconometricEstimator, ABC):
    """Abstract base for synthetic control with a nonlinear counterfactual.

    The linear synthetic-control family (``SyntheticControl``,
    ``AugmentedSyntheticControl``, ``RobustSyntheticControl`` ...) -- and Tian's
    ``NonlinearSyntheticControl``, which stays linear-in-weights -- builds the
    counterfactual as a combination of donor outcomes. This base drops that
    restriction: a subclass fits an arbitrary predictor ``f`` on the pre-period
    and the counterfactual is ``f(donor_matrix)`` (see ``KernelSyntheticControl``).
    This can track a treated unit that relates to the donors nonlinearly, at the
    cost of the interpretable weight vector -- ``self.model`` holds the fitted
    predictor callable, not a set of weights.

    A subclass only has to implement ``_fit_model(x, y)``, which trains on a
    pre-period ``(donor_matrix, treated_series)`` and returns a callable mapping a
    donor matrix to its counterfactual. Everything else is inherited from
    ``EconometricEstimator``: the shared lift table (``summarize``), the
    return-on-ad-spend (``_get_roas`` / ``_get_cost_per``), conformal p-values/intervals, the faithful
    jackknife+ leave-one-out loop and the parametric bootstrap. Those inference
    tools are model-agnostic -- they drive the refit through
    ``_fit_predict_weights``, which this base routes to ``_fit_model`` -- so a
    nonlinear counterfactual gets the same distribution-free inference as the
    linear family for free.
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
        conformal_q: float = 1.0,
    ) -> None:
        """Initialize the nonlinear estimator (abstract base for kernel/ML counterfactuals).

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
        conformal_q : float, default=1.0
            The exponent of the moving-block test statistic used for conformal
            inference (p-values and confidence intervals).
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
        self.dates: list[Any] | None = None
        self.actual_pre: Any = None
        self.actual_post: Any = None
        self.prediction_pre: Any = None
        self.prediction_post: Any = None
        self.conformal_q = conformal_q

    @abc.abstractmethod
    def _fit_model(self, x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        """Fit the nonlinear counterfactual on a pre-period subset.

        Parameters
        ----------
        x : numpy array, shape (n_periods, n_donors)
            Donor outcome matrix (one row per time period, one column per donor geo).
        y : numpy array, shape (n_periods,)
            Treated (aggregated) outcome on the same rows.

        Returns
        -------
        A callable mapping a donor matrix of shape ``(m, n_donors)`` to the
        counterfactual of shape ``(m,)``.
        """

    def _fit_predict_weights(self, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray | None:
        """Refit the nonlinear model on a subset and predict.

        Overrides the linear family's weight-fitting hook so the inherited
        jackknife+ (``_block_loo``) and parametric bootstrap (``_bootstrap_refit``)
        drive a full model refit instead of a linear re-solve. The donor-column
        order matches the cached ``_jk_x_*`` matrices.

        Parameters
        ----------
        x_train : numpy array, shape (n_train, n_donors)
            Pre-period donor rows used to refit.
        y_train : numpy array, shape (n_train,)
            Treated pre-period series on the same rows.
        x_eval : numpy array, shape (n_eval, n_donors)
            Donor rows to predict the counterfactual for.

        Returns
        -------
        The counterfactual for each ``x_eval`` row.
        """
        predictor = self._fit_model(np.asarray(x_train, dtype=float), np.asarray(y_train, dtype=float).ravel())
        return np.asarray(predictor(np.asarray(x_eval, dtype=float)), dtype=float)


class MLEstimator(Estimator, abc.ABC):
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
        """Initialize the ML estimator (abstract base class for GeoX).

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

        self.pre_control: nw.DataFrame | None = None
        self.post_control: nw.DataFrame | None = None
        self.pre_test: nw.DataFrame | None = None
        self.post_test: nw.DataFrame | None = None
        self.model: Any = None
        self.test_dates: nw.Series | None = None

    def pre_process(self) -> "MLEstimator":
        self._restrict_to_listed_geos()
        # Derive the treatment indicator from explicit geo lists, as
        # ``EconometricEstimator.pre_process`` does.
        if self.treatment_variable is not None:
            if self.test_geos is not None:
                self.data: nw.DataFrame = self.data.with_columns(
                    nw.col(self.geo_variable).is_in(self.test_geos).cast(nw.Int64).alias(self.treatment_variable)
                )
            elif self.control_geos is not None:
                self.data: nw.DataFrame = self.data.with_columns(
                    (1 - nw.col(self.geo_variable).is_in(self.control_geos).cast(nw.Int64)).alias(
                        self.treatment_variable
                    )
                )
        if self.treatment_variable is not None:
            # Compare dates through an ISO ``YYYY-MM-DD`` key (see ``date_key``);
            # dates between the two periods fall in neither filter.
            key = date_key(self.data, self.date_variable)
            in_pre = key <= period_key(self.pre_period, self.data, self.date_variable)
            in_post = key >= period_key(self.post_period, self.data, self.date_variable)
            self.pre_control, self.pre_test, _ = self._aligned_daily_sums(in_pre)
            self.post_control, self.post_test, self.test_dates = self._aligned_daily_sums(in_post)

        return self

    def _aligned_daily_sums(self, in_period: nw.Expr) -> tuple[nw.DataFrame, nw.DataFrame, nw.Series]:
        """Sum the control and test geos per date, keeping only dates both groups have.

        The date column is dropped from the returned frames and the series are
        later used positionally, so they are joined on date first: otherwise a
        date missing from one group would shift the two series against each other.

        Parameters
        ----------
        in_period : narwhals expression
            Boolean filter selecting the period's rows.

        Returns
        -------
        A tuple of (control sums, test sums, dates), each sorted by date.
        """
        y = self.y_variable

        def daily_sums(flag: int) -> nw.DataFrame:
            rows = self.data.filter(in_period & (nw.col(self.treatment_variable) == flag))
            return rows.group_by(self.date_variable).agg(nw.col(y).sum())

        control, test = daily_sums(0), daily_sums(1)
        joined = control.join(test, on=self.date_variable, how="inner", suffix="_test").sort(self.date_variable)
        if joined.shape[0] < max(control.shape[0], test.shape[0]):
            warnings.warn(
                f"{max(control.shape[0], test.shape[0]) - joined.shape[0]} date(s) are missing from the test or "
                "control geos and were dropped so the two series stay aligned",
                RuntimeWarning,
                stacklevel=3,
            )
        return joined.select(y), joined.select(nw.col(f"{y}_test").alias(y)), joined[self.date_variable]
