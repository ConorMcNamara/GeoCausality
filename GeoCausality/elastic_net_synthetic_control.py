"""Elastic-net synthetic control (Doudchenko-Imbens synthesis / intercept-shifted SC)."""

import warnings
from typing import Any

import narwhals as nw
import numpy as np
import polars as pl
from narwhals.typing import IntoDataFrame
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression

from GeoCausality._base import EconometricEstimator


class ElasticNetSyntheticControl(EconometricEstimator):
    """Run elastic-net synthetic control (the Doudchenko-Imbens synthesis) for our geo-test.

    Doudchenko & Imbens (2016) unify synthetic control, difference-in-differences
    and constrained regression by relaxing three restrictions of the classic
    Abadie estimator: they add an **intercept** (a level shift, so the treated
    unit can be tracked even when it sits systematically above or below any
    weighted average of donors), they drop the **sum-to-one** and **non-negativity**
    constraints on the donor weights, and they add **elastic-net** regularisation.
    The counterfactual is ``mu + donor_matrix @ w``, with ``(mu, w)`` fit on the
    pre-period by penalised least squares::

        min_{mu, w}  sum_t (y_t - mu - sum_j w_j x_{jt})^2
                     + lambda * [ (1 - l1_ratio)/2 * ||w||_2^2 + l1_ratio * ||w||_1 ]

    This MVP solves the unconstrained regime (the Doudchenko-Imbens default) with
    scikit-learn's elastic net: the intercept is toggled by ``intercept``,
    non-negativity by ``non_negative``, and ``lambda_`` is chosen by
    cross-validation when not supplied. The sum-to-one constrained regime is not
    yet implemented (it needs a constrained solver).

    Special cases: ``intercept=False, non_negative=True, sum_to_one=True,
    lambda_=0`` is classic synthetic control; ``intercept=True, l1_ratio=0`` is a
    ridge-augmented fit close to ``AugmentedSyntheticControl``; ``lambda_=0`` with
    no constraints is matched-market OLS.

    Inference reuses the shared conformal p-values / confidence intervals, the
    faithful jackknife+ and the parametric bootstrap, all driven through the donor
    weights via ``_fit_predict_weights``.
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
        intercept: bool = True,
        l1_ratio: float = 0.5,
        lambda_: float | None = None,
        non_negative: bool = False,
        sum_to_one: bool = False,
        conformal_q: float = 1.0,
    ) -> None:
        """Initialize the elastic-net synthetic control estimator (Doudchenko-Imbens synthesis).

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
            The alpha level for our experiment (the confidence-interval significance
            level; distinct from ``l1_ratio``, the elastic-net mixing parameter).
        msrp : float, default=0.0
            The average MSRP of our sale. Used to calculate incremental revenue.
        spend : float, default=0.0
            The amount we spent on our treatment. Used to calculate ROAS (return on ad spend)
             or cost-per-acquisition.
        intercept : bool, default=True
            Whether to fit the level-shift intercept ``mu``. The distinguishing
            feature of Doudchenko-Imbens over classic synthetic control.
        l1_ratio : float, default=0.5
            The elastic-net mixing parameter in ``[0, 1]``: ``0`` is pure ridge,
            ``1`` is pure lasso. (Named after scikit-learn's parameter to avoid the
            collision with ``alpha``, the significance level.) Values at or near
            ``0`` lean on scikit-learn's coordinate-descent solver, which is
            inefficient for pure L2; for a dedicated closed-form ridge fit prefer
            ``AugmentedSyntheticControl``.
        lambda_ : float, optional
            The elastic-net penalty strength. When ``None`` (default) it is chosen
            by cross-validation. ``0`` fits an unpenalised (OLS) regression.
        non_negative : bool, default=False
            Whether to constrain the donor weights to be non-negative.
        sum_to_one : bool, default=False
            Whether to constrain the donor weights to sum to one. Not implemented
            in this MVP (raises ``NotImplementedError``); the unconstrained regime
            is the Doudchenko-Imbens default.
        conformal_q : float, default=1.0
            The exponent of the moving-block test statistic used for conformal
            inference (p-values and confidence intervals).

        Notes
        -----
        Based on Doudchenko & Imbens :cite:`doudchenko2016`.
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
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio}")
        if lambda_ is not None and lambda_ < 0.0:
            raise ValueError(f"lambda_ must be non-negative, got {lambda_}")
        if sum_to_one:
            raise NotImplementedError(
                "The sum-to-one constrained regime is not implemented in this MVP; "
                "use sum_to_one=False (the Doudchenko-Imbens default)."
            )
        self.intercept = intercept
        self.l1_ratio = l1_ratio
        self.lambda_ = lambda_
        self.non_negative = non_negative
        self.sum_to_one = sum_to_one
        self.conformal_q = conformal_q
        self.intercept_: float = 0.0
        self.dates: list[Any] | None = None
        self.prediction_pre: nw.DataFrame | None = None
        self.prediction_post: nw.DataFrame | None = None
        self.actual_pre: nw.DataFrame | None = None
        self.actual_post: nw.DataFrame | None = None

    def pre_process(self) -> "ElasticNetSyntheticControl":
        """Tag the pre/post periods and record the full date axis.

        The donor matrices are assembled in ``generate`` (fit and prediction share
        the same sorted pivot, so the weights and the predicted rows always line up
        by donor).

        Returns
        -------
        ElasticNetSyntheticControl
            Itself, so it can be chained with generate().
        """
        super().pre_process()
        self.dates = sorted(self.data[self.date_variable].unique().to_list())
        return self

    def generate(self) -> "ElasticNetSyntheticControl":
        """Fit the elastic-net weights and intercept, then compute lift and inference.

        Returns
        -------
        ElasticNetSyntheticControl
            Itself, so it can be chained with summarize().
        """
        if self.treatment_variable is None:
            raise ValueError("treatment_variable must not be None")
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
        control_pre = (
            self.data.filter((nw.col(self.treatment_variable) == 0) & (nw.col("treatment_period") == 0))
            .select([self.y_variable, self.date_variable, self.geo_variable])
            .group_by([self.date_variable, self.geo_variable])
            .agg(nw.col(self.y_variable).sum())
            .sort([self.date_variable, self.geo_variable])
        )
        control_post = (
            self.data.filter((nw.col(self.treatment_variable) == 0) & (nw.col("treatment_period") == 1))
            .select([self.y_variable, self.date_variable, self.geo_variable])
            .group_by([self.date_variable, self.geo_variable])
            .agg(nw.col(self.y_variable).sum())
            .sort([self.date_variable, self.geo_variable])
        )
        control_pre_pivot = nw.from_native(
            control_pre.to_native().pivot(on=self.geo_variable, index=self.date_variable, values=self.y_variable),
            eager_only=True,
        )
        control_post_pivot = nw.from_native(
            control_post.to_native().pivot(on=self.geo_variable, index=self.date_variable, values=self.y_variable),
            eager_only=True,
        )
        control_pre_mat = control_pre_pivot.drop(self.date_variable).to_numpy().astype(float)
        control_post_mat = control_post_pivot.drop(self.date_variable).to_numpy().astype(float)
        treated_pre = self.actual_pre[self.y_variable].to_numpy().astype(float)

        # Cache the donor matrices for the shared faithful jackknife+ / bootstrap.
        self._jk_x_pre = control_pre_mat
        self._jk_x_post = control_post_mat
        self._jk_y_pre = treated_pre

        # Fit and predict on the same sorted donor matrix so weights and columns align.
        self.model = self._create_model(control_pre_mat, treated_pre)
        prediction_pre_arr = control_pre_mat @ self.model + self.intercept_
        prediction_post_arr = control_post_mat @ self.model + self.intercept_

        self.prediction_pre = nw.from_native(
            pl.DataFrame(
                {
                    self.date_variable: control_pre_pivot[self.date_variable].to_native(),
                    self.y_variable: prediction_pre_arr.flatten(),
                }
            ),
            eager_only=True,
        )
        self.prediction_post = nw.from_native(
            pl.DataFrame(
                {
                    self.date_variable: control_post_pivot[self.date_variable].to_native(),
                    self.y_variable: prediction_post_arr.flatten(),
                }
            ),
            eager_only=True,
        )
        self.results = {
            "test": self.actual_post[self.y_variable].to_numpy(),
            "counterfactual": self.prediction_post[self.y_variable].to_numpy(),
            "lift": self.actual_post[self.y_variable].to_numpy() - self.prediction_post[self.y_variable].to_numpy(),
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

    def _create_model(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit the elastic-net donor weights and intercept on the pre-period.

        When ``lambda_`` is ``None`` the penalty is selected by cross-validation
        (and cached, so the jackknife+ folds reuse the same penalty rather than
        re-selecting per fold). The fitted intercept is stored on ``self.intercept_``.

        Parameters
        ----------
        x : numpy array, shape (n_pre, n_donors)
            Pre-period donor matrix.
        y : numpy array, shape (n_pre,)
            Treated pre-period series.

        Returns
        -------
        The donor weight vector, shape (n_donors,).
        """
        if self.lambda_ is None:
            cv = max(2, min(5, x.shape[0]))
            est = self._silent_fit(
                ElasticNetCV(
                    l1_ratio=self.l1_ratio,
                    fit_intercept=self.intercept,
                    positive=self.non_negative,
                    cv=cv,
                    alphas=self._alpha_grid(x, y),
                    max_iter=100_000,
                    tol=1e-3,
                ),
                x,
                y,
            )
            self.lambda_ = float(est.alpha_)
        else:
            est = self._silent_fit(self._make_estimator(self.lambda_), x, y)
        self.intercept_ = float(est.intercept_) if self.intercept else 0.0
        return np.asarray(est.coef_, dtype=float)

    @staticmethod
    def _silent_fit(estimator: Any, x: np.ndarray, y: np.ndarray) -> Any:
        """Fit a scikit-learn estimator, suppressing its convergence advisories.

        We deliberately use coordinate descent for the whole elastic-net family
        (including near-OLS penalties and pure ridge), so the ``ConvergenceWarning``
        and the "coordinate descent without L1 regularization" advisory it emits on
        those regimes are expected noise, not a caller-actionable problem -- silence
        them rather than spam every fit and cross-validation fold.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.filterwarnings(
                "ignore", message="Coordinate descent without L1 regularization", category=UserWarning
            )
            return estimator.fit(x, y)

    def _alpha_grid(self, x: np.ndarray, y: np.ndarray, n_alphas: int = 100, ratio: float = 1e-3) -> np.ndarray:
        """Data-driven geometric penalty grid for the cross-validation.

        Supplied explicitly because scikit-learn's automatic grid divides by
        ``l1_ratio`` and so cannot generate one for pure ridge (``l1_ratio=0``).
        ``alpha_max`` is the lasso-style penalty that zeroes the weights, with an
        ``l1_ratio`` floor so the ridge case stays finite; the grid spans down to
        ``alpha_max * ratio``.

        Parameters
        ----------
        x : numpy array, shape (n_pre, n_donors)
            Pre-period donor matrix.
        y : numpy array, shape (n_pre,)
            Treated pre-period series.
        n_alphas : int, default=100
            Number of penalties on the path.
        ratio : float, default=1e-3
            Smallest penalty as a fraction of ``alpha_max``.

        Returns
        -------
        The penalty grid, descending.
        """
        n = x.shape[0]
        xc = x - x.mean(axis=0) if self.intercept else x
        yc = y - y.mean() if self.intercept else y
        alpha_max = float(np.max(np.abs(xc.T @ yc))) / (n * max(self.l1_ratio, 1e-3))
        if alpha_max <= 0.0:
            alpha_max = 1.0
        return np.geomspace(alpha_max, alpha_max * ratio, n_alphas)

    def _make_estimator(self, lambda_: float) -> Any:
        """Build the (non-CV) fitter for a fixed penalty.

        A zero penalty is fit with ``LinearRegression`` (scikit-learn's elastic net
        does not accept ``alpha=0``); any positive penalty uses ``ElasticNet``.

        Parameters
        ----------
        lambda_ : float
            The elastic-net penalty strength.

        Returns
        -------
        An unfitted scikit-learn estimator honouring ``intercept`` / ``non_negative``.
        """
        if lambda_ == 0.0:
            return LinearRegression(fit_intercept=self.intercept, positive=self.non_negative)
        return ElasticNet(
            alpha=lambda_,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.intercept,
            positive=self.non_negative,
            max_iter=100_000,
            tol=1e-3,
        )

    def _fit_predict_weights(self, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray | None:
        """Refit the elastic-net weights on a subset and predict.

        Holds ``self.lambda_`` at its full-fit value (set during ``_create_model``)
        so the leave-one-out / bootstrap refits reflect the same penalty. Drives
        the shared faithful jackknife+ and parametric bootstrap.

        Parameters
        ----------
        x_train : numpy array, shape (n_train, n_donors)
            Pre-period donor rows used to refit.
        y_train : numpy array, shape (n_train,)
            Treated pre-period series on the same rows.
        x_eval : numpy array, shape (n_eval, n_donors)
            Donor rows to predict.

        Returns
        -------
        The counterfactual for each ``x_eval`` row.
        """
        if self.lambda_ is None:
            raise ValueError("lambda_ must be set before refitting (call generate first)")
        est = self._silent_fit(self._make_estimator(self.lambda_), x_train, y_train)
        intercept = float(est.intercept_) if self.intercept else 0.0
        return x_eval @ np.asarray(est.coef_, dtype=float) + intercept
