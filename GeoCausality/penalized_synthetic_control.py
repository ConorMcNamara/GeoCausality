"""Penalized Synthetic Control method for geo-experiment causal inference."""

from typing import Any

import narwhals as nw
import numpy as np
import polars as pl
from narwhals.typing import IntoDataFrame
from scipy.optimize import Bounds, LinearConstraint, minimize

from GeoCausality._base import EconometricEstimator


class PenalizedSyntheticControl(EconometricEstimator):
    """Run penalized synthetic control for our geo-test."""

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
        lambda_: float = 0.1,
        conformal_q: float = 1.0,
    ) -> None:
        """Initialize the penalized synthetic control estimator.

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
        lambda_ : float, default=0.1
            Ridge parameter to use
        conformal_q : float, default=1.0
            The exponent of the moving-block test statistic used for conformal
            inference (p-values and confidence intervals).

        Notes
        -----
        Based on Abadie & L'Hour :cite:`penalized2021`
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
        self.prediction_pre: nw.DataFrame | None = None
        self.prediction_post: nw.DataFrame | None = None
        self.actual_pre: nw.DataFrame | None = None
        self.actual_post: nw.DataFrame | None = None
        self.lambda_ = lambda_
        self.conformal_q = conformal_q

    def pre_process(self) -> "PenalizedSyntheticControl":
        """Mark the treatment periods and record the sorted date axis.

        Returns
        -------
        PenalizedSyntheticControl
            Itself, so it can be chained with generate().
        """
        super().pre_process()
        self.dates = sorted(self.data[self.date_variable].unique().to_list())
        return self

    def generate(self) -> "PenalizedSyntheticControl":
        """Build the counterfactual from the weights and compute lift and conformal inference.

        Returns
        -------
        PenalizedSyntheticControl
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
        # Pivot: rows=dates, cols=geos — equivalent to the transposed pandas pivot
        control_pre_pivot = nw.from_native(
            control_pre.to_native().pivot(on=self.geo_variable, index=self.date_variable, values=self.y_variable),
            eager_only=True,
        )
        control_post_pivot = nw.from_native(
            control_post.to_native().pivot(on=self.geo_variable, index=self.date_variable, values=self.y_variable),
            eager_only=True,
        )
        # Drop the index column before matrix multiply; keep only geo columns
        control_pre_mat = control_pre_pivot.drop(self.date_variable).to_numpy()
        control_post_mat = control_post_pivot.drop(self.date_variable).to_numpy()
        # Cache the donor matrices for the shared faithful jackknife+ loop.
        self._jk_x_pre = control_pre_mat
        self._jk_x_post = control_post_mat
        self._jk_y_pre = self.actual_pre[self.y_variable].to_numpy()

        # Fit the penalized weights against the full pre-period trajectory (not a
        # single pre-period mean), so the synthetic control tracks the treated
        # unit's path the way the unpenalized estimator does.
        self.model = self._create_model(self.actual_pre[self.y_variable].to_numpy(), control_pre_mat)

        prediction_pre_arr = control_pre_mat @ self.model
        prediction_post_arr = control_post_mat @ self.model

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

    def _fit_predict_weights(self, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray | None:
        """Refit the penalized weights on a pre-period subset and predict.

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
        return x_eval @ self._create_model(y_train, x_train)

    def _create_model(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Fit the penalized synthetic-control weights over the pre-period.

        Minimises the Abadie & L'Hour (2021) penalized objective

            (1 / T) * ||y - X w||^2  +  lambda * sum_j w_j * D_j

        subject to the simplex constraint ``w >= 0, sum(w) = 1``, where ``D_j`` is
        the mean squared per-period distance between the treated series and donor
        ``j``. The aggregate-fit term and the pairwise-discrepancy penalty are both
        on a per-period (mean-of-squares) scale, so ``lambda_`` trades them off in
        comparable units: ``lambda_ -> 0`` recovers the unpenalized synthetic
        control, while larger values concentrate weight on donors close to the
        treated unit (nearest-neighbour matching in the limit).

        Parameters
        ----------
        y : numpy array, shape (n_pre,)
            The treated unit's pre-period series.
        x : numpy array, shape (n_pre, n_donors)
            The donor pre-period matrix (rows = dates, cols = donor geos).

        Returns
        -------
        An array containing the weights to be applied to each control geo.
        """
        _, n_c = x.shape
        discrepancy = np.mean((y.reshape(-1, 1) - x) ** 2, axis=0)
        bounds = Bounds(lb=np.full(n_c, 0.0), ub=np.full(n_c, 1.0))
        constraints = LinearConstraint(A=np.full(n_c, 1.0), lb=1.0, ub=1.0)
        x0 = np.full(n_c, 1 / n_c)
        res = minimize(
            fun=lambda w: self._loss_w(w, x, y, discrepancy),
            x0=x0,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
        )
        return res["x"]

    def _loss_w(self, w: np.ndarray, x: np.ndarray, y: np.ndarray, discrepancy: np.ndarray) -> float:
        """Penalized synthetic-control loss for a candidate weight vector.

        Parameters
        ----------
        w : numpy array, shape (n_donors,)
            Candidate donor weights.
        x : numpy array, shape (n_pre, n_donors)
            Donor pre-period matrix.
        y : numpy array, shape (n_pre,)
            Treated pre-period series.
        discrepancy : numpy array, shape (n_donors,)
            Mean squared per-period distance between the treated series and each
            donor (the pairwise-discrepancy penalty weights).

        Returns
        -------
        The mean-squared aggregate fit plus the lambda-weighted penalty.
        """
        fit = float(np.mean((y - x @ w) ** 2))
        penalty = float(self.lambda_ * np.sum(w * discrepancy))
        return fit + penalty

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
        self._plot_counterfactual(
            self.dates,
            self.actual_pre[self.y_variable].to_numpy(),
            self.actual_post[self.y_variable].to_numpy(),
            self.prediction_pre[self.y_variable].to_numpy(),
            self.prediction_post[self.y_variable].to_numpy(),
        )
