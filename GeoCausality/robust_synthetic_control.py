"""Robust Synthetic Control method for geo-experiment causal inference."""

from typing import Any

import narwhals as nw
import numpy as np
import polars as pl
from narwhals.typing import IntoDataFrame

from GeoCausality._base import EconometricEstimator


class RobustSyntheticControl(EconometricEstimator):
    """Run robust synthetic control for our geo-test."""

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
        threshold: float | None = None,
        sv_count: int | None = None,
        sv_energy: float = 0.999,
        conformal_q: float = 1.0,
    ) -> None:
        """Initialize the robust synthetic control estimator.

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
        threshold : float, optional
            Remove singular values that are less than this threshold. Overrides
            ``sv_count`` and ``sv_energy`` when set.
        sv_count : int, optional
            Keep this many of the largest singular values when reducing the
            outcome matrix. Overrides ``sv_energy`` when set.
        sv_energy : float, default=0.999
            Used when neither ``threshold`` nor ``sv_count`` is given: keep the
            fewest leading singular values that retain this fraction of the donor
            matrix's spectral (squared-singular-value) energy. This makes the
            estimator usable without manually choosing a rank.
        conformal_q : float, default=1.0
            The exponent of the moving-block test statistic used for conformal
            inference (p-values and confidence intervals).

        Notes
        -----
        Based on Amjad, Shah & Shen :cite:`robust2018` and https://github.com/sdfordham/pysyncon/blob/main/pysyncon/robust.py
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
        if not 0.0 < sv_energy <= 1.0:
            raise ValueError(f"sv_energy must be in (0, 1], got {sv_energy}")
        # ``threshold`` and ``sv_count`` are explicit rank overrides; when neither is
        # given we fall back to retaining ``sv_energy`` of the spectral energy, so the
        # estimator runs without manual rank selection.
        self.threshold = threshold
        self.sv_count = sv_count
        self.sv_energy = sv_energy
        self.conformal_q = conformal_q
        self.daily_x: nw.DataFrame | None = None
        self.daily_y: nw.DataFrame | None = None

    def pre_process(self) -> "RobustSyntheticControl":
        """Pivot the control geos into a daily outcome matrix and aggregate the test series.

        Returns
        -------
        RobustSyntheticControl
            Itself, so it can be chained with generate().
        """
        super().pre_process()
        self.dates = sorted(self.data[self.date_variable].unique().to_list())
        if self.treatment_variable is None:
            raise ValueError("treatment_variable must not be None")
        day_x = self.data.filter(nw.col(self.treatment_variable) == 0).select(
            [self.y_variable, self.geo_variable, self.date_variable]
        )
        # Pivot: rows=dates, cols=geos. Sort by date so the pre-period rows are the
        # leading block (the split in ``_create_model`` relies on this ordering).
        self.daily_x = nw.from_native(
            day_x.to_native().pivot(on=self.geo_variable, index=self.date_variable, values=self.y_variable),
            eager_only=True,
        ).sort(self.date_variable)
        daily_y = (
            self.data.filter(nw.col(self.treatment_variable) == 1)
            .select([self.y_variable, self.date_variable])
            .group_by(self.date_variable)
            .agg(nw.col(self.y_variable).sum())
            .sort(self.date_variable)
        )
        self.daily_y = daily_y
        return self

    def generate(self) -> "RobustSyntheticControl":
        """Build the counterfactual from the de-noised weights and compute lift and inference.

        Returns
        -------
        RobustSyntheticControl
            Itself, so it can be chained with summarize().
        """
        self.model = self._create_model()
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
        # Pivot: rows=dates, cols=geos
        control_pre_pivot = nw.from_native(
            control_pre.to_native().pivot(on=self.geo_variable, index=self.date_variable, values=self.y_variable),
            eager_only=True,
        )
        control_post_pivot = nw.from_native(
            control_post.to_native().pivot(on=self.geo_variable, index=self.date_variable, values=self.y_variable),
            eager_only=True,
        )
        control_pre_mat = control_pre_pivot.drop(self.date_variable).to_numpy()
        control_post_mat = control_post_pivot.drop(self.date_variable).to_numpy()
        # Cache the donor matrices for the shared faithful jackknife+ loop.
        self._jk_x_pre = control_pre_mat
        self._jk_x_post = control_post_mat
        self._jk_y_pre = self.actual_pre[self.y_variable].to_numpy()

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
        self.results = self._finalize_counterfactual_results(
            self.actual_pre,
            self.prediction_pre,
            self.actual_post,
            self.prediction_post,
            q=self.conformal_q,
        )
        return self

    def _create_model(self) -> np.ndarray:
        """Generate the weights used to predict our counterfactual.

        Returns
        -------
        The weights matrix used to create our model
        """
        if self.daily_x is None:
            raise ValueError("daily_x must not be None")
        if self.daily_y is None:
            raise ValueError("daily_y must not be None")
        # daily_x pivot has date column first; drop it to get geo columns only
        daily_x_mat = self.daily_x.drop(self.date_variable).to_numpy()
        daily_x_transposed = daily_x_mat.T
        M_hat = self._svd(daily_x_transposed).T
        # Number of pre-period rows: compare dates as ISO strings so the split is
        # backend- and format-agnostic (matches the base class; integer years like
        # "1989" are not valid ``date.fromisoformat`` input). ``daily_x`` is sorted
        # by date, so the pre-period is the leading block.
        date_list = self.daily_x[self.date_variable].to_list()
        end_idx = sum(1 for d in date_list if str(d) <= self.pre_period)
        M_hat_neg = M_hat[:end_idx, :]
        Y1_neg = self.daily_y[self.y_variable].to_numpy()[:end_idx]

        W = np.matmul(
            np.linalg.inv(M_hat_neg.T @ M_hat_neg + self.lambda_ * np.identity(M_hat_neg.shape[1])),
            M_hat_neg.T @ Y1_neg,
        )
        return W

    def _fit_predict_weights(self, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray | None:
        """Refit the SVD-denoised ridge weights on a subset and predict.

        Denoises the training donor matrix via the same truncated SVD as the full
        fit, solves the ridge weights on the denoised pre-period, and predicts the
        raw evaluation rows (matching ``generate``'s fit-denoised / predict-raw
        scheme).

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
        denoised = self._svd(x_train.T).T
        n_c = denoised.shape[1]
        weights = np.linalg.solve(denoised.T @ denoised + self.lambda_ * np.identity(n_c), denoised.T @ y_train)
        return x_eval @ weights

    def _svd(self, groupby_x_transposed: np.ndarray) -> np.ndarray:
        """Perform singular value decomposition of our groupby_x_transposed matrix.

        Parameters
        ----------
        groupby_x_transposed : numpy array
            The transpose of our groupby_data. Formatted such that for each geo, we list the average
            y_metric specified in our class initiation

        Returns
        -------
        M_hat, a matrix based on our SVD.
        """
        u, s, v = np.linalg.svd(groupby_x_transposed)
        s_shape = s.shape[0] - 1
        if self.threshold:
            idx = 0
            while s[idx] > self.threshold and idx < s_shape:
                idx += 1
        elif self.sv_count is not None:
            idx = self.sv_count
        else:
            # Default: keep the fewest leading singular values retaining
            # ``sv_energy`` of the squared-singular-value (spectral) energy.
            energy = np.cumsum(s**2) / np.sum(s**2)
            idx = int(np.searchsorted(energy, self.sv_energy) + 1)
        idx = min(idx, s.shape[0])
        s_res = np.zeros_like(groupby_x_transposed)
        s_res[:idx, :idx] = np.diag(s[:idx])
        r, c = groupby_x_transposed.shape
        p_hat = max(np.count_nonzero(groupby_x_transposed) / (r * c), 1 / (r * c))
        M_hat = (1 / p_hat) * (u @ s_res @ v)
        return M_hat
