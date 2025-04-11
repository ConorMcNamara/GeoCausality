import abc
from abc import ABC
from typing import Union, Optional

import numpy as np
import pandas as pd
import polars as pl


class Estimator(abc.ABC):

    def __init__(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        geo_variable: str = "geo",
        test_geos: Optional[list[str]] = None,
        control_geos: Optional[list[str]] = None,
        treatment_variable: Optional[str] = "is_treatment",
        date_variable: str = "date",
        pre_period: str = "2021-01-01",
        post_period: str = "2021-01-02",
        y_variable: str = "y",
        alpha: float = 0.1,
        msrp: float = 0.0,
        spend: float = 0.0,
    ) -> None:
        """An abstract class for initializing our different geo-causality methods

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
        self.data = data.copy()
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
        self.results = None

    @abc.abstractmethod
    def pre_process(self) -> "Estimator":
        """Method used to pre-process our data to make it usable for our estimator

        Returns
        -------
        Itself, so it can be chained with generate()
        """
        pass

    @abc.abstractmethod
    def generate(self) -> "Estimator":
        """Method used to take our pre-processed data and run our models to estimate causality

        Returns
        -------
        Itself, so it can be chained with summarize()
        """
        pass

    @abc.abstractmethod
    def summarize(self, lift: str) -> None:
        """Method used to summarize the results of generated models on our pre-processed data

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
        percent = int((1 - self.alpha) * 100)
        return f"{percent}%"

    @abc.abstractmethod
    def _get_roas(self) -> tuple:
        """Returns our Return on Ad Spend (ROAS) and CIs

        Returns
        -------
        A tuple containing our ROAS as well as Confidence Intervals
        """
        pass


class EconometricEstimator(Estimator, ABC):

    def __init__(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        geo_variable: str = "geo",
        test_geos: Optional[list[str]] = None,
        control_geos: Optional[list[str]] = None,
        treatment_variable: Optional[str] = "is_treatment",
        date_variable: str = "date",
        pre_period: str = "2021-01-01",
        post_period: str = "2021-01-02",
        y_variable: str = "y",
        alpha: float = 0.1,
        msrp: float = 0.0,
        spend: float = 0.0,
    ) -> None:
        """An abstract class used for FixedEffects as well as Diff-in-Diff

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
        self.model = None

    def pre_process(self) -> "EconometricEstimator":
        if self.test_geos is not None:
            self.data[self.treatment_variable] = int(
                self.data[self.geo_variable].isin(self.test_geos)
            )
        elif self.control_geos is not None:
            self.data[self.treatment_variable] = 1 - int(
                self.data[self.geo_variable].isin(self.control_geos)
            )
        else:
            pass
        self.data["treatment_period"] = np.where(
            self.data[self.date_variable] <= self.pre_period,
            0,
            np.where(self.data[self.date_variable] >= self.post_period, 1, 0),
        )

        return self


class MLEstimator(Estimator, abc.ABC):

    def __init__(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        geo_variable: str = "geo",
        test_geos: Optional[list[str]] = None,
        control_geos: Optional[list[str]] = None,
        treatment_variable: Optional[str] = "is_treatment",
        date_variable: str = "date",
        pre_period: str = "2021-01-01",
        post_period: str = "2021-01-02",
        y_variable: str = "y",
        alpha: float = 0.1,
        msrp: float = 0.0,
        spend: float = 0.0,
    ) -> None:
        """An abstract class used for GeoX as well as Synthetic Control

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

        self.pre_control, self.post_control, self.pre_test, self.post_test = (
            None,
            None,
            None,
            None,
        )
        self.model = None
        self.test_dates = None

    def pre_process(self) -> "MLEstimator":
        pre_control = self.data[
            (self.data[self.date_variable] <= self.pre_period)
            & (self.data[self.treatment_variable] == 0)
        ]
        self.pre_control = (
            pre_control.groupby([self.date_variable])[self.y_variable]
            .sum()
            .reset_index()
        )
        self.pre_control.drop([self.date_variable], axis=1, inplace=True)
        pre_test = self.data[
            (self.data[self.date_variable] <= self.pre_period)
            & (self.data[self.treatment_variable] == 1)
        ]
        self.pre_test = (
            pre_test.groupby([self.date_variable])[self.y_variable].sum().reset_index()
        )
        self.pre_test.drop([self.date_variable], axis=1, inplace=True)
        post_control = self.data[
            (self.data[self.date_variable] >= self.post_period)
            & (self.data[self.treatment_variable] == 0)
        ]
        self.post_control = (
            post_control.groupby([self.date_variable])[self.y_variable]
            .sum()
            .reset_index()
        )
        self.test_dates = self.post_control[self.date_variable]
        self.post_control.drop([self.date_variable], axis=1, inplace=True)
        post_test = self.data[
            (self.data[self.date_variable] >= self.post_period)
            & (self.data[self.treatment_variable] == 1)
        ]
        self.post_test = (
            post_test.groupby([self.date_variable])[self.y_variable].sum().reset_index()
        )
        self.post_test.drop([self.date_variable], axis=1, inplace=True)
        return self
