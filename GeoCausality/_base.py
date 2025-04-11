import pandas as pd

import polars as pl

import abc

from typing import Union


class Estimator(abc.ABC):

    def __init__(
            self,
    data: Union[pl.DataFrame, pd.DataFrame],
    test_geos: list[str],
    control_geos: list[str],
    pre_period: str,
    post_period: str,
    ) -> None:
        """

        Parameters
        ----------
        data: pandas or polars data frame
            Our geo-based time-series data
        test_geos: list
            The geos that were assigned treatment
        control_geos: list
            The geos that were withheld from treatment
        pre_period: str
            The time period used to train our models. Starts from the first date in our data to pre_period.
        post_period: str
            The time period used to evaluate our performance. Starts from post_period to the last date in our data
        """
        self.data = data
        self.test_geos = test_geos
        self.control_geos = control_geos
        self.pre_period = pre_period
        self.post_period = post_period

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def generate(self):
        pass

    def summarize(self):
        pass