from math import ceil
from typing import Union, Optional

import numpy as np
import pandas as pd
import polars as pl
from linearmodels.panel import PanelOLS
from tabulate import tabulate

from GeoCausality._base import EconometricEstimator


class FixedEffects(EconometricEstimator):

    def __init__(
        self,
        data: Union[pl.DataFrame, pd.DataFrame],
        geo_variable: str,
        test_geos: Optional[list[str]] = None,
        control_geos: Optional[list[str]] = None,
        treatment_variable: Optional[str] = None,
        date_variable: str = None,
        pre_period: str = None,
        post_period: str = None,
        y_variable: str = None,
        alpha: float = 0.1,
        msrp: float = 0.0,
        spend: float = 0.0,
    ) -> None:
        """A class to run our FixedEffects Model for geo-tests.

        Parameters
        ----------
        data: pandas or polars data frame
            Our geo-based time-series data
        geo_variable: str
            The name of the variable representing our geo-data
        test_geos: list, optional
            The geos that were assigned treatment. If not provided, rely on treatment variable
        control_geos: list, optional
            The geos that were withheld from treatment. If not provided, rely on treatment variable
        treatment_variable: str, optional
            If test and control geos are not provided, the column denoting which is test and control. Assumes that
            1 is coded as "treatment" and 0 is coded as "control"
        date_variable: str
            The name of the variable representing our dates
        pre_period: str
            The time period used to train our models. Starts from the first date in our data to pre_period.
        post_period: str
            The time period used to evaluate our performance. Starts from post_period to the last date in our data
        y_variable: str
            The name of the variable representing the results of our data
        alpha: float, default=0.1
            The alpha level for our experiment
        msrp: float, default=0.0
            The average MSRP of our sale. Used to calculate incremental revenue.
        spend: float, default=0.0
            The amount we spent on our treatment. Used to calculate ROAS (return on ad spend)
             or cost-per-acquisition.

        Notes
        -----
        Based on https://matheusfacure.github.io/python-causality-handbook/14-Panel-Data-and-Fixed-Effects.html
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
        self.n_dates = None
        self.n_geos = None

    def pre_process(self) -> "FixedEffects":
        super().pre_process()
        self.data["campaign_treatment"] = (
            self.data["treatment_period"] * self.data[self.treatment_variable]
        )
        self.n_dates = int(
            self.data.loc[
                self.data["campaign_treatment"] == 1, "campaign_treatment"
            ].sum()
            / len(
                self.data.loc[
                    self.data["campaign_treatment"] == 1, f"{self.geo_variable}"
                ].unique()
            )
        )
        self.n_geos = len(
            self.data.loc[
                self.data["campaign_treatment"] == 1, f"{self.geo_variable}"
            ].unique()
        )
        return self

    def generate(self) -> "FixedEffects":
        model = PanelOLS.from_formula(
            f"{self.y_variable} ~ campaign_treatment + EntityEffects + TimeEffects",
            data=self.data.set_index([self.geo_variable, self.date_variable]),
        )
        self.model = model.fit(
            cov_type="clustered", cluster_entity=True, cluster_time=True
        )
        cis = self.model.conf_int(1 - self.alpha)
        self.results = {
            "test": float(self.model.params.iloc[0]),
            "control": 0.0,
            "lift": float(self.model.params.iloc[0]),
            "lift_ci_lower": float(cis["lower"].iloc[0]),
            "lift_ci_upper": float(cis["upper"].iloc[0]),
            "incrementality": float(
                self.model.params.iloc[0] * self.n_dates * self.n_geos
            ),
            "incrementality_ci_lower": float(
                cis["lower"].iloc[0] * self.n_dates * self.n_geos
            ),
            "incrementality_ci_upper": float(
                cis["upper"].iloc[0] * self.n_dates * self.n_geos
            ),
            "p_value": float(self.model.pvalues.iloc[0]),
        }
        return self

    def summarize(self, lift) -> None:
        lift = lift.casefold()
        if lift not in [
            "absolute",
            "incremental",
            "cost-per",
            "revenue",
            "roas",
        ]:
            raise ValueError(
                f"Cannot measure {lift}. Choose one of `absolute`,  `incremental`, `cost-per`, `revenue` "
                f"or `roas`"
            )
        table_dict = {}
        ci_alpha = self._get_ci_print()
        if lift == "incremental":
            table_dict["Metric"] = [self.y_variable]
            table_dict["Lift Type "] = ["Incremental"]
            table_dict["Lift"] = [f"""{ceil(self.results["incrementality"]):,}"""]
            table_dict[f"{ci_alpha} Lower CI"] = [
                f"""{ceil(self.results["incrementality_ci_lower"]):,}"""
            ]
            table_dict[f"{ci_alpha} Upper CI"] = [
                f"""{ceil(self.results["incrementality_ci_upper"]):,}"""
            ]
        elif lift == "absolute":
            table_dict["Metric"] = [self.y_variable]
            table_dict["Lift Type "] = ["Absolute"]
            table_dict["Lift"] = [f"""{ceil(self.results["lift"]):,}"""]
            table_dict[f"{ci_alpha} Lower CI"] = [
                f"""{ceil(self.results["lift_ci_lower"]):,}"""
            ]
            table_dict[f"{ci_alpha} Upper CI"] = [
                f"""{ceil(self.results["lift_ci_upper"]):,}"""
            ]
        elif lift == "revenue":
            table_dict["Metric"] = ["Revenue"]
            table_dict["Lift Type "] = ["Incremental"]
            table_dict["Lift"] = [
                f"""${round(self.results["incrementality"] * self.msrp, 2):,}"""
            ]
            table_dict[f"{ci_alpha} Lower CI"] = [
                f"""${round(self.results["incrementality_ci_lower"] * self.msrp, 2):,}"""
            ]
            table_dict[f"{ci_alpha} Upper CI"] = [
                f"""${round(self.results["incrementality_ci_upper"] * self.msrp, 2):,}"""
            ]
        else:
            table_dict["Metric"] = ["ROAS"]
            table_dict["Lift Type "] = ["Incremental"]
            roas_lift, roas_ci_lower, roas_ci_upper = self._get_roas()
            table_dict["Lift"] = [f"${round(roas_lift, 2)}"]
            table_dict[f"{ci_alpha} Lower CI"] = [f"${round(roas_ci_lower, 2)}"]
            table_dict[f"{ci_alpha} Upper CI"] = [f"${round(roas_ci_upper, 2)}"]
        table_dict["p_value"] = [self.results["p_value"]]
        print(tabulate(table_dict, headers="keys", tablefmt="grid"))

    def _get_roas(self) -> tuple:
        lift = ceil(self.results["incrementality"])
        roas_lift = self.spend / lift if lift > 0 else np.inf
        ci_upper = ceil(self.results["incrementality_ci_upper"])
        roas_ci_lower = self.spend / ci_upper if ci_upper > 0 else np.inf
        ci_lower = ceil(self.results["incrementality_ci_lower"])
        roas_ci_upper = self.spend / ci_lower if ci_lower > 0 else np.inf
        return roas_lift, roas_ci_lower, roas_ci_upper
