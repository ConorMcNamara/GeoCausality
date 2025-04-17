from math import ceil
from typing import Union, Optional

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf
from tabulate import tabulate  # type: ignore

from GeoCausality._base import EconometricEstimator


class DiffinDiff(EconometricEstimator):
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
        """A class to run difference-in-differences for our geo-test.

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

        Notes
        -----
        Based on https://matheusfacure.github.io/python-causality-handbook/13-Difference-in-Differences.html
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
        self.groupby_data = None
        self.n_dates = None

    def pre_process(self) -> "DiffinDiff":
        super().pre_process()
        self.groupby_data = (
            self.data.groupby(
                [
                    f"{self.treatment_variable}",
                    "treatment_period",
                    f"{self.date_variable}",
                ]
            )
            .agg({f"{self.y_variable}": "sum"})
            .reset_index()
        )
        self.n_dates = len(
            self.groupby_data.loc[
                (self.groupby_data["is_treatment"] == 1) & (self.groupby_data["treatment_period"] == 1)
            ]
        )
        return self

    def generate(self) -> "DiffinDiff":
        self.model = smf.ols(
            f"{self.y_variable} ~ {self.treatment_variable} * treatment_period",
            data=self.groupby_data,
        ).fit()
        cis = self.model.conf_int(alpha=self.alpha, cols=None).iloc[-1]
        self.results = dict(
            test=float(self.model.params["treatment_period"]),
            control=float(self.model.params["Intercept"]),
            lift=self.model.params[f"{self.treatment_variable}:treatment_period"],
            lift_ci_lower=cis[0],
            lift_ci_upper=cis[1],
        )
        self.results["incrementality"] = self.results["lift"] * self.n_dates
        self.results["incrementality_ci_lower"] = self.results["lift_ci_lower"] * self.n_dates
        self.results["incrementality_ci_upper"] = self.results["lift_ci_upper"] * self.n_dates
        self.results["p_value"] = float(self.model.pvalues[f"{self.treatment_variable}:treatment_period"])
        return self

    def summarize(self, lift: str) -> None:
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
        table_dict = {
            "Variant": [self.results["control"] * self.n_dates + self.results["incrementality"]],
            "Baseline": [self.results["control"] * self.n_dates],
        }
        ci_alpha = self._get_ci_print()
        if lift == "incremental":
            table_dict["Metric"] = [self.y_variable]
            table_dict["Lift Type "] = ["Incremental"]
            table_dict["Lift"] = [f"""{ceil(self.results["incrementality"]):,}"""]
            table_dict[f"{ci_alpha} Lower CI"] = [f"""{ceil(self.results["incrementality_ci_lower"]):,}"""]
            table_dict[f"{ci_alpha} Upper CI"] = [f"""{ceil(self.results["incrementality_ci_upper"]):,}"""]
        elif lift == "absolute":
            table_dict["Metric"] = [self.y_variable]
            table_dict["Lift Type "] = ["Absolute"]
            table_dict["Lift"] = [f"""{ceil(self.results["lift"]):,}"""]
            table_dict[f"{ci_alpha} Lower CI"] = [f"""{ceil(self.results["lift_ci_lower"]):,}"""]
            table_dict[f"{ci_alpha} Upper CI"] = [f"""{ceil(self.results["lift_ci_upper"]):,}"""]
        elif lift == "relative":
            table_dict["Metric"] = [self.y_variable]
            table_dict["Lift Type"] = ["Relative"]
            table_dict["Lift"] = [
                f"""{round(self.results["incrementality"] * 100 / (self.results["control"] * self.n_dates), 2)}%"""
            ]
            table_dict[f"{ci_alpha} Lower CI"] = [
                f"""{
                    round(self.results["incrementality_ci_lower"] * 100 / (self.results["control"] * self.n_dates), 2)
                }%"""
            ]
            table_dict[f"{ci_alpha} Upper CI"] = [
                f"""{
                    round(self.results["incrementality_ci_upper"] * 100 / (self.results["control"] * self.n_dates), 2)
                }%"""
            ]
        elif lift == "revenue":
            table_dict["Metric"] = ["Revenue"]
            table_dict["Lift Type "] = ["Incremental"]
            table_dict["Lift"] = [f"""${round(self.results["incrementality"] * self.msrp, 2):,}"""]
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
