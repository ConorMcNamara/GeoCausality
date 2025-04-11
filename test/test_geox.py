import pytest
import pandas as pd

from GeoCausality import geox


class TestGeoX:

    @staticmethod
    def test_geox_results() -> None:
        data_path = "https://raw.githubusercontent.com/juanitorduz/website_projects/master/data/zipcodes_data.csv"
        data_df = pd.read_csv(data_path, parse_dates=["date"])
        data_df["is_test"] = data_df["variant"] == "treatment"
        geo_x = geox.GeoX(
            data_df,
            geo_variable="zipcode",
            treatment_variable="is_test",
            date_variable="date",
            pre_period="2022-06-30",
            post_period="2022-07-01",
            y_variable="orders",
            msrp=7.00,
            spend=500_000,
        )
        geo_x.pre_process().generate()
        # tbr = TBR()
        # tbr.fit(data_df, target="orders", key_geo="zipcode", key_group="variant", key_period="is_campaign",
        # key_date="date", group_control="control", group_treatment="treatment", period_test=True, period_pre=False)
        # tbr.summary(tails=2, report='last')
        #
        # {"date":{"0":1659225600000},"estimate":{"0":170826.0173391506},"precision":{"0":4131.0130609572},
        # "lower":{"0":166695.0042781934},"upper":{"0":174957.0304001078},"scale":{"0":2485.3351529485},
        # "level":{"0":0.9},"probability":{"0":1.0},"posterior_threshold":{"0":0.0},"p_value":{"0":3.22162521e-79}}'
        results = pd.DataFrame(geo_x.results).iloc[-1]
        estimate = 170826.0173391506
        ci_upper = 174957.0304001078
        ci_lower = 166695.0042781934
        p_value = 3.22162521e-79
        assert results["cumulative_incrementality"] == pytest.approx(
            estimate,
            abs=1e-6,
        )
        assert results["cumulative_incrementality_ci_upper"] == pytest.approx(
            ci_upper, abs=1e-6
        )
        assert results["cumulative_incrementality_ci_lower"] == pytest.approx(
            ci_lower, abs=1e-6
        )
        assert results["p_value"] == pytest.approx(p_value, abs=1e-10)

    @staticmethod
    def test_geox_wrongInputs() -> None:
        data_path = "https://raw.githubusercontent.com/juanitorduz/website_projects/master/data/zipcodes_data.csv"
        data_df = pd.read_csv(data_path, parse_dates=["date"])
        data_df["is_test"] = data_df["variant"] == "treatment"
        geo_x = geox.GeoX(
            data_df,
            geo_variable="zipcode",
            treatment_variable="is_test",
            date_variable="date",
            pre_period="2022-06-30",
            post_period="2022-07-01",
            y_variable="orders",
            msrp=7.00,
            spend=500_000,
        )
        geo_x.pre_process().generate()
        with pytest.raises(
            ValueError,
            match="Cannot measure blarg. Choose one of `absolute`, `relative`,  `incremental`, `cost-per`, `revenue` "
            "or `roas`",
        ):
            geo_x.summarize(lift="blarg")


if __name__ == "__main__":
    pytest.main()
