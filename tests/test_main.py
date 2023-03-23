import pandas as pd
import unittest
from datetime import datetime
from kingback.kingback import PriceData, FeatureData, Strategy


class TestPriceData(unittest.TestCase):
    def setUp(self):
        self.price_data = PriceData("TEST")

    def test_validate_utc_datetime_index(self):
        # Test case with invalid index
        df = pd.DataFrame(
            {
                "open": [1, 2, 3],
                "high": [4, 5, 6],
                "low": [7, 8, 9],
                "close": [10, 11, 12],
            }
        )
        with self.assertRaises(Exception):
            self.price_data.validate_utc_datetime_index(df)

        # Test case with valid index but wrong timezone
        df.index = pd.date_range(
            start=datetime(2022, 3, 1), end=datetime(2022, 3, 3), freq="D"
        )
        with self.assertRaises(Exception):
            self.price_data.validate_utc_datetime_index(df)

        # Test case with valid index and timezone
        df.index = pd.date_range(
            start=datetime(2022, 3, 1), end=datetime(2022, 3, 3), freq="D", tz="UTC"
        )
        self.assertIsNone(self.price_data.validate_utc_datetime_index(df))

    def test_validate_columns_dtype(self):
        # Test case with invalid dtype
        df = pd.DataFrame(
            {
                "open": ["1", "2", "A"],
                "high": ["4", "5", "6"],
                "low": ["7", "8", "9"],
                "close": ["10", "11", "12"],
            }
        )
        with self.assertRaises(Exception):
            self.price_data.validate_columns_dtype(df)

        # Test case with valid dtype
        df = pd.DataFrame(
            {
                "open": [1, 2, 3],
                "high": [4, 5, 6],
                "low": [7, 8, 9],
                "close": [10, 11, 12],
            }
        )
        self.assertIsNone(self.price_data.validate_columns_dtype(df))

    def test_validate_columns_exist(self):
        # Test case with missing columns
        df = pd.DataFrame({"open": [1, 2, 3], "high": [4, 5, 6], "low": [7, 8, 9]})
        with self.assertRaises(Exception):
            self.price_data.validate_columns_exist(df)

        # Test case with all columns present
        df = pd.DataFrame(
            {
                "open": [1, 2, 3],
                "high": [4, 5, 6],
                "low": [7, 8, 9],
                "close": [10, 11, 12],
            }
        )
        self.assertIsNone(self.price_data.validate_columns_exist(df))


class TestFeatureData(unittest.TestCase):
    def setUp(self):
        self.fd = FeatureData("TEST")

    def test_validate_utc_datetime_index(self):
        # create data with non-datetime index
        df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        self.assertRaises(Exception, self.fd.validate_utc_datetime_index, df1)

        # create data with datetime index but not in UTC timezone
        df2 = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            index=pd.date_range(
                start="2022-01-01", end="2022-01-03", freq="D", tz="US/Eastern"
            ),
        )
        self.assertRaises(Exception, self.fd.validate_utc_datetime_index, df2)

        # create data with datetime index in UTC timezone
        df3 = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            index=pd.date_range(
                start="2022-01-01", end="2022-01-03", freq="D", tz="UTC"
            ),
        )
        self.assertIsNone(self.fd.validate_utc_datetime_index(df3))


class TestStrategy(unittest.TestCase):
    def test_useData(self):
        # Test that the useData method returns the correct output
        class SimpleStrategy(Strategy):
            def onData(self, i: int):
                return

        strategy = SimpleStrategy()
        price_data = pd.DataFrame(
            {"AAPL": [10, 20, 30], "GOOG": [100, 200, 300]},
            index=["2021-01-01", "2021-01-02", "2021-01-03"],
        )

        expected_output = pd.DataFrame(
            {"AAPL": [10, 20, 30], "GOOG": [100, 200, 300]},
            index=["2021-01-01", "2021-01-02", "2021-01-03"],
        )
        output = strategy.useData(price_data)
        pd.testing.assert_frame_equal(expected_output, output)

    def test_useFeature(self):
        # Test that the useFeature method returns the correct output
        class SimpleStrategy(Strategy):
            def onData(self, i: int):
                return

        strategy = SimpleStrategy()
        feature_data = pd.DataFrame(
            {"Momentum": [0.1, 0.2, 0.3], "Trend": [1, 1, -1]},
            index=["2021-01-01", "2021-01-02", "2021-01-03"],
        )
        expected_output = pd.DataFrame(
            {"Momentum": [0.1, 0.2, 0.3], "Trend": [1, 1, -1]},
            index=["2021-01-01", "2021-01-02", "2021-01-03"],
        )
        output = strategy.useFeature(feature_data)
        pd.testing.assert_frame_equal(expected_output, output)

    def test_useData_and_useFeature(self):
        # Test that the useData and useFeature methods work together correctly
        class SimpleStrategy(Strategy):
            def onData(self, i: int):
                return

        strategy = SimpleStrategy()
        price_data = pd.DataFrame(
            {"AAPL": [10, 20, 30], "GOOG": [100, 200, 300]},
            index=["2021-01-01", "2021-01-02", "2021-01-03"],
        )
        feature_data = pd.DataFrame(
            {"Momentum": [0.1, 0.2, 0.3], "Trend": [1, 1, -1]},
            index=["2021-01-01", "2021-01-02", "2021-01-03"],
        )
        expected_output = pd.DataFrame(
            {
                "AAPL": [10, 20, 30],
                "GOOG": [100, 200, 300],
                "Momentum": [0.1, 0.2, 0.3],
                "Trend": [1, 1, -1],
            },
            index=["2021-01-01", "2021-01-02", "2021-01-03"],
        )
        strategy.useData(price_data)
        output = strategy.useFeature(feature_data)
        pd.testing.assert_frame_equal(expected_output, output)


if __name__ == "__main__":
    unittest.main()
