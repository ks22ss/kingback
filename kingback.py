import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pytz import UTC
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import itertools
import copy


class Params:
    def __init__(self, params: dict[list]):
        self.comb_dict_list = self.generate_combination(params)

    def __call__(self):
        print(self.comb_dict_list)
        return self.comb_dict_list

    def generate_combination(self, parmas: dict[list]):
        params_keys = [k for k in parmas.keys()]
        params_values = [v for v in parmas.values()]
        comb_list = list(itertools.product(*params_values))
        print(comb_list)
        comb_dict_list = []
        for comb in comb_list:
            comb_dict = {}
            for i, k in enumerate(params_keys):
                comb_dict[f"{k}"] = comb[i]
            comb_dict_list.append(comb_dict)
        return comb_dict_list


class PriceData:
    def __init__(self, name: str):
        self.name = name
        self.price_data = pd.DataFrame()

    def __call__(self) -> pd.DataFrame:
        return self.price_data

    def set_price_data(self, price_data: pd.DataFrame):
        """
        Any pd.DataFrame with
        1) a datetime64 utc index
        2) open, high, low, close
        3) correct column data type that could convert to float
        """
        price_data.columns = [col.lower() for col in price_data.columns]

        self.validate_columns_exist(price_data)
        self.validate_columns_dtype(price_data)
        self.validate_utc_datetime_index(price_data)

        # Set the data
        price_data.columns = [f"{c}_{self.name}" for c in price_data.columns]
        self.price_data = price_data

    def validate_utc_datetime_index(self, priceData: pd.DataFrame):
        if not isinstance(priceData.index, pd.DatetimeIndex):
            raise Exception(f"Index is not datetime index.")

        elif priceData.index.tzinfo != UTC:
            raise Exception(f"Index is not format as datetime64[ns, UTC].")

    def validate_columns_dtype(self, priceData: pd.DataFrame):
        dtype_err = []
        col_include = ["open", "high", "low", "close"]
        for col in col_include:
            # Check if column exists
            if col in priceData.columns:
                # Check if it can convert to numeric
                try:
                    priceData[col].astype("float")
                except Exception as e:
                    dtype_err.append(col)
        if len(dtype_err) > 0:
            raise Exception(f"Column(s) {dtype_err} cannot be convert into float.")

    def validate_columns_exist(self, priceData: pd.DataFrame):
        validate_err = []
        col_include = ["open", "high", "low", "close"]
        for col in col_include:
            # Check if column exists
            if col not in priceData.columns:
                validate_err.append(col)

        print(validate_err)
        if len(validate_err) > 0:
            raise Exception(
                f"Column(s) {validate_err} needs to include in the price data."
            )


class FeatureData:
    def __init__(self, name: str):
        self.feature_data = pd.DataFrame()
        self.name = name

    def __call__(self) -> pd.DataFrame:
        return self.feature_data

    def set_feature_data(self, feature_data: pd.DataFrame):
        """
        Any pd.DataFrame with a datetime64 utc index
        """
        feature_data.columns = [col.lower() for col in feature_data.columns]

        self.validate_utc_datetime_index(feature_data)

        feature_data.columns = [f"{c}_{self.name}" for c in feature_data.columns]
        self.feature_data = feature_data

    def validate_utc_datetime_index(self, featureData: pd.DataFrame):
        if not isinstance(featureData.index, pd.DatetimeIndex):
            raise Exception(f"Index is not datetime index.")

        elif featureData.index.tzinfo != UTC:
            raise Exception(f"Index is not format as datetime64[ns, UTC].")


@dataclass
class Env:
    last_realized_capital: float
    unrealized_pnl: float = 0
    pos: float = 0
    pos_open_price: float = 0
    pos_close_price: float = 0
    pnl: float = 0
    pnl_list: list = field(default_factory=list)
    equity_value_list: list = field(default_factory=list)
    dd_dollar_list: list = field(default_factory=list)
    dd_pct_list: list = field(default_factory=list)
    params: dict = field(default_factory=dict)


class Strategy(ABC):
    def __init__(self):
        self.df = pd.DataFrame()
        self.param = {}

    def useData(self, price_data: pd.DataFrame):
        if len(self.df) == 0:
            self.df.index = price_data.index

        self.df = pd.concat([self.df, price_data], axis=1, join="outer")
        return self.df

    def useFeature(self, feature_data: pd.DataFrame):
        if len(self.df) == 0:
            self.df.index = feature_data.index

        self.df = pd.concat([self.df, feature_data], axis=1, join="outer")
        return self.df

    def useParam(self, param: dict):
        self.param = param

    def columns(self):
        return self.df.columns

    @abstractmethod
    def onData(self, i: int, env: Env, df: pd.DataFrame):
        pass


class Backtest:
    def __init__(self, strategy: Strategy):
        self.strategy = strategy

    def setEnv(self, env: Env):
        self.env = env

    def run(self):
        # initialize
        df = self.strategy.df.reset_index()
        env = self.env

        # Begin Loop
        for i in range(len(df)):
            env = self.strategy.onData(i, env, df)  # Keep update the env

        # Report
        print(f"Total Profit: {sum(env.pnl_list)}")

        return {
            "Count Trades": len(env.pnl_list),
            "Total Profit": sum(env.pnl_list),
            "Average Profit": sum(env.pnl_list) / len(env.pnl_list),
        }


class Optimizer:
    def __init__(
        self, params_combs: Params, strategy: Strategy, default_env: Env
    ) -> None:
        self.params_combs = params_combs
        self.s = strategy
        self.default_env = default_env
        self.result_sets = []

    def run(self):
        for param in self.params_combs.comb_dict_list:
            self.s.useParam(param)

            tester = Backtest(self.s)
            default_env = copy.deepcopy(self.default_env)
            tester.setEnv(default_env)

            result = tester.run()

            self.result_sets.append((param, result))
