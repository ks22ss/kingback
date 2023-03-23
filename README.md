# Kingback

Kingback is a Python library for doing backtesting in python. It helps you organize your code and modularize different common component encounter during backtesting.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install kingback
```

## Usage

```python
from kingback import (
    PriceData,
    FeatureData,
    Strategy,
    Env,
    Backtest,
    Params,
    Optimizer
)
import pandas as pd
import yfinance as yf

btc = yf.Ticker("BTC-USD")
btchist = btc.history(interval="1d", start="2021-01-01", end="2023-03-01")
btchist.reset_index(inplace=True)
btchist["Date"] = pd.to_datetime(btchist["Date"], utc=True)
btchist = btchist.set_index("Date")

eth = yf.Ticker("ETH-USD")
ethhist = eth.history(interval="1d", start="2021-01-01", end="2023-03-01")
ethhist.reset_index(inplace=True)
ethhist["Date"] = pd.to_datetime(ethhist["Date"], utc=True)
ethhist = ethhist.set_index("Date")


class ComplexStrategy(Strategy):
    def __init__(self):
        super().__init__()

    def onData(self, i: int, env: Env, df: pd.DataFrame):
        now_date = df.loc[i, "Date"]
        now_open = df.loc[i, "open_BTC"]
        now_high = df.loc[i, "high_BTC"]
        now_low = df.loc[i, "low_BTC"]
        now_close = df.loc[i, "close_BTC"]

        env.unrealized_pnl = (now_close - env.pos_open_price) * env.pos
        env.equity_value_list.append(env.last_realized_capital + env.unrealized_pnl)

        # trade logic
        trade_logic = True

        # Open Position
        if env.pos == 0 and trade_logic:
            env.pos = env.last_realized_capital // now_close
            env.pos_open_price = now_close

        # Close Long Position
        elif env.pos > 0:
            env.pos_close_price = now_close
            env.pnl = (
                env.pos_close_price - env.pos_open_price
            ) * env.pos  # if Short, reverse
            env.pnl_list.append(env.pnl)
            env.last_realized_capital = env.last_realized_capital + env.pnl
            env.pos = 0
        return env


params_combs = Params({"tp": [1, 3, 5], "sl": [2, 4]})
p = PriceData("BTC")
p.set_price_data(btchist)

e = FeatureData("ETH")
e.set_feature_data(ethhist)

s = ComplexStrategy()
s.useData(p())
s.useFeature(e())

opt = Optimizer(params_combs=params_combs, strategy=s, default_env=Env(100000))

opt.run()

print(opt.result_sets)







```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
