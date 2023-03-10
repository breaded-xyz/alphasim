import os

import pandas as pd

import alphasim.backtest as bt
import alphasim.stats as stats


def test_backtest_stonks():
    prices = _load_test_data("stonk_prices.csv").fillna(0)
    weights = _load_test_data("stonk_weights.csv").fillna(0)
    tb = 0.1

    result = bt.backtest(prices, weights, trade_buffer=tb, discrete_shares=True)
    assert result is not None

    benchmark_prices = prices[["VTI"]]
    result_stats = stats.backtest_stats(result, benchmark=benchmark_prices)

    print(result_stats)


def _load_test_data(filename, dtype=float):
    wd = os.getcwd()
    return pd.read_csv(
        f"{wd}/tests/data/{filename}",
        index_col="dt",
        parse_dates=["dt"],
        dtype=dtype,
    )
