import numpy as np
import pandas as pd
import os

import alphasim.backtest as bt
import alphasim.stats as stats
import alphasim.money as mn

def test_backtest_crypto():
    prices = _load_test_data("crypto_prices.csv")
    weights = _load_test_data("crypto_weights.csv")
    funding = _load_test_data("crypto_funding.csv")
    tb = 0.05

    result = bt.backtest(prices, weights, funding, trade_buffer=tb, money_func=mn.total_equity)
    assert result is not None

    benchmark_prices = prices[["BTCUSDT"]]
    result_stats = stats.backtest_stats(
        result, benchmark=benchmark_prices, 
        freq=12, freq_unit="H", trading_days_year=365)

    assert result.loc[:,"start_portfolio"].isna().sum().sum() == 0

    print(result_stats)


def _load_test_data(filename):
    wd = os.getcwd()
    return pd.read_csv(
        f"{wd}/tests/data/{filename}",
        index_col="dt",
        parse_dates=["dt"],
        dtype=np.float64,
    )
