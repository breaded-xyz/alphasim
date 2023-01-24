import numpy as np
import pandas as pd
import os

import alphasim.backtest as bt
import alphasim.stats as stats


def test_bench_backtest():
    prices = _load_test_data("price_sample.csv")
    weights = _load_test_data("weight_sample.csv")
    tb = 0.1

    result = bt.backtest(prices, weights, trade_buffer=tb)
    assert result is not None

    benchmark_prices = prices[["VTI"]]
    result_stats = stats.backtest_stats(result, benchmark=benchmark_prices)

    print(result_stats)


def _load_test_data(filename):
    wd = os.getcwd()
    return pd.read_csv(
        f"{wd}/tests/data/{filename}",
        index_col="Date",
        parse_dates=["Date"],
        dtype={
            "VTI": np.float64,
            "TLT": np.float64,
            "GLD": np.float64,
        },
    )
