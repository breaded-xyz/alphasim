import os
import time
from functools import partial

import pandas as pd

import alphasim.backtest as bt
import alphasim.commission as cm
import alphasim.money as mn
import alphasim.stats as stats


def test_backtest_crypto():
    prices = _load_test_data("crypto_prices.csv").fillna(0)
    weights = _load_test_data("crypto_weights.csv").fillna(0)
    funding = _load_test_data("crypto_funding.csv").fillna(0)
    tb = 0.05

    t0 = time.perf_counter()
    result = bt.backtest(
        prices,
        weights,
        funding_rates=funding,
        trade_buffer=tb,
        money_func=mn.total_equity,
        commission_func=partial(cm.linear_pct_commission, pct_commission=0.001),
        funding_on_abs_position=True,
    )
    t1 = time.perf_counter()

    assert result is not None

    print(t1 - t0)

    benchmark_prices = prices[["BTCUSDT"]]
    result_stats = stats.backtest_stats(
        result,
        benchmark=benchmark_prices,
        freq=24,
        freq_unit="H",
        trading_days_year=365,
    )

    print(result_stats)


def _load_test_data(filename, dtype=float):
    wd = os.getcwd()
    return pd.read_csv(
        f"{wd}/tests/data/{filename}",
        index_col="dt",
        parse_dates=["dt"],
        dtype=dtype,
    )
