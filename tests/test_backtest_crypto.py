from functools import partial

import pandas as pd
import os

import alphasim.backtest as bt
import alphasim.stats as stats
import alphasim.money as mn
import alphasim.commission as cm


def test_backtest_crypto():
    prices = _load_test_data("crypto_prices.csv")
    weights = _load_test_data("crypto_weights.csv")
    mask = _load_test_data("crypto_mask.csv", bool)
    funding = _load_test_data("crypto_funding.csv")
    tb = 0.05

    result = bt.backtest(
        prices, weights,
        portfolio_mask=mask,
        funding_rates=funding, 
        trade_buffer=tb, 
        money_func=mn.total_equity,
        commission_func=partial(cm.linear_pct_commission, pct_commission=0.001),
        funding_on_abs_position=True
    )
    assert result is not None

    benchmark_prices = prices[["BTCUSDT"]]
    result_stats = stats.backtest_stats(
        result,
        benchmark=benchmark_prices,
        freq=12,
        freq_unit="H",
        trading_days_year=365,
    )

    #assert result.loc[:, "start_portfolio"].isna().sum().sum() == 0

    print(result_stats)


def _load_test_data(filename, dtype=float):
    wd = os.getcwd()
    return pd.read_csv(
        f"{wd}/tests/data/{filename}",
        index_col="dt",
        parse_dates=["dt"],
        dtype=dtype,
    )
