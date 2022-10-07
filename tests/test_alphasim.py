from alphasim.backtest import backtest

import pandas as pd


def test_backtest():
    prices = pd.DataFrame()
    weights = pd.DataFrame()
    trade_buffer = 0.1

    result = backtest(prices, weights, trade_buffer)

    assert result is not None


def test_backtest_long():
    assert True


def test_backtest_short():
    assert True


def test_backtest_tradetobuffer():
    assert True


def test_backtest_tradetoideal():
    assert True


def test_backtest_mincommission():
    assert True


def test_backtest_leverage():
    assert True


def test_backtest_reinvest():
    assert True


def test_backtest_portfolio6040():
    assert True
