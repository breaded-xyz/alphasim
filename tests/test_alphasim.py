from alphasim.backtest import backtest, CASH

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

    prices = pd.DataFrame([10, 15, 30], columns=["Acme"])
    weights = pd.DataFrame([-1, -1, -1], columns=["Acme"])
    trade_buffer = 0
    result = backtest(prices, weights, trade_buffer)

    # In a short position the trade cost is added to the cash total
    assert result.loc[(0, CASH)]["end_portfolio"] == 2000
    assert result.loc[(0, "Acme")]["end_portfolio"] == -100

    # Asset mark-to-market expsoure is negative
    assert result.loc[(1, CASH)]["exposure"] == 2000
    assert result.loc[(1, "Acme")]["exposure"] == -1500

    # Cash and asset expsoure then correctly net off
    # In this case the price increased and we incurred a loss
    assert result.loc[1, :]["exposure"].sum() == 500


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
