import alphasim.backtest as bt

import pandas as pd


def test_backtest():
    prices = pd.DataFrame()
    weights = pd.DataFrame()
    trade_buffer = 0.1

    result = bt.backtest(prices, weights, trade_buffer)

    assert result is not None


def test_backtest_long():
    assert True


def test_backtest_short():

    prices = pd.DataFrame([10, 15, 30], columns=["Acme"])
    weights = pd.DataFrame([-1, -1, -1], columns=["Acme"])
    trade_buffer = 0
    result = bt.backtest(prices, weights, trade_buffer)

    # In a short position the trade cost is added to the cash total
    assert result.loc[(0, bt.CASH)]["end_portfolio"] == 2000
    assert result.loc[(0, "Acme")]["end_portfolio"] == -100

    # Asset mark-to-market expsoure is negative
    assert result.loc[(1, bt.CASH)][bt.EQUITY] == 2000
    assert result.loc[(1, "Acme")][bt.EQUITY] == -1500

    # Cash and asset expsoure then correctly net off
    # In this case the price increased and we incurred a loss
    assert result.loc[1, :][bt.EQUITY].sum() == 500


def test_backtest_dolimittradesize():

    prices = pd.DataFrame([100, 300, 300], columns=["Acme"])
    weights = pd.DataFrame([0.5, 1, 1], columns=["Acme"])
    trade_buffer = 0.25
    result = bt.backtest(prices, weights, trade_buffer, do_limit_trade_size=True)

    # When opening a new position (current weight is zero) ignore trade buffer
    assert result.loc[(0, bt.CASH)]["end_portfolio"] == 500
    assert result.loc[(0, "Acme")]["end_portfolio"] == 5

    # Delta of current to target weight is now 0.5. Our buffer is 0.25.
    # We limit our trade size to the minimum required to stay within the buffer zone.
    assert result.loc[(1, "Acme")]["adj_target_weight"] == 1.25
    assert result.loc[(1, "Acme")]["adj_delta_weight"] == -0.25

    assert True


def test_backtest_tradetoideal():
    assert True


def test_backtest_mincommission():
    assert True


def test_backtest_leverage():
    assert True


def test_backtest_reinvest():
    assert True
