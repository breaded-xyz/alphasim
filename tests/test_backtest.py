import alphasim.backtest as bt

import pandas as pd


def test_backtest():
    prices = pd.DataFrame()
    weights = pd.DataFrame()
    tb = 0.1

    result = bt.backtest(prices, weights, trade_buffer=tb)

    assert result is not None


def test_backtest_long():
    assert True


def test_backtest_short():

    prices = pd.DataFrame([10, 15, 30], columns=["Acme"])
    weights = pd.DataFrame([-1, -1, -1], columns=["Acme"])
    result = bt.backtest(prices, weights)

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
    tb = 0.25
    result = bt.backtest(prices, weights, trade_buffer=tb, do_limit_trade_size=True)

    # Open a new position but respect the trade buffer.
    assert result.loc[(0, bt.CASH)]["end_portfolio"] == 750
    assert result.loc[(0, "Acme")]["end_portfolio"] == 2.5

    # Delta of current weight (0.75) to target weight (1.25) is 0.5.
    # These weights are a result of price increase and no re-investment flag.
    # We limit our trade size to the minimum required to stay within the buffer zone.
    # Our buffer is 0.25 so we trade to a weight of 1.0.
    assert result.loc[(1, "Acme")]["adj_target_weight"] == 1
    assert result.loc[(1, "Acme")]["adj_delta_weight"] == 0.25

    # Position flips short
    assert result.loc[(2, "Acme")]["adj_target_weight"] == -0.75
    assert result.loc[(2, "Acme")]["adj_delta_weight"] == -1.75

    assert True


def test_backtest_tradetoideal():
    assert True


def test_backtest_mincommission():
    assert True


def test_backtest_leverage():
    assert True


def test_backtest_reinvest():
    assert True
