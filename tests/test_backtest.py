from operator import index
import numpy as np
import pandas as pd

import alphasim.backtest as bt
import alphasim.money as mo


def test_backtest():
    prices = pd.DataFrame()
    weights = pd.DataFrame()
    result = bt.backtest(prices, weights)
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


def test_backtest_tradetobuffer():

    prices = pd.DataFrame([100, 300, 300, 200, 200], columns=["Acme"])
    weights = pd.DataFrame([0.5, 1.25, -1, -2, 0], columns=["Acme"])
    result = bt.backtest(prices, weights, trade_buffer=0.25, do_trade_to_buffer=True)

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

    # Continue short
    assert result.loc[(3, "Acme")]["adj_target_weight"] == -1.75
    assert result.loc[(3, "Acme")]["adj_delta_weight"] == -1.25

    # Position reverses
    assert result.loc[(4, "Acme")]["adj_target_weight"] == -0.25
    assert result.loc[(4, "Acme")]["adj_delta_weight"] == 1.5

    assert True


def test_backtest_fundingrates():

    prices = pd.DataFrame([100, 100, 100, 100, 100], columns=["Acme"])
    weights = pd.DataFrame([1, 1, 1, -1, -1], columns=["Acme"])
    rates = pd.DataFrame([0.1, 0.1, -0.2, -0.2, -0.2], columns=["Acme"])
    result = bt.backtest(prices, weights, rates)

    # Funding is paid on the positions from the previous period, so no impact when i == 0
    assert result.loc[(0, "Acme")]["funding_payment"] == 0

    # Positive rate so we get paid 10% on our 1K position
    assert result.loc[(1, "Acme")]["funding_payment"] == 100

    # Funding flips negative so we deduct 20% from our long
    assert result.loc[(2, "Acme")]["funding_payment"] == -200

    # Now short on negative funding so get paid
    assert result.loc[(4, "Acme")]["funding_payment"] == 200

    assert True


def test_backtest_tradetoideal():
    assert True


def test_backtest_commission():
    assert True


def test_backtest_reinvest_sqrt():

    prices = pd.DataFrame([10, 20, 30], columns=["Acme"])
    weights = pd.DataFrame([1, 1, 1], columns=["Acme"])
    result = bt.backtest(prices, weights, money_func=mo.sqrt_profit)

    assert result.loc[(0, bt.CASH)]["end_portfolio"] == 0
    assert result.loc[(0, "Acme")]["end_portfolio"] == 100

    # NAV of portfolio has doubled to 2000
    # But we only reinvest sqrt of profit c.1414
    assert result.loc[(1, bt.CASH)]["end_portfolio"] == 585.7864376269048
    assert result.loc[(1, "Acme")]["end_portfolio"] == 70.71067811865476

    # TODO: test for short side equity

    assert True

def test_backtest_finalportfolio():

    prices = pd.DataFrame([100, 300, 300, 200, 200], columns=["Acme"])
    weights = pd.DataFrame([0.5, 1.25, -1, -2, -2], columns=["Acme"])
    final_portfolio = pd.Series([0, 400], index=["Acme", bt.CASH])

    result = bt.backtest(prices, weights, final_portfolio=final_portfolio)

    # Confirm last period was initiated with the given final portfolio
    assert result.loc[(4, bt.CASH)]["start_portfolio"] == 400
    assert result.loc[(4, "Acme")]["start_portfolio"] == 0

    assert True
