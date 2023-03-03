from functools import partial

import pandas as pd

import alphasim.backtest as bt
import alphasim.money as mn
import alphasim.commission as cn

def test_backtest_long():
    prices = pd.DataFrame([10, 15, 30], columns=["Acme"])
    weights = pd.DataFrame([1, 1, 0], columns=["Acme"])
    result = bt.backtest(prices, weights, money_func=mn.total_equity)

    # In a long position the trade value is substracted from the cash total
    assert result.loc[(0, bt.CASH)]["end_portfolio"] == 0
    assert result.loc[(0, "Acme")]["end_portfolio"] == 100

    # Asset price increased
    assert result.loc[(1, bt.CASH)][bt.EQUITY] == 0
    assert result.loc[(1, "Acme")][bt.EQUITY] == 1500

    # Liquidate position and realize gain
    assert result.loc[2, bt.CASH]["start_portfolio"] == 0
    assert result.loc[2, bt.CASH]["end_portfolio"] == 3000


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
    result = bt.backtest(prices, weights, trade_buffer=0.25)

    # Open a new position with respect to the trade buffer
    assert result.loc[(0, bt.CASH)]["end_portfolio"] == 750
    assert result.loc[(0, "Acme")]["end_portfolio"] == 2.5

    # Delta of current weight (0.75) to target weight (1.25) is 0.5.
    # These weights are a result of price increase and no re-investment flag.
    # We limit our trade size to the minimum required to stay within the buffer zone.
    # Our buffer is 0.25 so we trade to a weight of 1.0.
    assert result.loc[(1, "Acme")]["target_weight"] == 1.25
    assert result.loc[(1, "Acme")]["adj_delta_weight"] == 0.25

    # Position flips short
    assert result.loc[(2, "Acme")]["adj_delta_weight"] == -1.75

    # Continue short
    assert result.loc[(3, "Acme")]["adj_delta_weight"] == -1.25

    # Position reverses
    assert result.loc[(4, "Acme")]["adj_delta_weight"] == 1.5


def test_backtest_fundingrates():

    prices = pd.DataFrame([100, 100, 100, 100, 100], columns=["Acme"])
    weights = pd.DataFrame([1, 1, 1, -1, -1], columns=["Acme"])
    rates = pd.DataFrame([0.1, 0.1, -0.2, -0.2, -0.2], columns=["Acme"])
    result = bt.backtest(prices, weights, funding_rates=rates)

    # Funding is paid on the positions from the previous period, 
    # so no impact when i == 0
    assert result.loc[(0, "Acme")]["funding_payment"] == 0

    # Positive rate so we get paid 10% on our 1K position
    assert result.loc[(1, "Acme")]["funding_payment"] == 100

    # Funding flips negative so we deduct 20% from our long
    assert result.loc[(2, "Acme")]["funding_payment"] == -200

    # Now short on negative funding so get paid
    assert result.loc[(4, "Acme")]["funding_payment"] == 200

def test_backtest_abs_fundingrates():

    prices = pd.DataFrame([100, 100, 100, 100, 100], columns=["Acme"])
    weights = pd.DataFrame([1, 1, 1, -1, -1], columns=["Acme"])
    rates = pd.DataFrame([-0.1, -0.1, -0.2, -0.2, -0.2], columns=["Acme"])
    result = bt.backtest(prices, weights, funding_rates=rates, funding_on_abs_position=True)

    # Funding is paid on the positions from the previous period, 
    # so no impact when i == 0
    assert result.loc[(0, "Acme")]["funding_payment"] == 0

    # In abs mode the sign of the position is ignored
    assert result.loc[(1, "Acme")]["funding_payment"] == -100
    assert result.loc[(2, "Acme")]["funding_payment"] == -200
    assert result.loc[(4, "Acme")]["funding_payment"] == -200

def test_backtest_commission():

    prices = pd.DataFrame([10, 15, 30], columns=["Acme"])
    weights = pd.DataFrame([0.5, 1, 0], columns=["Acme"])

    cmn = partial(cn.linear_pct_commission, pct_commission=0.1)
    result = bt.backtest(prices, weights, commission_func=cmn)

    assert result.loc[(0, bt.CASH)]["end_portfolio"] == 450

def test_backtest_leverage_long():
    prices = pd.DataFrame([10, 10, 30], columns=["Acme"])
    weights = pd.DataFrame([1, 2, 0], columns=["Acme"])
    result = bt.backtest(prices, weights, money_func=mn.total_equity)

    # Initiate long position with leverage 1x
    assert result.loc[(0, bt.CASH)]["end_portfolio"] == 0
    assert result.loc[(0, "Acme")]["end_portfolio"] == 100

    # Increase leverage to 2x based on a portfolio NAV of 1000
    assert result.loc[(1, bt.CASH)]["end_portfolio"] == -1000
    assert result.loc[(1, "Acme")]["end_portfolio"] == 200

    # Liquidate position and realize leveraged gain
    assert result.loc[2, bt.CASH]["start_portfolio"] == -1000
    assert result.loc[2, bt.CASH]["end_portfolio"] == 5000


def test_backtest_leverage_short():
    prices = pd.DataFrame([10, 10, 30], columns=["Acme"])
    weights = pd.DataFrame([-1, -2, 0], columns=["Acme"])
    result = bt.backtest(prices, weights, money_func=mn.total_equity)

    # Initiate short position with leverage 1x
    assert result.loc[(0, bt.CASH)]["end_portfolio"] == 2000
    assert result.loc[(0, "Acme")]["end_portfolio"] == -100

    # Increase leverage to 2x based on a portfolio NAV of 1000
    assert result.loc[(1, bt.CASH)]["end_portfolio"] == 3000
    assert result.loc[(1, "Acme")]["end_portfolio"] == -200