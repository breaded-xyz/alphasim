from typing import Callable
import numpy as np
import pandas as pd

CASH = "cash"
EQUITY = "equity"
RESULT_KEYS = [
    "price",
    EQUITY,
    "do_trade",
    "adj_target_weight",
    "adj_delta_weight",
    "trade_value",
    "trade_size",
    "commission",
    "end_portfolio",
]


def zero_commission(trade_size, trade_value):
    return 0


def backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    trade_buffer: float,
    do_limit_trade_size: bool = False,
    commission_func: Callable[[float, float], float] = zero_commission,
    initial_capital: float = 1000,
    do_reinvest: bool = False,
) -> pd.DataFrame:

    # Ensure prices and weights have the same dimensions
    if prices.shape != weights.shape:
        raise ValueError()

    # Add cash asset to track trade balance changes
    prices[CASH] = 1.0
    weights[CASH] = 1.0 - weights.abs().sum(axis=1)

    # Portfolio to record the units held of a ticker
    port_df = pd.DataFrame(index=weights.index, columns=weights.columns, dtype="float")
    port_df[:] = 0.0

    # Track mark-to-market for the portfolio
    equity_df = port_df.copy()

    # Final collated result returned to caller
    midx = pd.MultiIndex.from_product([weights.index, weights.columns])
    result_df = pd.DataFrame(index=midx, columns=RESULT_KEYS)

    # Time periods for the given simulation
    periods = len(weights)

    # Default is not to re-invest profits and fix risk capital at initial capital
    risk_capital = initial_capital

    # Step through periods in chronological order
    for i in range(periods):

        # Portfolio position at start of period
        # Initialize from starting capital if first period
        start_port = port_df.iloc[i - 1]
        if i == 0:
            start_port[CASH] = initial_capital

        # Slice to get data for current period
        price = prices.iloc[i]
        equity = equity_df.iloc[i]

        # Mark-to-market the portfolio
        equity = start_port * price
        nav = equity.sum()

        # Stop simulation if rekt
        if nav <= 0:
            break

        # Set the risk capital
        if do_reinvest:
            risk_capital = nav

        # Calc current portfolio weight based on risk capital
        curr_weight = equity / risk_capital

        # Calc delta of current to target weight
        target_weight = weights.iloc[i]
        delta_weight = target_weight - curr_weight

        # Based on buffer decide if trade should be made
        do_trade = (delta_weight.abs() > trade_buffer) | (curr_weight == 0)
        do_trade[CASH] = False

        # Default is to trade to the ideal target weight when commission is a fixed minimum or zero
        # Trade to buffer when commission is a linear pct of trade value (e.g. crypto)
        # Always trade to target weight when opening new poistion (i.e. current weight is zero)
        adj_target_weight = target_weight.copy()
        if do_limit_trade_size:
            adj_target_weight.loc[do_trade] += delta_weight.abs() - trade_buffer
            adj_target_weight.loc[curr_weight == 0] = target_weight

        # If no trade indicated then set target weight to current weight
        adj_target_weight.loc[~do_trade] = curr_weight

        # Calc adjusted delta for final trade sizing
        adj_delta_weight = adj_target_weight - curr_weight

        # Calc trades required to achieve adjusted target weight
        trade_value = adj_delta_weight * risk_capital
        trade_size = trade_value / price

        # Calc commission for the traded tickers using the given commission func
        commission = trade_value.copy()
        commission[do_trade] = [
            commission_func(x, y) for x, y in zip(trade_size, trade_value)
        ]

        # Calc post trade port positions
        # Account for changes to cash from trade activity
        end_port = start_port.copy()
        end_port[do_trade] += trade_size[do_trade]
        end_port[CASH] -= trade_value.loc[do_trade].sum()
        end_port[CASH] -= commission.loc[do_trade].sum()
        port_df.iloc[i] = end_port

        # Append data for this time period to the result
        result_df.loc[weights.index[i]] = np.array(
            [
                price,
                equity,
                do_trade,
                adj_target_weight,
                adj_delta_weight,
                trade_value,
                trade_size,
                commission,
                end_port,
            ]
        ).T

    return result_df
