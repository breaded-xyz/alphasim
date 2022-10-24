from typing import Callable
from itertools import repeat
import numpy as np
import pandas as pd

from alphasim.util import like
from alphasim.commission import zero_commission
from alphasim.money import initial_capital

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
    "funding",
    "commission",
    "end_portfolio",
]


def backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    funding_rates: pd.DataFrame = None,
    trade_buffer: float = 0,
    do_trade_to_buffer: bool = False,
    commission_func: Callable[[float, float], float] = zero_commission,
    initial_capital: float = 1000,
    money_func: Callable[[float, float], float] = initial_capital,
) -> pd.DataFrame:

    # Ensure prices and weights have the same dimensions
    if prices.shape != weights.shape:
        raise ValueError("shape of prices must match weights")

    if funding_rates is None:
        funding_rates = like(weights)

    if funding_rates.shape != weights.shape:
        raise ValueError("shape of funding_rates must match weights")

    # Add cash asset to track trade balance changes
    prices[CASH] = 1.0
    weights[CASH] = 1.0 - weights.abs().sum(axis=1)
    funding_rates[CASH] = 0.0

    # Portfolio to record the units held of a ticker
    port_df = like(weights)

    # Track mark-to-market for the portfolio
    equity_df = like(port_df)

    # Final collated result returned to caller
    midx = pd.MultiIndex.from_product([weights.index, weights.columns])
    result_df = pd.DataFrame(index=midx, columns=RESULT_KEYS)
    result_df[:] = 0.0

    # Time periods for the given simulation
    periods = len(weights)

    # Starting capital is initial capital
    capital = initial_capital

    # Step through periods in chronological order
    for i in range(periods):

        # Portfolio position at start of period
        # Initialize from starting capital if first period
        start_port = port_df.iloc[i - 1]
        if i == 0:
            start_port[CASH] = capital

        # Slice to get data for current period
        price = prices.iloc[i]
        funding_rate = funding_rates.iloc[i]
        equity = equity_df.iloc[i]

        # Mark-to-market the portfolio
        equity = start_port * price
        nav = equity.sum()

        # Stop simulation if rekt
        if nav <= 0:
            break

        # Set the investable capital
        capital = money_func(initial=initial_capital, total=nav)

        # Calc current portfolio weight based on risk capital
        curr_weight = equity / capital

        # Calc delta of current to target weight
        target_weight = weights.iloc[i]
        delta_weight = target_weight - curr_weight

        # Based on buffer decide if trade should be made
        do_trade = abs(delta_weight) > trade_buffer
        do_trade[CASH] = False

        # Default is to trade to the ideal target weight when commission is a fixed minimum or zero
        # Trade to buffer when commission is a linear pct of trade value (e.g. crypto)
        # Always trade to target weight when opening new poistion (i.e. current weight is zero)
        adj_target_weight = target_weight.copy()
        if do_trade_to_buffer:
            adj_target_weight[do_trade] = [
                _buffer_target(x, y, z)
                for x, y, z in zip(target_weight, delta_weight, repeat(trade_buffer))
            ]

        # If no trade indicated then set target weight to current weight
        adj_target_weight[~do_trade] = curr_weight

        # Calc adjusted delta for final trade sizing
        adj_delta_weight = adj_target_weight - curr_weight

        # Calc trades required to achieve adjusted target weight
        trade_value = adj_delta_weight * capital
        trade_size = trade_value / price

        # Calc funding payments
        funding = like(equity)
        funding = equity * funding_rate

        # Calc commission for the traded tickers using the given commission func
        commission = like(trade_value)
        commission[do_trade] = [
            commission_func(x, y) for x, y in zip(trade_size, trade_value)
        ]
        commission = -commission

        # Calc post trade port positions
        # Account for changes to cash from trade activity
        end_port = start_port.copy()
        end_port[do_trade] += trade_size[do_trade]
        end_port[CASH] -= trade_value.loc[do_trade].sum()
        end_port[CASH] += commission.loc[do_trade].sum()
        end_port[CASH] += funding.sum()
        port_df.iloc[i] = end_port

        # Append data for this time period to the result
        period_result = np.array(
            [
                price,
                equity,
                do_trade,
                adj_target_weight,
                adj_delta_weight,
                trade_value,
                trade_size,
                funding,
                commission,
                end_port,
            ]
        )
        result_df.loc[weights.index[i]] = period_result.T

    return result_df


def _buffer_target(target_weight, delta_weight, trade_buffer):

    target = target_weight

    if delta_weight > 0:
        target -= trade_buffer

    if delta_weight < 0:
        target += trade_buffer

    return target
