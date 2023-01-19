from typing import Callable

import numpy as np
import pandas as pd

from alphasim.commission import zero_commission
from alphasim.money import initial_capital
from alphasim.util import like


CASH = "cash"
EQUITY = "equity"
RESULT_KEYS = [
    "price",
    "funding_rate",
    "start_portfolio",
    EQUITY,
    "start_weight",
    "do_trade",
    "adj_target_weight",
    "adj_delta_weight",
    "trade_value",
    "trade_size",
    "funding_payment",
    "commission",
    "end_portfolio",
]


def backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    funding_rates: pd.DataFrame = None,
    trade_buffer: float = 0,
    do_trade_to_buffer: bool = False,
    do_ignore_buffer_on_new: bool = False,
    do_liquidate_on_zero_weight: bool = False,
    commission_func: Callable[[float, float], float] = zero_commission,
    fixed_slippage: float = 0,
    initial_capital: float = 1000,
    money_func: Callable[[float, float], float] = initial_capital
) -> pd.DataFrame:

    # Ensure prices and weights have the same dimensions
    if prices.shape != weights.shape:
        raise ValueError("shape of prices must match weights")

    # Create empty (zero) funding if none given
    if funding_rates is None:
        funding_rates = like(weights)

    if funding_rates.shape != weights.shape:
        raise ValueError("shape of funding_rates must match weights")

    # Add cash asset to track trade balance changes
    prices[CASH] = 1
    weights[CASH] = 0
    funding_rates[CASH] = 0

    # Portfolio to record the units held of a ticker
    port = like(weights)

    # Final collated result
    midx = pd.MultiIndex.from_product([weights.index, weights.columns])
    result = pd.DataFrame(index=midx, columns=RESULT_KEYS)
    result[:] = 0

    # Time periods for the given simulation
    periods = len(weights)

    # Starting capital is initial capital
    capital = initial_capital

    # Step through periods in chronological order
    for i in range(periods):

        # Portfolio position at start of period
        # Initialize from starting capital if first period
        start_port = port.iloc[i - 1]
        if i == 0:
            start_port[CASH] = capital

        # Slice to get data for current period
        price = prices.iloc[i]
        funding_rate = funding_rates.iloc[i]

        # Mark-to-market the portfolio
        equity = start_port * price
        nav = equity.sum()

        # Stop simulation if rekt
        if nav <= 0:
            break

        # Set the investable capital
        capital = money_func(initial=initial_capital, total=nav)

        # Calc current portfolio weight based on risk capital
        start_weight = equity / capital

        # Calc delta of current to target weight
        target_weight = weights.iloc[i]
        delta_weight = target_weight - start_weight

        # Based on buffer decide if trade should be made
        do_trade = abs(delta_weight) > trade_buffer

        # Calculate adjusted target weight based trade buffer params
        adj_target_weight = target_weight.copy()
        if do_trade_to_buffer:
            adj_target_weight[do_trade] = [
                _buffer_target(x, y, trade_buffer)
                for x, y in zip(target_weight, delta_weight)
            ]

        # If no trade indicated then set adjusted target weight to current weight
        adj_target_weight[~do_trade] = start_weight

        # Open new positions ignoring trade buffer constraint
        if do_ignore_buffer_on_new:
            mask = start_weight.eq(0) & (target_weight.abs().le(trade_buffer))
            do_trade[mask] = True
            adj_target_weight[mask] = target_weight

        # Liquidate open positions in full (do not respect trade buffer)
        if do_liquidate_on_zero_weight:
            mask = start_weight.abs().gt(0) & target_weight.eq(0)
            do_trade[mask] = True
            adj_target_weight[mask] = 0

        # Calc adjusted delta for final trade sizing
        adj_delta_weight = adj_target_weight - start_weight

        # Calc trades required to achieve adjusted target weight using a fixed slippage factor
        trade_value = adj_delta_weight * capital
        slippage_price = [
            _slippage_price(x, y, fixed_slippage)
            for x, y in zip(adj_target_weight, price)
        ]
        trade_size = trade_value / slippage_price

        # Calc funding payments
        funding_payment = like(equity)
        funding_payment = equity * funding_rate

        # Calc commission for the traded tickers using the given commission func
        commission = like(trade_value)
        commission[do_trade] = [
            commission_func(x, y) for x, y in zip(trade_size, trade_value)
        ]
        commission = -commission

        # Zero the cash asset which is not directly traded
        do_trade[CASH] = False
        adj_target_weight[CASH] = 0
        adj_delta_weight[CASH] = 0
        trade_value[CASH] = 0
        trade_size[CASH] = 0
        funding_payment[CASH] = 0
        commission[CASH] = 0

        # Update portfolio and cash changes
        end_port = start_port.copy()
        end_port[CASH] -= trade_value[do_trade].sum()
        end_port[CASH] += commission[do_trade].sum()
        end_port[CASH] += funding_payment.sum()
        end_port[do_trade] += trade_size[do_trade]
        port.iloc[i] = end_port

        # Append data for this time period to the result
        period_result = np.array(
            [
                price,
                funding_rate,
                start_port,
                equity,
                start_weight,
                do_trade,
                adj_target_weight,
                adj_delta_weight,
                trade_value,
                trade_size,
                funding_payment,
                commission,
                end_port,
            ]
        )
        result.loc[weights.index[i]] = period_result.T

    return result


def _slippage_price(target_weight, price, slippage_pct):

    slippage_price = price

    if target_weight > 0:
        slippage_price *= 1 + slippage_pct

    if target_weight < 0:
        slippage_price *= 1 - slippage_pct

    return slippage_price


def _buffer_target(target_weight, delta_weight, trade_buffer):

    buffer_target = target_weight

    if delta_weight > 0:
        buffer_target -= trade_buffer

    if delta_weight < 0:
        buffer_target += trade_buffer

    return buffer_target
