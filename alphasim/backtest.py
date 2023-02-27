from typing import Callable

import numpy as np
import pandas as pd

from alphasim.portfolio import allocate
from alphasim.commission import zero_commission
from alphasim.money import initial_capital
from alphasim.util import like


CASH = "cash"
EQUITY = "equity"
RESULT_KEYS = [
    "price",
    "funding_rate",
    "start_portfolio",
    "equity",
    "start_weight",
    "target_weight",
    "adj_target_weight",
    "adj_delta_weight",
    "is_trade",
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
    funding_on_abs_position: bool = False,
    trade_buffer: float = 0,
    ignore_buffer_on_new: bool = False,
    ignore_buffer_on_zero: bool = False,
    commission_func: Callable[[float, float], float] = zero_commission,
    fixed_slippage: float = 0,
    initial_capital: float = 1000,
    money_func: Callable[[float, float], float] = initial_capital,
) -> pd.DataFrame | bool:

    if len(prices) == 0:
        raise ValueError("prices length must be greater than 0")

    if len(weights) == 0:
        raise ValueError("weights length must be greater than 0")

    # Ensure prices and weights have the same dimensions
    if prices.shape != weights.shape:
        raise ValueError("shape of prices must match weights")

    # Create empty (zero) funding if none given
    if funding_rates is None:
        funding_rates = like(weights)

    if funding_rates.shape != weights.shape:
        raise ValueError("shape of funding_rates must match weights")

    # Add asset to track cash balance
    prices[CASH] = 1
    weights[CASH] = 0
    funding_rates[CASH] = 0

    # Portfolio to record the units held of a ticker
    port = like(weights)
    port.iloc[0][CASH] = initial_capital

    # Final collated result
    midx = pd.MultiIndex.from_product([weights.index, weights.columns])
    result = pd.DataFrame(index=midx, columns=RESULT_KEYS)
    result[:] = 0

    # Time periods for the given simulation
    periods = len(weights)

    # Step through periods in chronological order
    for i in range(periods):

        # Slice to get data for current period
        # Start port is initialized with the final positions from last period
        if i == 0:
            start_port = port.iloc[0]
        else:
            start_port = port.iloc[i - 1]

        price = prices.iloc[i]
        funding_rate = funding_rates.iloc[i]

        # Mark-to-market the portfolio
        equity = start_port * price
        total = equity.sum()

        # Stop simulation if rekt
        if total <= 0:
            break

        # Set the investable capital used during allocation
        capital = money_func(initial=initial_capital, total=total)

        # Allocate to the portfolio using the latest target weights
        rebal = allocate(
            capital,
            equity,
            weights.iloc[i],
            trade_buffer,
            ignore_buffer_on_new,
            ignore_buffer_on_zero,
            CASH,
        )
        (
            start_weight,
            target_weight,
            adj_target_weight,
            adj_delta_weight,
            trade_value,
        ) = rebal

        # Calc trade size using a fixed slippage factor
        slipped_price = like(price)
        slipped_price[:] = [
            _slippage_price(x, y, fixed_slippage) for x, y in zip(trade_value, price)
        ]
        trade_size = (trade_value / slipped_price).fillna(0)

        # Calc funding payments
        funding_payment = like(equity)
        if funding_on_abs_position:
            funding_payment = equity.abs() * funding_rate
        else:
            funding_payment = equity * funding_rate

        # Calc commission for the traded tickers using the given commission func
        commission = like(trade_value)
        commission[:] = [commission_func(x, y) for x, y in zip(trade_size, trade_value)]

        # Zero out any cash values
        trade_value[CASH] = 0
        trade_size[CASH] = 0
        commission[CASH] = 0
        funding_payment[CASH] = 0

        # Update portfolio and cash positions
        end_port = start_port.copy()
        end_port += trade_size
        end_port[CASH] += (
            trade_value.mul(-1).sum() + commission.sum() + funding_payment.sum()
        )

        # Create mask to indicate if the asset is traded to aid later analysis
        is_trade = trade_size.abs().gt(0)

        # Append data for this time period to the result
        period_result = np.array(
            [
                price,
                funding_rate,
                start_port,
                equity,
                start_weight,
                target_weight,
                adj_target_weight,
                adj_delta_weight,
                is_trade,
                trade_value,
                trade_size,
                funding_payment,
                commission,
                end_port,
            ]
        )
        result.loc[weights.index[i]] = period_result.T

        port.iloc[i] = end_port

    return result


def _slippage_price(delta, price, slippage_pct):

    slippage_price = price

    if delta > 0:
        slippage_price *= 1 + slippage_pct

    if delta < 0:
        slippage_price *= 1 - slippage_pct

    return slippage_price
