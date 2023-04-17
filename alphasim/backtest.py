from typing import Callable, cast

import numpy as np
import pandas as pd

from alphasim.commission import zero_commission
from alphasim.money import initial_capital
from alphasim.portfolio import allocate
from alphasim.util import fillnan, like

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
    "quote_qty",
    "base_qty",
    "funding_payment",
    "commission",
    "end_portfolio",
]


def backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    funding_rates: pd.DataFrame | None = None,
    funding_on_abs_position: bool = False,
    trade_buffer: float = 0,
    commission_func: Callable[[float, float], float] = zero_commission,
    initial_capital: float = 1000,
    money_func: Callable[[float, float], float] = initial_capital,
    discrete_shares: bool = False,
    short_f: float = 1,
    spread_f: float = 0,
) -> pd.DataFrame:
    # Validate args
    if len(prices) == 0:
        raise ValueError("prices length must be greater than 0")

    if prices.isna().sum().sum() != 0:
        raise ValueError("prices must not have any NaNs")

    if len(weights) == 0:
        raise ValueError("weights length must be greater than 0")

    if weights.isna().sum().sum() != 0:
        raise ValueError("weights must not have any NaNs")

    if prices.shape != weights.shape:
        raise ValueError("shape of prices must match weights")

    # Create empty (zero) funding if none given
    if funding_rates is None:
        funding_rates = cast(pd.DataFrame, like(weights))

    if funding_rates.isna().sum().sum() != 0:
        raise ValueError("funding must not have any NaNs")

    if funding_rates.shape != weights.shape:
        raise ValueError("shape of funding_rates must match weights")

    # Track cash balance
    cash = initial_capital

    # By default we allow partial shares to be
    # transacted in lots of size 1 in the quote currency.
    # Or we set lot_sizes to None which will enforce
    # that only whole shares can be transacted.
    lot_sizes = cast(pd.Series, like(prices.iloc[0], 1))
    if discrete_shares:
        lot_sizes = None

    # Portfolio to record the units held of a ticker
    port = cast(pd.DataFrame, like(weights))

    # Final collated result for all assets and cash position
    asset_list = weights.columns.tolist()
    asset_list.append(CASH)
    midx = pd.MultiIndex.from_product([weights.index, asset_list])
    result = pd.DataFrame(index=midx, columns=RESULT_KEYS)
    result[:] = 0

    # Time periods for the given simulation
    periods = len(weights)

    # Step through periods in chronological order
    for i in range(periods):
        start_cash = cash

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
        total = equity.sum() + cash

        # Stop simulation if rekt
        if total <= 0:
            break

        # Set the investable capital used during allocation
        capital = money_func(initial_capital, total)

        # Use target weight direction to apply spread factor to the price
        target_weight = weights.iloc[i]
        quote = price.copy()
        if spread_f > 0:
            quote[:] = [
                quote_spread(x, y, spread_f) for x, y in zip(price, target_weight)
            ]

        # Allocate to the portfolio using the latest target weights and quote price
        rebal = allocate(
            capital,
            quote,
            equity,
            target_weight,
            trade_buffer,
            lot_sizes,
            short_f,
        )
        (
            start_weight,
            target_weight,
            adj_target_weight,
            adj_delta_weight,
            base_qty,
            quote_qty,
        ) = rebal

        # Ensure consistency by filling with zero
        base_qty = fillnan(base_qty, 0)
        quote_qty = fillnan(quote_qty, 0)

        # Support rotating portfolios by ignoring the buffer
        # and forcing liquidations on a zero target weight
        liquidate = start_port.abs().gt(0) & target_weight.eq(0)
        adj_target_weight[liquidate] = 0
        adj_delta_weight[liquidate] = target_weight - start_weight
        base_qty[liquidate] = start_port.mul(-1)
        quote_qty[liquidate] = base_qty * price

        # Calc funding payments
        funding_payment = like(equity)
        if funding_on_abs_position:
            funding_payment = equity.abs() * funding_rate
        else:
            funding_payment = equity * funding_rate

        # Calc commission for the traded tickers using the given commission func
        commission = like(quote_qty)
        commission[:] = [
            commission_func(float(x), float(y)) for x, y in zip(base_qty, quote_qty)
        ]

        # Create mask to indicate if the asset is traded to aid later analysis
        is_trade = base_qty.abs().gt(0)

        # Update portfolio
        end_port = start_port.copy()
        end_port += base_qty
        port.iloc[i] = end_port

        # Update cash position
        cash = (
            start_cash
            + quote_qty.mul(-1).sum()
            + commission.sum()
            + funding_payment.sum()
        )

        # Record cash value results
        price[CASH] = 1
        start_port[CASH] = start_cash
        equity[CASH] = start_cash
        start_weight[CASH] = start_cash / capital
        end_port[CASH] = cash

        # Add empty value so arrays align for fast insertion to results
        funding_rate[CASH] = None
        target_weight[CASH] = None
        adj_target_weight[CASH] = None
        adj_delta_weight[CASH] = None
        is_trade[CASH] = None
        quote_qty[CASH] = None
        base_qty[CASH] = None
        funding_payment[CASH] = None
        commission[CASH] = None

        # Append data for this time period to the result
        period_results = np.array(
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
                quote_qty,
                base_qty,
                funding_payment,
                commission,
                end_port,
            ]
        )
        result.loc[weights.index[i]] = period_results.T

    return result


def quote_spread(mid: float, target_weight: float, f: float) -> float:
    quote = mid
    spread = mid * f
    half = spread / 2

    if target_weight > 0:
        quote += half
    elif target_weight < 0:
        quote -= half

    return quote
