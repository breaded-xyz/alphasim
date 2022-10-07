from typing import Callable
import pandas as pd

CASH = "cash"


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
    assert prices.shape == weights.shape

    # Coerce to floats
    prices = prices.astype("float")
    weights = weights.astype("float")

    # Add cash asset to track trade balance changes
    prices[CASH] = 1.0
    weights[CASH] = 1.0 - weights.sum(axis=1)

    # Portfolio to record the units held of a ticker
    port_df = pd.DataFrame(index=weights.index, columns=weights.columns, dtype="float")
    port_df[:] = 0.0

    # Track mark-to-market for the portfolio
    exposure_df = port_df.copy(deep=True)

    # Final collated result returned to caller
    result_df = pd.DataFrame()

    # Time periods for the given simulation
    periods = len(weights)

    # Step through periods in chronological order
    for i in range(periods):

        # Portfolio position at start of period
        start_port = start_port = port_df.iloc[0].copy(deep=True)

        # First period initialize from initial capital
        if i == 0:
            start_port["cash"] = initial_capital
        else:
            start_port = port_df.iloc[i - 1].copy(deep=True)

        # Slice to get data for current period
        price = prices.iloc[i]
        exposure = exposure_df.iloc[i]

        # Mark-to-market the portfolio
        exposure = start_port * price
        nav = exposure.sum()

        # Stop simulation if rekt
        if nav <= 0:
            break

        # Set the risk capital
        risk_capital = float(initial_capital)
        if do_reinvest:
            risk_capital = nav

        # Calc current portfolio weight based on risk capital
        curr_weight = exposure / risk_capital

        # Calc delta of current to target weight
        target_weight = weights.iloc[i].copy(deep=True)
        delta_weight = target_weight - curr_weight

        # Based on buffer decide if trade should be made
        do_trade = (delta_weight.abs() > trade_buffer) | (curr_weight == 0)
        do_trade[CASH] = False

        # Default is to trade to the ideal target weight when commission is a fixed minimum or zero
        # Trade to buffer when commission is a linear pct of trade value (e.g. crypto)
        # Always trade to target weight when opening new poistion (i.e. current weight is zero)
        adj_target_weight = target_weight.copy(deep=True)
        if do_limit_trade_size:
            adj_target_weight.loc[do_trade] += delta_weight.abs() - trade_buffer
            adj_target_weight.loc[curr_weight == 0] = target_weight.copy(deep=True)

        # If no trade indicated then set target weight to current weight
        adj_target_weight.loc[~do_trade] = curr_weight

        # Calc adjusted delta for final trade sizing
        adj_delta_weight = adj_target_weight - curr_weight

        # Calc trades required to achieve adjusted target weight
        trade_value = adj_delta_weight * risk_capital
        trade_size = trade_value / price

        # Calc commission for the traded tickers using the given commission func
        commission = trade_value.copy(deep=True)
        commission[do_trade] = [
            commission_func(x, y) for x, y in zip(trade_size, trade_value)
        ]

        # Calc post trade port positions
        # Account for changes to cash from trade activity
        end_port = start_port.copy(deep=True)
        end_port[do_trade] += trade_size[do_trade]
        end_port[CASH] -= trade_value.loc[do_trade].sum()
        end_port[CASH] -= commission.loc[do_trade].sum()
        port_df.iloc[i] = end_port

        # Append data for this time period to the result
        series = pd.concat(
            [
                price,
                start_port,
                exposure,
                curr_weight,
                target_weight,
                delta_weight,
                do_trade,
                adj_target_weight,
                adj_delta_weight,
                trade_value,
                trade_size,
                end_port,
            ],
            keys=[
                "price",
                "start_portfolio",
                "exposure",
                "current_weight",
                "target_weight",
                "delta_weight",
                "do_trade",
                "adj_target_weight",
                "adj_delta_weight",
                "trade_value",
                "trade_size",
                "end_portfolio",
            ],
            axis=1,
        )
        series["datetime"] = weights.index[i]
        series = series.set_index(["datetime", series.index])
        result_df = pd.concat([result_df, series])

    return result_df
