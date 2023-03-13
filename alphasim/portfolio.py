from typing import cast

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def distribute(weights: pd.Series, max_weight: float) -> np.ndarray:
    """
    Distribute weights using a maximum individual weight constraint.
    Input weights sould be positive real numbers that sum to 1.
    Returned weights will sum to the given weights whilst obeying the
    maximum weight constraint. Excess is distributed proportional to the
    input weights.
    """

    def objective(x):
        return np.sum(np.square(x - weights))

    constraints = [
        {"type": "eq", "fun": lambda x: np.amax(x) - max_weight},
        {"type": "eq", "fun": lambda x: np.sum(x) - np.sum(weights)},
    ]

    bounds = [(0, 1) for i in range(len(weights))]

    result = minimize(objective, weights, bounds=bounds, constraints=constraints)

    return result.x


def to_weights(x: pd.Series) -> pd.Series:
    """
    Transform a continous signed forecast into
    weights with an absolute sum of 1.
    """
    weights = x.abs()
    weights /= weights.sum()
    return cast(pd.Series, np.copysign(weights, x))


def allocate(
    capital: float,
    price: pd.Series,
    marked_portfolio: pd.Series,
    target_weight: pd.Series,
    trade_buffer: float = 0,
    lot_size: pd.Series | None = None,
) -> tuple[pd.Series, ...]:
    """
    Allocate capital to a portfolio given a set of weights.
    Serves to descretize continous weights into lots (shares).
    Trade buffer is used to optimize allocation.
    Lot size (in quote units) can be set to support assets which
    allow partial buy/sell.
    """

    start_weight = marked_portfolio / capital

    adj_target_weight = target_weight.copy()
    adj_target_weight[:] = [
        _buffer_target(x, y, trade_buffer) for x, y in zip(target_weight, start_weight)
    ]

    adj_delta_weight = adj_target_weight - start_weight

    if lot_size is None:
        lot_size = price.copy()

    lots = _discretize(capital, adj_delta_weight, lot_size)

    quote_qty = lots * lot_size
    base_qty = quote_qty / price

    return (
        start_weight,
        target_weight,
        adj_target_weight,
        adj_delta_weight,
        base_qty,
        quote_qty,
    )


def _buffer_target(target: float, current: float, buffer: float) -> float:
    buffered = current

    if current < (target - buffer):
        buffered = target - buffer

    if current > (target + buffer):
        buffered = target + buffer

    return buffered


def _discretize(capital: float, weights: pd.Series, lot_sizes: pd.Series) -> pd.Series:

    budget = (weights * capital).round()

    rem = budget % lot_sizes

    lots = (budget - rem) / lot_sizes

    return lots
