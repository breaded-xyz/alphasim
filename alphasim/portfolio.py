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
        {"type": "ineq", "fun": lambda x: max_weight - np.amax(np.abs(x))},
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
    return np.copysign(weights, x)


def allocate(
    capital: float,
    price: pd.Series,
    marked_portfolio: pd.Series,
    target_weight: pd.Series,
    trade_buffer: float = 0,
) -> tuple:

    start_weight = marked_portfolio / capital

    adj_target_weight = target_weight.copy()
    adj_target_weight[:] = [
        _buffered_target(x, y, trade_buffer) 
        for x, y in zip(target_weight, start_weight)
    ]

    adj_delta_weight = adj_target_weight - start_weight

    trade_value = adj_delta_weight * capital

    trade_size = trade_value / price

    return (
        start_weight, target_weight,
        adj_target_weight, adj_delta_weight, 
        trade_size, trade_value
    )


def _buffered_target(x, y, b) -> float:
    target = y

    if y < (x - b):
        target = x - b

    if y > (x + b):
        target = x + b

    return target
