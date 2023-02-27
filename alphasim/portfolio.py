import numpy as np
import pandas as pd
from scipy.optimize import minimize


def distribute(weights: pd.Series, max_weight: float) -> np.ndarray:
    """
    Distribute weights constrained by a maximum individual weight.
    Given weights sould be positive numbers that sum to 1.
    Returned weights will sum to given weights.
    """

    objective = lambda x: -np.sum(x)

    constraints = [
        {"type": "ineq", "fun": lambda x: max_weight - np.amax(np.abs(x))},
        {"type": "eq", "fun": lambda x: np.sum(x) - np.sum(weights)},
    ]

    bounds = [(0, 1) for i in range(len(weights))]

    result = minimize(objective, weights, bounds=bounds, constraints=constraints)

    return result.x


def norm_signed(x: pd.Series) -> pd.Series:
    weights = x.abs()
    weights /= weights.sum()
    return np.copysign(weights, x)


def allocate(
    capital: float,
    port: pd.Series,
    weight: pd.Series,
    buffer: float = 0,
    ignore_buffer_on_new: bool = False,
    ignore_buffer_on_zero: bool = False,
    quote: str = "USD",
) -> tuple:

    start_weight = port / capital

    adj_target_weight = weight.copy()
    adj_target_weight[:] = [
        _buffered_target(x, y, buffer) for x, y in zip(weight, start_weight)
    ]

    if ignore_buffer_on_new:
        mask = weight.abs().gt(0) & start_weight.eq(0)
        mask[quote] = False
        adj_target_weight[mask] = weight

    if ignore_buffer_on_zero:
        mask = weight.eq(0) & start_weight.abs().gt(0)
        mask[quote] = False
        adj_target_weight[mask] = 0

    adj_delta_weight = adj_target_weight - start_weight

    trade_value = adj_delta_weight * capital

    return start_weight, weight, adj_target_weight, adj_delta_weight, trade_value


def _buffered_target(x, y, b) -> float:
    target = y

    if y < (x - b):
        target = x - b

    if y > (x + b):
        target = x + b

    return target
