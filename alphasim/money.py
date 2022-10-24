import numpy as np


def initial_capital(
    initial: float = 0, open_equity: float = 0, closed_equity: float = 0
) -> float:
    return initial


def total_equity(
    initial: float = 0, open_equity: float = 0, closed_equity: float = 0
) -> float:
    return open_equity + closed_equity


def sqrt_profit(
    initial: float = 0, open_equity: float = 0, closed_equity: float = 0
) -> float:
    profit = (open_equity + closed_equity) - initial
    return initial * np.sqrt(1 + (profit / initial))
