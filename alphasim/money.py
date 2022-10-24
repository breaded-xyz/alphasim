import numpy as np


def initial_capital(initial: float = 0, total: float = 0) -> float:
    return initial


def total_equity(initial: float = 0, total: float = 0) -> float:
    return total


def sqrt_profit(initial: float = 0, total: float = 0) -> float:
    profit = total - initial
    return initial * np.sqrt(1 + (profit / initial))
