import numpy as np


def initial_capital(initial, cash, total: float) -> float:
    return initial


def total_equity(initial, cash, total: float) -> float:
    return total


def sqrt_profit(initial, cash, total: float) -> float:
    profit = total - initial
    return initial * np.sqrt(1 + (profit / initial))
