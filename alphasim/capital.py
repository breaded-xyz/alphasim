import numpy as np

def initial_capital(initial: float, total: float) -> float:
    return initial

def total_capital(initial: float, total: float) -> float:
    return total

def sqrt_profit_capital(initial: float, total: float) -> float:
    return initial * np.sqrt(1 + (total-initial)/initial)
