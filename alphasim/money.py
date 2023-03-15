import math


def initial_capital(initial: float, total: float) -> float:
    """
    Money is initial stake only.
    No reinvestment of profits.
    """
    return initial


def total_equity(initial: float, total: float) -> float:
    """
    Money is reinvestment of all profits.
    """
    return total


def sqrt_profit(initial, total):
    """
    Money is a function of the sqrt of the capital growth rate.
    Partial reinvestment of profits guards against likelihood of
    increasingly severe drawdowns as equity grows.
    See https://zorro-project.com/manual/en/tutorial_kelly.htm
    """
    return initial * math.sqrt(1 + (total - initial) / initial)
