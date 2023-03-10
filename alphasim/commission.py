import numpy as np


def zero_commission(trade_size: float, trade_value: float) -> float:
    return 0


def fixed_commission(
    trade_size: float, trade_value: float, fixed_commission: float
) -> float:
    return -fixed_commission


def linear_pct_commission(
    trade_size: float, trade_value: float, pct_commission: float
) -> float:
    commission = abs(trade_value) * pct_commission
    return -commission


def tiered_pct_commission(
    trade_size: float,
    trade_value: float,
    min_fee_per_order: float,
    fee_per_unit: float,
    max_pct_per_order: float,
) -> float:
    commission = np.min(
        [abs(trade_size) * fee_per_unit, abs(trade_value) * max_pct_per_order]
    )
    commission = np.min([min_fee_per_order, commission])
    return -commission
