import numpy as np

def zero_commission(trade_size, trade_value) -> float:
    return 0

def fixed_commission(trade_size, trade_value, fixed_commission) -> float:
    return fixed_commission

def linear_pct_commission(trade_size, trade_value, pct_commission) -> float:
    commission = abs(trade_value) * pct_commission
    return commission

def tiered_pct_commission(trade_size, trade_value, min_fee_per_order, fee_per_unit, max_pct_per_order) -> float:
    commission = np.min([abs(trade_size) * fee_per_unit, abs(trade_value) * max_pct_per_order])
    commission = np.min([min_fee_per_order, commission])
    return commission