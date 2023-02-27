import pandas as pd


def allocate(
    capital: float,
    port: pd.Series,
    target_weight: pd.Series,
    buffer: float,
    ignore_buffer_on_new: bool,
    ignore_buffer_on_zero: bool,
    quote: str,
) -> tuple:

    start_weight = port / capital

    adj_target_weight = target_weight.copy()
    adj_target_weight[:] = [
        _buffered_target(x, y, buffer) for x, y in zip(target_weight, start_weight)
    ]

    if ignore_buffer_on_new:
        mask = target_weight.abs().gt(0) & start_weight.eq(0)
        mask[quote] = False
        adj_target_weight[mask] = target_weight

    if ignore_buffer_on_zero:
        mask = target_weight.eq(0) & start_weight.abs().gt(0)
        mask[quote] = False
        adj_target_weight[mask] = 0

    adj_delta_weight = adj_target_weight - start_weight

    trade_value = adj_delta_weight * capital

    return start_weight, target_weight, adj_target_weight, adj_delta_weight, trade_value


def _buffered_target(x, y, b) -> float:
    target = y

    if y < (x - b):
        target = x - b

    if y > (x + b):
        target = x + b

    return target
