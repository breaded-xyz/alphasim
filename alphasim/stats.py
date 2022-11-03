import numpy as np
import pandas as pd

import alphasim.backtest as bt

VOLA_EWMA_ALPHA = 1.0 - 0.94
TRADING_DAYS_YEAR = 252


def calc_stats(result: pd.DataFrame) -> pd.DataFrame:

    sum_df = result.groupby(level=0).sum()
    years = len(sum_df) / TRADING_DAYS_YEAR
    ret_df = sum_df[bt.EQUITY].pct_change()

    df = pd.DataFrame(index=["result"])
    df["start"] = sum_df.index.values[0]
    df["end"] = sum_df.index.values[-1]
    df["initial"] = sum_df[bt.EQUITY].iloc[0]
    df["final"] = sum_df[bt.EQUITY].iloc[-1]
    df["profit"] = df["final"] - df["initial"]
    df["cagr"] = (df["final"] / df["initial"]) ** (1 / years) - 1
    df["ann_volatility"] = ret_df.std() * np.sqrt(TRADING_DAYS_YEAR)
    df["ann_sharpe"] = df["cagr"] / df["ann_volatility"]
    df["commission"] = sum_df["commission"].sum()
    df["funding_payment"] = sum_df["funding_payment"].sum()
    df["cost_profit_pct"] = (df["commission"] + df["funding_payment"]) / df["profit"]
    df["trade_count"] = result["do_trade"].sum()
    df["skew"] = ret_df.skew()

    ann_mean_equity = sum_df[bt.EQUITY].mean().squeeze() * years
    buy_value = result["trade_value"].loc[result["trade_size"] > 0].abs().sum()
    sell_value = result["trade_value"].loc[result["trade_size"] < 0].abs().sum()
    tx_value = np.min([buy_value, sell_value])
    df["ann_turnover"] = tx_value / ann_mean_equity

    return df.T


def calc_pnl(result: pd.DataFrame) -> pd.DataFrame:
    return _rollup_equity(result)


def calc_log_returns(result: pd.DataFrame) -> pd.DataFrame:
    pnl = _rollup_equity(result)
    return np.log(pnl / pnl.shift(1))


def calc_rolling_ann_vola(result: pd.DataFrame) -> pd.DataFrame:
    pnl = _rollup_equity(result)
    pnl = pnl.pct_change()
    return pnl.ewm(alpha=VOLA_EWMA_ALPHA).std() * np.sqrt(TRADING_DAYS_YEAR)


def _rollup_equity(result: pd.DataFrame) -> pd.DataFrame:
    df = result[bt.EQUITY].astype(np.float64).groupby(level=0).sum().to_frame()
    return df
