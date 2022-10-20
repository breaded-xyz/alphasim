import numpy as np
import pandas as pd

import alphasim.backtest as bt

VOLA_EWMA_ALPHA = 1.0 - 0.94
TRADING_DAYS_YEAR = 252

def _pnl(result: pd.DataFrame) -> pd.DataFrame:
    df = result[bt.EQUITY].astype("float").groupby(level=0).sum().to_frame()
    return df


def calc_stats(result: pd.DataFrame) -> pd.DataFrame:
    
    sum_df = result.groupby(level=0).sum()
    days = len(sum_df)
    years = days/TRADING_DAYS_YEAR
    ret_df = sum_df[bt.EQUITY].pct_change()

    df = pd.DataFrame(index=["result"])
    df["start"] = sum_df.index.values[0]
    df["end"] = sum_df.index.values[-1]
    df["initial"] = sum_df[bt.EQUITY].iloc[0]
    df["final"] = sum_df[bt.EQUITY].iloc[-1]
    df["profit"] = df["final"] - df["initial"]
    df["cagr"] = (df["final"] / df["initial"]) ** (1/years) - 1
    df["ann_volatility"] = ret_df.std() * np.sqrt(TRADING_DAYS_YEAR)
    df["ann_sharpe"] = df["cagr"] / df["ann_volatility"]
    df["commission"] = sum_df["commission"].sum()
    df["funding"] = sum_df["funding"].sum()
    df["cost_profit_pct"] = (df["commission"] + df["funding"]) / df["profit"]
    df["trade_count"] = result["do_trade"].sum()

    return df.T

def calc_log_returns(result: pd.DataFrame) -> pd.DataFrame:
    pnl_df = _pnl(result)
    ret_df = np.log(pnl_df / pnl_df.shift(1)).dropna()
    return ret_df

def calc_rolling_ann_vola(result: pd.DataFrame) -> pd.DataFrame:
    ret_df = calc_log_returns(result)
    vola_df = ret_df.ewm(alpha=VOLA_EWMA_ALPHA, adjust=False).std() * np.sqrt(TRADING_DAYS_YEAR)
    return vola_df
