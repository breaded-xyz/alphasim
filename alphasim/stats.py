import numpy as np
import pandas as pd
import ffn

import alphasim.backtest as bt

VOLA_EWMA_ALPHA = 1.0 - 0.94
TRADING_DAYS_YEAR = 252


def _pnl(result: pd.DataFrame) -> pd.DataFrame:
    df = result[bt.EQUITY].astype("float").groupby(level=0).sum().to_frame()
    return df


def calc_stats(result: pd.DataFrame) -> pd.DataFrame:
    df = ffn.calc_stats(_pnl(result)).stats.T
    grouped_result_df = result.groupby(level=0).sum()
    df["initial"] = grouped_result_df[bt.EQUITY][0]
    df["final"] = grouped_result_df[bt.EQUITY][-1]
    df["profit"] = df["final"] - df["initial"]
    df["commission"] = grouped_result_df["commission"].sum()
    df["cost_profit_pct"] = df["commission"] / df["profit"]
    df["trade_count"] = result.loc[result["do_trade"] == True].count()
    return df.T


def calc_log_returns(result: pd.DataFrame) -> pd.DataFrame:
    pnl_df = _pnl(result)
    ret_df = np.log(pnl_df / pnl_df.shift(1)).dropna()
    return ret_df


def calc_rolling_ann_vola(result: pd.DataFrame) -> pd.DataFrame:
    ret_df = calc_log_returns(result)
    vola_df = ret_df.ewm(alpha=VOLA_EWMA_ALPHA, adjust=False).std() * np.sqrt(
        TRADING_DAYS_YEAR
    )
    return vola_df
