import numpy as np
import pandas as pd
import ffn

VOLA_EWMA_ALPHA = 1.0 - 0.94
TRADING_DAYS_YEAR = 252


def _pnl(result: pd.DataFrame) -> pd.DataFrame:
    df = result["exposure"].astype("float").groupby(["datetime"]).sum().to_frame()
    df = df.rename(columns={"exposure": "portfolio"})
    return df


def calc_stats(result: pd.DataFrame) -> pd.DataFrame:
    return ffn.calc_stats(_pnl(result)).stats


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
