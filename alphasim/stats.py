import numpy as np
import pandas as pd

import alphasim.const as const
import alphasim.backtest as bt


def backtest_stats(
    result: pd.DataFrame,
    benchmark: pd.DataFrame = None,
    freq: int = 1,
    freq_unit: str = "D",
) -> pd.DataFrame:

    summed = result.groupby(level=0).sum()
    start = summed.index[0]
    end = summed.index[-1]
    days = (end - start).days
    years = days / const.TRADING_DAYS_YEAR

    ret = backtest_returns(result)
    ret_per_day = pd.Timedelta(1, unit="D") / pd.Timedelta(freq, unit=freq_unit)

    df = pd.DataFrame(index=["result"])
    df["start"] = start
    df["end"] = end
    df["trading_days_year"] = const.TRADING_DAYS_YEAR
    df["initial"] = summed[bt.EQUITY].iloc[0]
    df["final"] = summed[bt.EQUITY].iloc[-1]
    df["profit"] = df["final"] - df["initial"]
    df["cagr"] = (df["final"] / df["initial"]) ** (1 / years) - 1
    df["ann_volatility"] = ret.std() * np.sqrt(ret_per_day * const.TRADING_DAYS_YEAR)
    df["ann_sharpe"] = df["cagr"] / df["ann_volatility"]
    df["commission"] = summed["commission"].sum()
    df["funding_payment"] = summed["funding_payment"].sum()
    df["cost_profit_pct"] = (df["commission"] + df["funding_payment"]) / df["profit"]
    df["trade_count"] = result["do_trade"].sum()
    df["skew"] = ret.skew()
    df["kurtosis"] = ret.kurtosis()

    # Max drawdown
    cum_ret = (1 + ret).cumprod() - 1
    nav = ((1 + cum_ret) * 100).fillna(100)
    hwm = nav.cummax()
    dd = nav / hwm - 1
    df["max_drawdown"] = dd.min()


    # Turnover
    ann_mean_equity = summed[bt.EQUITY].mean().squeeze() * years
    buy_value = result["trade_value"].loc[result["trade_size"] > 0].abs().sum()
    sell_value = result["trade_value"].loc[result["trade_size"] < 0].abs().sum()
    tx_value = np.min([buy_value, sell_value])
    df["ann_turnover"] = tx_value / ann_mean_equity

    if benchmark is not None:
        benchmark_stats = _asset_stats(
            benchmark, initial=df["initial"].squeeze(), freq=freq, freq_unit=freq_unit
        )
        df = pd.concat(
            [benchmark_stats, df], join="outer", keys=["benchmark", "backtest"]
        ).droplevel(1)

    return df.T


def backtest_returns(result: pd.DataFrame) -> pd.DataFrame:
    return result[bt.EQUITY].astype(np.float64).groupby(level=0).sum().pct_change()


def backtest_log_returns(result: pd.DataFrame) -> pd.DataFrame:
    equity = result[bt.EQUITY].astype(np.float64).groupby(level=0).sum()
    return (equity / equity.shift(1)).apply(np.log)


def _asset_stats(
    prices: pd.DataFrame, initial: float = 1000, freq: int = 1, freq_unit: str = "D"
) -> pd.DataFrame:

    start = prices.index[0]
    end = prices.index[-1]
    days = (end - start).days
    years = days / const.TRADING_DAYS_YEAR

    ret = prices.pct_change()
    ret_per_day = pd.Timedelta(1, unit="D") / pd.Timedelta(freq, unit=freq_unit)

    port_units = initial / prices.iloc[0].squeeze()

    df = pd.DataFrame(index=[prices.columns[0]])
    df["start"] = start
    df["end"] = end
    df["trading_days_year"] = const.TRADING_DAYS_YEAR
    df["initial"] = initial
    df["final"] = prices.iloc[-1].squeeze() * port_units
    df["profit"] = df["final"] - df["initial"]
    df["cagr"] = (df["final"] / df["initial"]) ** (1 / years) - 1
    df["ann_volatility"] = ret.std() * np.sqrt(ret_per_day * const.TRADING_DAYS_YEAR)
    df["ann_sharpe"] = df["cagr"] / df["ann_volatility"]
    df["trade_count"] = 1
    df["skew"] = ret.skew()
    df["kurtosis"] = ret.kurtosis()

    cum_ret = (1 + ret).cumprod() - 1
    nav = ((1 + cum_ret) * 100).fillna(100)
    hwm = nav.cummax()
    dd = nav / hwm - 1
    df["max_drawdown"] = dd.min()

    return df
