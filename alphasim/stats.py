import numpy as np
import pandas as pd

import alphasim.backtest as bt
import alphasim.const as const


def backtest_stats(
    result: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
    freq: int = 1,
    freq_unit: str = "D",
    trading_days_year: int = const.TRADING_DAYS_YEAR,
) -> pd.DataFrame | None:

    if result is None or len(result) == 0:
        raise ValueError("result must not be None or empty")

    summary = result.groupby(level=0).sum()
    start = summary.index[0]
    end = summary.index[-1]
    days = (end - start).days

    cal_years = days / const.CALENDAR_DAYS_YEAR

    ret = backtest_returns(result)
    ret_per_day = pd.Timedelta(1, unit="D") / pd.Timedelta(freq, unit=freq_unit)

    initial = summary[bt.EQUITY].iloc[0]
    final = summary[bt.EQUITY].iloc[-1]
    cagr = (final / initial) ** (1 / cal_years) - 1
    vol = ret.std() * np.sqrt(ret_per_day * trading_days_year)
    sr = cagr / vol

    df = pd.DataFrame(index=["result"])
    df["start"] = start
    df["end"] = end
    df["trading_days_year"] = trading_days_year
    df["price_freq"] = f"{freq}{freq_unit}"
    df["risk_free_rate"] = const.RISK_FREE_RATE
    df["initial"] = initial
    df["final"] = final
    df["profit"] = final - initial
    df["cagr"] = cagr
    df["ann_volatility"] = vol
    df["ann_sharpe"] = sr
    df["kelly_f"] = cagr / (vol**2)
    df["kelly_f_cagr"] = (sr**2) / 2
    df["commission"] = summary["commission"].sum()
    df["funding_payment"] = summary["funding_payment"].sum()
    df["cost_profit_pct"] = (df["commission"] + df["funding_payment"]) / df["profit"]
    df["trade_count"] = result["is_trade"].sum()
    df["skew"] = ret.skew()
    df["kurtosis"] = ret.kurtosis()

    # Max drawdown
    cum_ret = (1 + ret).cumprod() - 1
    nav = ((1 + cum_ret) * 100).fillna(100)
    hwm = nav.cummax()
    dd = nav / hwm - 1
    df["max_drawdown"] = dd.min()

    # Turnover
    mean_equity = summary[bt.EQUITY].mean()
    buy_value = result["quote_qty"].loc[result["base_qty"] > 0].abs().sum()
    sell_value = result["quote_qty"].loc[result["base_qty"] < 0].abs().sum()
    tx_value = np.min([buy_value, sell_value])
    turnover = tx_value / mean_equity
    df["ann_turnover"] = turnover / cal_years

    if benchmark is not None:
        benchmark_stats = _asset_stats(
            benchmark,
            initial=initial,
            freq=freq,
            freq_unit=freq_unit,
            trading_days_year=trading_days_year,
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
    prices: pd.DataFrame,
    initial: float = 1000,
    freq: int = 1,
    freq_unit: str = "D",
    trading_days_year: int = const.TRADING_DAYS_YEAR,
) -> pd.DataFrame:

    start = prices.index[0]
    end = prices.index[-1]
    days = (end - start).days

    cal_years = days / trading_days_year

    ret = prices.pct_change()
    ret_per_day = pd.Timedelta(1, unit="D") / pd.Timedelta(freq, unit=freq_unit)

    port_units = initial / prices.iloc[0].squeeze()

    df = pd.DataFrame(index=[prices.columns[0]])
    df["start"] = start
    df["end"] = end
    df["trading_days_year"] = trading_days_year
    df["price_freq"] = f"{freq}{freq_unit}"
    df["risk_free_rate"] = const.RISK_FREE_RATE
    df["initial"] = initial
    df["final"] = prices.iloc[-1].squeeze() * port_units
    df["profit"] = df["final"] - df["initial"]
    df["cagr"] = (df["final"] / df["initial"]) ** (1 / cal_years) - 1
    df["ann_volatility"] = ret.std() * np.sqrt(ret_per_day * trading_days_year)
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
