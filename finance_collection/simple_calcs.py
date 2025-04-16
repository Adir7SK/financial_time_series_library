import pandas as pd
import numpy as np
import os
from empyrical import (
    sharpe_ratio,
    calmar_ratio,
    sortino_ratio,
    max_drawdown,
    downside_risk,
    annual_return,
    annual_volatility,
)
from data_preprocess.read_raw import _get_directory_name
from typing import Tuple, Dict, List
from finance_collection.constants import VOL_TARGET, VOL_LOOKBACK


def calc_signal(srs: pd.Series, short_timescale: int, long_timescale: int) -> float:
    """Calculate MACD signal for a signal short/long timescale combination

    Args:
        srs ([type]): series of prices
        short_timescale ([type]): short timescale
        long_timescale ([type]): long timescale

    Returns:
        float: MACD signal
    """

    def _calc_halflife(timescale):
        return np.log(0.5) / np.log(1 - 1 / timescale)

    macd = (
            srs.ewm(halflife=_calc_halflife(short_timescale)).mean()
            - srs.ewm(halflife=_calc_halflife(long_timescale)).mean()
    )
    q = macd / srs.rolling(63).std().ffill()  # fillna(method="bfill")
    t = q / q.rolling(252).std().ffill()  # fillna(method="bfill")
    if np.isinf(t).any():
        raise ValueError("MACD signal contains Infinite values")
    return t


def sharpe_ratio(returns, risk_free=0.0, periods=252):
    """
    Calculate the annualized Sharpe ratio. This is used for test and validation.

    Args:
        returns: Array-like of returns
        risk_free: Risk-free rate (default 0)
        periods: Number of periods in a year (252 for daily raw_data_prep_before_feature_construction, assuming trading days)

    Returns:
        float: Annualized Sharpe ratio
    """
    returns = np.array(returns)
    # print("The first 10 returns are: ", returns[:10])
    # print("The shape of returns is: ", returns.shape)
    excess_returns = returns - risk_free
    # print("The first 10 excess_returns are: ", excess_returns[:10])
    # print("The shape of excess_returns is: ", excess_returns.shape)
    if len(excess_returns) < 2:
        raise ValueError(f"The length of returns is: {len(returns)} causing the Sharpe Ratio to be NaN")
        return np.nan

    mean_excess_return = np.mean(excess_returns)
    mean_excess_return = np.nanmean(excess_returns)
    std_excess_return = np.std(excess_returns, ddof=1)
    num_nans = np.isnan(excess_returns).sum()
    std_excess_return = np.nanstd(excess_returns, ddof=1)

    if std_excess_return == 0 or np.isnan(std_excess_return):
        # print("The first 10 returns are: ", returns[:10])
        # print("The first 10 excess returns are: ", excess_returns[:10])
        # print("The number of NaNs in excess returns is: ", num_nans)
        # print("The standard deviation without NaNs is: ", std_excess_return)
        raise ValueError("The standard deviation of excess returns is zero or NaN")

    if std_excess_return == 0:
        return np.nan

    sharpe = mean_excess_return / std_excess_return
    # print("You reached the end of the calculation, the mean is: ", mean_excess_return)
    # print("You reached the end of the calculation, the std is: ", std_excess_return)
    # print("You reached the end of the calculation, the Sharpe ratio is: ", sharpe)
    return np.sqrt(periods) * sharpe

def calc_returns(srs: pd.Series, day_offset: int = 1) -> pd.Series:
    """for each element of a pandas time-series srs,
    calculates the returns over the past number of days
    specified by offset

    Args:
        srs (pd.Series): time-series of prices
        day_offset (int, optional): number of days to calculate returns over. Defaults to 1.

    Returns:
        pd.Series: series of returns
    """
    returns = srs / srs.shift(day_offset) - 1.0
    return returns


def calc_daily_vol(daily_returns):
    return (
        daily_returns.ewm(span=VOL_LOOKBACK, min_periods=VOL_LOOKBACK)
        .std()
        .ffill()
        # .fillna(method="bfill")
    )


def calc_vol_scaled_returns(daily_returns, daily_vol=pd.Series(dtype=float)):
    """calculates volatility scaled returns for annualised VOL_TARGET of 15%
    with input of pandas series daily_returns"""
    if not len(daily_vol):
        daily_vol = calc_daily_vol(daily_returns)
    annualised_vol = daily_vol * np.sqrt(252)  # annualised
    return daily_returns * VOL_TARGET / annualised_vol.shift(1)


def calc_performance_metrics(data: pd.DataFrame, metric_suffix="", num_identifiers = None) -> dict:
    """Performance metrics for evaluating strategy

    Args:
        captured_returns (pd.DataFrame): dataframe containing captured returns, indexed by date

    Returns:
        dict: dictionary of performance metrics
    """
    if not num_identifiers:
        num_identifiers = len(data.dropna()["identifier"].unique())
    srs = data.dropna().groupby(level=0)["captured_returns"].sum()/num_identifiers
    return {
        f"annual_return{metric_suffix}": annual_return(srs),
        f"annual_volatility{metric_suffix}": annual_volatility(srs),
        f"sharpe_ratio{metric_suffix}": sharpe_ratio(srs),
        f"downside_risk{metric_suffix}": downside_risk(srs),
        f"sortino_ratio{metric_suffix}": sortino_ratio(srs),
        f"max_drawdown{metric_suffix}": -max_drawdown(srs),
        f"calmar_ratio{metric_suffix}": calmar_ratio(srs),
        f"perc_pos_return{metric_suffix}": len(srs[srs > 0.0]) / len(srs),
        f"profit_loss_ratio{metric_suffix}": np.mean(srs[srs > 0.0])
        / np.mean(np.abs(srs[srs < 0.0])),
    }


def calc_sharpe_by_year(data: pd.DataFrame, suffix: str = None) -> dict:
    """Sharpe ratio for each year in dataframe

    Args:
        data (pd.DataFrame): dataframe containing captured returns, indexed by date

    Returns:
        dict: dictionary of Sharpe by year
    """
    if not suffix:
        suffix = ""

    data = data.copy()
    data.index = pd.to_datetime(data.index)
    data["year"] = data.index.year

    # mean of year is year for same date
    sharpes = (
        data.dropna()[["year", "captured_returns"]]
        .groupby(level=0)
        .mean()
        .groupby("year")
        .apply(lambda y: sharpe_ratio(y["captured_returns"]))
    )

    sharpes.index = "sharpe_ratio_" + sharpes.index.map(int).map(str) + suffix

    return sharpes.to_dict()


def calc_performance_metrics_subset(srs: pd.Series, metric_suffix="") -> dict:
    """Performance metrics for evaluating strategy

    Args:
        captured_returns (pd.Series): series containing captured returns, aggregated by date

    Returns:
        dict: dictionary of performance metrics
    """
    return {
        f"annual_return{metric_suffix}": annual_return(srs),
        f"annual_volatility{metric_suffix}": annual_volatility(srs),
        f"downside_risk{metric_suffix}": downside_risk(srs),
        f"max_drawdown{metric_suffix}": -max_drawdown(srs),
    }


def _captured_returns_from_all_windows(
    experiment_name: str,
    train_intervals: List[Tuple[int, int, int]],
    volatility_rescaling: bool = True,
    only_standard_windows: bool = True,
    volatilites_known: List[float] = None,
    filter_identifiers: List[str] = None,
    captured_returns_col: str = "captured_returns",
    standard_window_size: int = 1,
) -> pd.Series:
    """get sereis of captured returns from all intervals
    Purpose: Collects and processes returns raw_data_prep_before_feature_construction across multiple training windows

    Args:
        experiment_name (str): name of experiment
        train_intervals (List[Tuple[int, int, int]]): list of training intervals
        volatility_rescaling (bool, optional): rescale to target annualised volatility. Defaults to True.
        only_standard_windows (bool, optional): only include full windows. Defaults to True.
        volatilites_known (List[float], optional): list of annualised volatities, if known. Defaults to None.
        filter_identifiers (List[str], optional): only run for specified tickers. Defaults to None.
        captured_returns_col (str, optional): column name of captured returns. Defaults to "captured_returns".
        standard_window_size (int, optional): number of years in standard window. Defaults to 1.

    Returns:
        pd.Series: series of captured returns
    """
    srs_list = []
    volatilites = volatilites_known if volatilites_known else []
    for interval in train_intervals:
        if only_standard_windows and (
            interval[2] - interval[1] == standard_window_size
        ):
            df = pd.read_csv(
                os.path.join(
                    _get_directory_name(experiment_name, interval),
                    "captured_returns_sw.csv",
                ),
            )

            if filter_identifiers:
                filter = pd.DataFrame({"identifier": filter_identifiers})
                df = df.merge(filter, on="identifier")
            num_identifiers = len(df["identifier"].unique())
            srs = df.groupby("time")[captured_returns_col].sum() / num_identifiers
            srs_list.append(srs)
            if volatility_rescaling and not volatilites_known:
                volatilites.append(annual_volatility(srs))
    if volatility_rescaling:
        return pd.concat(srs_list) * VOL_TARGET / np.mean(volatilites)
    else:
        return pd.concat(srs_list)


def date_interpreter(date: str) -> pd.Timestamp:
    """Converts date string to datetime object

    Args:
        date (str): date string

    Returns:
        pd.Timestamp: datetime object
    """
    try:
        _ = int(date)
        date = date + '-01-01'
    except TypeError:
        pass
    return pd.to_datetime(date, format="%Y-%m-%d", errors='coerce')
