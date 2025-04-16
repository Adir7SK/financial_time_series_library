import pandas as pd
import numpy as np
import os

from finance_collection.simple_calcs import calc_signal, calc_returns, calc_daily_vol, calc_vol_scaled_returns
from finance_collection.constants import HALFLIFE_WINSORISE, VOL_THRESHOLD, ASSET_DATA_QUALITY_THRESHOLD, MAX_NO_VOL_SEQ_LENGTH, trd_days_year, trd_days_month, trd_days_week, trd_days_day, trd_days_quarter, trd_days_biannual, short_ma_window, mid_ma_window, long_ma_window


def deep_momentum_strategy_features(df_asset: pd.DataFrame, quality_threshold=0.2) -> pd.DataFrame:
    """prepare input features for deep learning model

    Args:
        df_asset (pd.DataFrame): time-series for asset with column close

    Returns:
        pd.DataFrame: input features
    """

    df_asset = df_asset[
        ~df_asset["close"].isna()
        | ~df_asset["close"].isnull()
        | (df_asset["close"] > 1e-8)  # price is zero
        ].copy()

    # Handle consecutive rows with the same "close" value
    non_nan_rows = df_asset.dropna(subset=["close"])
    consecutive_counts = non_nan_rows["close"].groupby(
        (non_nan_rows["close"] != non_nan_rows["close"].shift()).cumsum()).transform('size')
    mask = (consecutive_counts > MAX_NO_VOL_SEQ_LENGTH) & (non_nan_rows["close"] == non_nan_rows["close"].shift(1))
    non_nan_rows.loc[mask, "close"] = np.nan

    # Check if more than 20% of non-NaN rows were changed to NaN
    if non_nan_rows["close"].isna().sum() / len(non_nan_rows) > ASSET_DATA_QUALITY_THRESHOLD:
        return pd.DataFrame(np.nan, index=df_asset.index, columns=df_asset.columns)

    df_asset.loc[mask, :] = np.nan
    df_asset = df_asset.loc[:df_asset.dropna(how='all').index[-1]]

    # winsorize using rolling 5X standard deviations to remove outliers. MEANING; remove extreme spikes in data by
    # keeping values between [mean - 5*std, mean + 5*std], where mean calculated as exponential moving average.
    df_asset["srs"] = df_asset["close"]
    ewm = df_asset["srs"].ewm(halflife=HALFLIFE_WINSORISE)
    means = ewm.mean()
    stds = ewm.std()
    df_asset["srs"] = np.minimum(df_asset["srs"], means + VOL_THRESHOLD * stds)
    df_asset["srs"] = np.maximum(df_asset["srs"], means - VOL_THRESHOLD * stds)

    df_asset["daily_returns"] = calc_returns(df_asset["srs"])
    df_asset["daily_vol"] = calc_daily_vol(df_asset["daily_returns"])
    # vol scaling and shift to be next day returns
    df_asset["target_returns"] = calc_vol_scaled_returns(
        df_asset["daily_returns"], df_asset["daily_vol"]
    ).shift(-1)

    def calc_normalised_returns(day_offset):
        return (
                calc_returns(df_asset["srs"], day_offset)
                / df_asset["daily_vol"]
                / np.sqrt(day_offset)
        )

    df_asset["norm_daily_return"] = calc_normalised_returns(trd_days_day)
    df_asset["norm_monthly_return"] = calc_normalised_returns(trd_days_month)
    df_asset["norm_quarterly_return"] = calc_normalised_returns(trd_days_quarter)
    df_asset["norm_biannual_return"] = calc_normalised_returns(trd_days_biannual)
    df_asset["norm_annual_return"] = calc_normalised_returns(trd_days_year)

    # Calculates Moving Average Convergence Divergence for three pairs of windows: 8/24 days, 16/48 days, 32/96 days
    trend_combinations = [short_ma_window, mid_ma_window, long_ma_window]
    for short_window, long_window in trend_combinations:
        df_asset[f"macd_{short_window}_{long_window}"] = calc_signal(
            df_asset["srs"], short_window, long_window
        )

    # date features
    if len(df_asset):
        df_asset["day_of_week"] = df_asset.index.dayofweek
        df_asset["day_of_month"] = df_asset.index.day
        df_asset["week_of_year"] = df_asset.index.isocalendar().week
        df_asset["month_of_year"] = df_asset.index.month
        df_asset["year"] = df_asset.index.year
        # df_asset["date"] = df_asset.index  # duplication but sometimes makes life easier
    else:
        df_asset["day_of_week"] = []
        df_asset["day_of_month"] = []
        df_asset["week_of_year"] = []
        df_asset["month_of_year"] = []
        df_asset["year"] = []
        df_asset["date"] = []

    return df_asset.dropna()


def read_changepoint_results_and_fill_na(
        file_path: str, lookback_window_length: int
) -> pd.DataFrame:
    """Read output data from changepoint detection module into a dataframe.
    For rows where the module failed, information for changepoint location and severity is
    filled using the previous row.


    Args:
        file_path (str): the file path of the csv containing the results
        lookback_window_length (int): lookback window length - necessary for filling in the blanks for norm location

    Returns:
        pd.DataFrame: changepoint severity and location information
    """

    return (
        pd.read_csv(file_path, index_col=0, parse_dates=True)
        .fillna(method="ffill")
        .dropna()  # if first values are na
        .assign(
            cp_location_norm=lambda row: (row["t"] - row["cp_location"])
                                         / lookback_window_length
        )  # fill by assigning the previous cp and score, then recalculate norm location
    )


def prepare_cpd_features(folder_path: str, lookback_window_length: int) -> pd.DataFrame:
    """Read output data from changepoint detection module for all assets into a dataframe.


    Args:
        file_path (str): the folder path containing csvs with the CPD the results
        lookback_window_length (int): lookback window length

    Returns:
        pd.DataFrame: changepoint severity and location information for all assets
    """

    return pd.concat(
        [
            read_changepoint_results_and_fill_na(
                os.path.join(folder_path, f), lookback_window_length
            ).assign(ticker=os.path.splitext(f)[0])
            for f in os.listdir(folder_path)
        ]
    )


def include_changepoint_features(
    features: pd.DataFrame, cpd_folder_name: pd.DataFrame, lookback_window_length: int
) -> pd.DataFrame:
    """combine CP features and DMN featuress

    Args:
        features (pd.DataFrame): features
        cpd_folder_name (pd.DataFrame): folder containing CPD results
        lookback_window_length (int): LBW used for the CPD

    Returns:
        pd.DataFrame: features including CPD score and location
    """
    features = features.merge(
        prepare_cpd_features(cpd_folder_name, lookback_window_length)[
            ["ticker", "cp_location_norm", "cp_score"]
        ]
        .rename(
            columns={
                "cp_location_norm": f"cp_rl_{lookback_window_length}",
                "cp_score": f"cp_score_{lookback_window_length}",
            }
        )
        .reset_index(),  # for date column
        on=["date", "ticker"],
    )

    features.index = features["date"]

    return features
