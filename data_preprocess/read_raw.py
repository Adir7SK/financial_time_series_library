import os
from typing import List, Tuple

import pandas as pd

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)


def pull_clc_asset(ticker: str) -> pd.DataFrame:
    "Open, High, Low, Settle, Open Interest, Total Volume"
    df = pd.read_csv(os.path.join(project_root, "dataset", "preprocessed", "CLCDATA", f"{ticker}.csv"), header=None,
                     parse_dates=[0])
    df.columns = ["date", "Open", "High", "Low", "Settle", "OpenInterest", "TotalVolume"]
    df = df[["date", "Settle"]]
    df = df.set_index("date").replace(0.0, np.nan)
    return df.rename(columns={"Trade Date": "date", "Date": "date", "Settle": "close"})


def pull_futures_close(remove_tickers):
    if type(remove_tickers) == str:
        remove_tickers = [remove_tickers]
    df = pd.read_csv(os.path.join(project_root, "data", "Finance", "pin-futs-close-1990-2025.csv"),
                     parse_dates=[0])
    df = df.set_index("date")
    return df.drop(remove_tickers, axis=1)


def _get_directory_name(experiment_name: str, train_interval: Tuple[int, int, int] = None) -> str:
    """The directory name for saving results
    Purpose: Creates a standardized directory path for saving experiment results

        Args:
            experiment_name (str): name of experiment
            train_interval (Tuple[int, int, int], optional): (start yr, end train yr / start test yr, end test year) Defaults to None.

        Returns:
            str: folder name
        """
    if train_interval is None:
        return os.path.join("results", experiment_name)
    return os.path.join("results", experiment_name, f"{train_interval[1]}-{train_interval[2]}")


########### FILES AND FILE PATHS ###########
CPD_QUANDL_OUTPUT_FOLDER = lambda lbw: os.path.join(
    f"quandl_cpd_{(lbw if lbw else 'none')}lbw"
)

FEATURES_QUANDL_FILE_PATH = lambda lbw: os.path.join(
    f"quandl_cpd_{(lbw if lbw else 'none')}lbw.csv"
)

ticker_file = lambda path, ticker: f"{path.rsplit('.csv', 1)[0]}_{ticker}.csv"

