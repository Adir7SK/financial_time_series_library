import argparse
from typing import List

import pandas as pd
import os

from data_preprocess.read_raw import pull_clc_asset, pull_futures_close
from data_preprocess.read_raw import CPD_QUANDL_OUTPUT_FOLDER, FEATURES_QUANDL_FILE_PATH, ticker_file
from data_preprocess.only_close_price_data_prep import deep_momentum_strategy_features, include_changepoint_features


def main(
    # tickers: List[str],
    cpd_module_folder: str,
    lookback_window_length: int,
    output_file_path: str,
    extra_lbw: List[int],
):
    all_close_price_data = pull_futures_close("EC")
    tickers = all_close_price_data.columns.tolist()
    features = {ticker:
            deep_momentum_strategy_features(
                all_close_price_data[[ticker]].rename(columns={ticker: "close"}))
            .dropna().assign(ticker=ticker)
            for ticker in tickers
    }

    ticker_category = {ticker: i for i, ticker in enumerate(tickers)}

    for ticker, feature in features.items():
        feature.date = feature.index
        feature.index.name = "Date"
        feature['ticker'] = ticker_category[ticker]

        if lookback_window_length:
            features_w_cpd = include_changepoint_features(
                feature, cpd_module_folder, lookback_window_length
            )

            if extra_lbw:
                for extra in extra_lbw:
                    extra_data = pd.read_csv(
                        output_file_path.replace(
                            FEATURES_QUANDL_FILE_PATH(lookback_window_length),
                            FEATURES_QUANDL_FILE_PATH(extra),
                        ),
                        index_col=0,
                        parse_dates=True,
                    ).reset_index()[
                        ["Date", "ticker", f"cp_rl_{extra}", f"cp_score_{extra}"]
                    ]
                    extra_data["Date"] = pd.to_datetime(extra_data["Date"])

                    features_w_cpd = pd.merge(
                        features_w_cpd.set_index(["Date", "ticker"]),
                        extra_data.set_index(["Date", "ticker"]),
                        left_index=True,
                        right_index=True,
                    ).reset_index()
                    features_w_cpd.index = features_w_cpd["Date"]
                    features_w_cpd.index.name = "Date"
            else:
                features_w_cpd.index.name = "Date"
            features_w_cpd.to_csv(ticker_file(output_file_path, ticker))
        else:
            feature.to_csv(ticker_file(output_file_path, ticker))


if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""

        parser = argparse.ArgumentParser(description="Run changepoint detection module")
        # parser.add_argument(
        #     "cpd_module_folder",
        #     metavar="c",
        #     type=str,
        #     nargs="?",
        #     default=CPD_QUANDL_OUTPUT_FOLDER_DEFAULT,
        #     # choices=[],
        #     help="Input folder for CPD outputs.",
        # )
        parser.add_argument(
            "lookback_window_length",
            metavar="l",
            type=int,
            nargs="?",
            default=None,
            # choices=[],
            help="Input folder for CPD outputs.",
        )
        # parser.add_argument(
        #     "output_file_path",
        #     metavar="f",
        #     type=str,
        #     nargs="?",
        #     default=FEATURES_QUANDL_FILE_PATH_DEFAULT,
        #     # choices=[],
        #     help="Output file location for csv.",
        # )
        parser.add_argument(
            "extra_lbw",
            metavar="-e",
            type=int,
            nargs="*",
            default=[],
            # choices=[],
            help="Fill missing prices.",
        )

        args = parser.parse_known_args()[0]
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return (
            # COMMOD_FUTURES_TICKERS,
            CPD_QUANDL_OUTPUT_FOLDER(args.lookback_window_length),
            args.lookback_window_length,
            os.path.join(project_root, "dataset", "FinanceStrategiesFutures", FEATURES_QUANDL_FILE_PATH(args.lookback_window_length)),
            args.extra_lbw,
        )
    main(*get_args())
