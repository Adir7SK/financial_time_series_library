import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
# import scikit_learn as sklearn
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single
from finance_collection.simple_calcs import date_interpreter

warnings.filterwarnings('ignore')


class FinanceVerticalIteration(Dataset):
    def __init__(self, args, root_path, flag='train', size=None, features='MS',
                 data_path='', target='target_returns', scale=True, timeenc=0,
                 freq='d', seasonal_patterns=None):
        """
        RECALL THAT WE SET SHUFFLE FLAG TO FALSE!!!
        """
        self.args = args
        self.args.start_date = date_interpreter(args.start_date)
        self.args.end_date = date_interpreter(args.end_date)

        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        ###
        self.last_20_indeces = []

        self.root_path = root_path
        self.tickers = []
        self.tickers_left = []
        self.data = dict()
        self.total_len = 0
        self.data_prev_len = 0
        self.the_ticker = None
        self.global_to_local = dict()
        self.__read_data__()
        self.__build_index_mapping__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        files = glob.glob(os.path.join(self.root_path, 'quandl_cpd_nonelbw_*.csv'))
        cols_data = None
        files = files[:self.args.limit_asset_number] \
            if self.args.limit_asset_number else files
        for file in files:
            df = pd.read_csv(file, parse_dates=['Date'])
            if not len(df):
                continue
            ticker = os.path.basename(file).split('_')[-1].split('.')[0]
            if cols_data is None:
                cols_data = [col for col in df.columns if col != 'Date' and col != self.target] + [self.target]
            df = df[(df['Date'] >= self.args.start_date) & (df['Date'] <= self.args.end_date)]
            df_data = df[cols_data]

            if len(df_data) <= self.seq_len + self.pred_len + self.label_len:
                continue

            self.tickers.append(ticker)
            self.data[ticker] = dict()

            num_train = int(len(df_data) * self.args.train_ratio)
            num_test = int(len(df_data) * self.args.test_ratio)
            num_vali = len(df_data) - num_train - num_test

            border1s = [0, num_train - self.seq_len, len(df_data) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_data)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.scale:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            df_stamp = df[['Date']][border1:border2]
            df_stamp['Date'] = pd.to_datetime(df_stamp.Date)

            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.Date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.Date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.Date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.Date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(['Date'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['Date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)

            data_x = data[border1:border2].copy()
            data_y = data[border1:border2].copy()

            # if self.set_type == 0 and self.args.augmentation_ratio > 0:
            #     data_x, data_y, augmentation_tags = run_augmentation_single(data_x, data_y,
            #                                                                           self.args)

            self.data[ticker]['data_x'] = np.array(data_x)
            self.data[ticker]['data_y'] = np.array(data_y)
            self.data[ticker]['data_stamp'] = np.array(data_stamp)
        self.keep_track_idx = -1
        self.idx_map = dict()

    def __build_index_mapping__(self):
        """Create mapping from global indices to specific (ticker, local_index) pairs"""
        global_idx = 0

        # Process tickers in the order they should be used
        for ticker in self.tickers:
            data_len = len(self.data[ticker]['data_x']) - self.seq_len - self.pred_len
            if data_len <= 0:
                continue  # Skip tickers with insufficient data

            # Map each global index to its corresponding ticker and local index
            for local_idx in range(data_len):
                self.global_to_local[global_idx] = (ticker, local_idx)
                global_idx += 1

        # Update total length based on actual valid indices
        self.total_len = global_idx

    def __getitem__(self, index):
        ticker, local_idx = self.global_to_local[index]
        if len(self.last_20_indeces) and index in self.last_20_indeces:
            print(f'Ticker {ticker} with local index {local_idx} already in last 20 indices!!!')
        if len(self.last_20_indeces) < 20:
            self.last_20_indeces.append(index)
        else:
            self.last_20_indeces.pop(0)
            self.last_20_indeces.append(index)

        # ticker = self.tickers_left[0]
        # i = index - self.data_prev_len
        # self.keep_track_idx += 1

        data_x = self.data[ticker]['data_x'].copy()
        data_y = self.data[ticker]['data_y'].copy()
        data_stamp = self.data[ticker]['data_stamp'].copy()

        s_begin = local_idx # i
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = np.array(data_x[s_begin:s_end])
        seq_y = np.array(data_y[r_begin:r_end])
        seq_x_mark = np.array(data_stamp[s_begin:s_end])
        seq_y_mark = np.array(data_stamp[r_begin:r_end])

        # assert s_end <= len(data_x), f"s_end {s_end} out of bounds for data_x with len {len(data_x)}"
        # assert r_end <= len(data_y), f"r_end {r_end} out of bounds for data_y with len {len(data_y)}"
        # assert r_end <= len(data_stamp), f"r_end {r_end} out of bounds for data_stamp with len {len(data_stamp)}"

        # print(f"IDX: {index}, Ticker: {ticker}, Local idx: {local_idx}")
        # print(f"seq_x shape: {seq_x.shape}, seq_y shape: {seq_y.shape}")

        # print(f"index: {index}, local index: {local_idx}, ticker: {ticker}, x range: {s_begin}-{s_end}, y range: {r_begin}-{r_end}, seq_x shape: {seq_x.shape}, seq_y shape: {seq_y.shape}, seq_x_mark shape: {seq_x_mark.shape}, seq_y_mark shape: {seq_y_mark.shape}")

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.total_len


class FinanceHorizontalIteration(Dataset):
    def __init__(self, args, root_path, flag='train', size=None, features='MS',
                 data_path='', target='target_returns', scale=True, timeenc=0,
                 freq='d', seasonal_patterns=None):
        self.args = args
        self.args.start_date = date_interpreter(args.start_date)
        self.args.end_date = date_interpreter(args.end_date)
        # if size is None:
        #    self.seq_len = 24 * 4 * 4
        #    self.label_len = 24 * 4
        #    self.pred_len = 24 * 4
        # else:
        #    self.seq_len = size[0]
        #    self.label_len = size[1]
        #    self.pred_len = size[2]

        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.ticker_dict_use = dict()
        self.tickers = []
        self.data = {}
        self.batch_seq = dict()
        self.total_len = 0
        # self.dates = set()
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        files = glob.glob(os.path.join(self.root_path, 'quandl_cpd_nonelbw_*.csv'))
        self.cols_data = None
        dates = set()
        for file in files:
            df = pd.read_csv(file, parse_dates=['Date'])
            if not len(df):
                continue
            ticker = os.path.basename(file).split('_')[-1].split('.')[0]
            if self.cols_data is None:
                self.cols_data = [col for col in df.columns if col != 'Date']
            df = df[(df['Date'] >= self.args.start_date) & (df['Date'] <= self.args.end_date)]
            df_data = df[self.cols_data]

            if len(df_data) <= self.seq_len + self.pred_len + self.label_len:
                continue

            self.ticker_dict_use[ticker] = True
            self.tickers.append(ticker)
            self.data[ticker] = dict()

            num_train = int(len(df_data) * 0.7)
            num_test = int(len(df_data) * 0.2)
            num_vali = len(df_data) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(df_data) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_data)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.scale:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            df_stamp = df[['Date']][border1:border2]
            df_stamp['Date'] = pd.to_datetime(df_stamp.Date)
            dates.update(df_stamp['Date'].values)
            self.data[ticker]['date_range'] = (df_stamp['Date'].min(), df_stamp['Date'].max())

            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.Date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.Date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.Date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.Date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(['Date'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['Date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)

            data_x = data[border1:border2]
            data_y = data[border1:border2]

            if self.set_type == 0 and self.args.augmentation_ratio > 0:
                data_x, data_y, augmentation_tags = run_augmentation_single(data_x, data_y,
                                                                                      self.args)

            self.data[ticker]['data_x'] = data_x
            self.data[ticker]['data_y'] = data_y
            self.data[ticker]['data_stamp'] = data_stamp

        dates = sorted(list(dates))
        for ticker in self.tickers: # Marking the index of the first and last date found for each asset
            self.data[ticker]['first_index'] = dates.index(self.data[ticker]['date_range'][0])
            self.data[ticker]['last_index'] = dates.index(self.data[ticker]['date_range'][1])

        self.first_idx = min([self.data[ticker]['first_index'] for ticker in self.tickers])
        if self.first_idx != 0:
            raise ValueError('The overall first index for dates inside the list of date should be the first date, and somehow it is not!! The fist index is:', self.first_idx)

        required_period = self.seq_len + self.pred_len + 1
        for i in range(len(dates)-self.pred_len):
            for ticker in self.tickers:
                if self.data[ticker]['first_index'] <= i and i + required_period < self.data[ticker]['last_index']:
                    self.batch_seq[self.total_len] = [i, ticker] # Should be i - self.data[ticker]['first_index'] because we are starting inside data_x from index 0
                    self.total_len += 1
            # df.set_index('date', inplace=True)
            # self.data[ticker] = df
            # self.dates.update(df.index)
        # if self.set_type == 0:
            #     for ticker in self.tickers:
            #        print(f"Ticker: {ticker}, Date Range: {self.data[ticker]['date_range']}, First Index: {self.data[ticker]['first_index']}, Last Index: {self.data[ticker]['last_index']}")
        # self.dates = sorted(self.dates)
        # print(self.tickers)
        # print(len(self.tickers))

    def __build_index_mapping__(self):
        """Create mapping from global indices to chronologically sorted (ticker, local_index) pairs"""
        # Store (date, ticker, local_index) tuples
        all_dates = []

        # First collect all valid dates with their corresponding ticker and local index
        for ticker in self.tickers:
            dates = pd.to_datetime(self.data[ticker]['data_stamp'][:, 0])  # Assuming dates are in first column
            data_len = len(dates) - self.seq_len - self.pred_len

            if data_len <= 0:
                continue  # Skip tickers with insufficient data

            for local_idx in range(data_len):
                # Get the date corresponding to this local index
                current_date = dates[local_idx]
                all_dates.append((current_date, ticker, local_idx))

        # Sort by date
        all_dates.sort(key=lambda x: x[0])

        # Build the mapping
        self.global_to_local = {}
        for global_idx, (date, ticker, local_idx) in enumerate(all_dates):
            self.global_to_local[global_idx] = (ticker, local_idx)

        # Update total length
        self.total_len = len(self.global_to_local)

    def __getitem__(self, index):
        '''
        SEE BELOW what is causeing the ERROR!!
        Must do debug which will throw the error in the for-loop at the training!
        RECALL FOR THE VERTICAL WE SET SHUFFLE FLAG TO FALSE!!
        '''
        i, ticker = self.batch_seq[index]

        data_x = self.data[ticker]['data_x']
        data_y = self.data[ticker]['data_y']
        data_stamp = self.data[ticker]['data_stamp']

        # print("SUCCESSFULLY LOADED TICKER: ", ticker)
        s_begin = i
        # s_begin = i - self.data[ticker]['first_index']
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data_x[s_begin:s_end]
        seq_y = data_y[r_begin:r_end]       # ERRORs HERE!! This becomes a Tensor of size [32, 21, 1] instead of [32, 21, 19] which causes a: RuntimeError: Trying to resize storage that is not resizable
        seq_x_mark = data_stamp[s_begin:s_end]
        seq_y_mark = data_stamp[r_begin:r_end]



        # if seq_x.shape[0] != self.seq_len or seq_y.shape[0] != self.label_len + self.pred_len:
    #     continue

        # satisfied_ticker = True

        # Validate shapes to ensure consistency
        # if seq_x.shape[0] != self.seq_len or seq_y.shape[0] != self.label_len + self.pred_len:
        #      print(f"VALUE_ERROR! Inconsistent shapes: seq_x shape {seq_x.shape}, seq_y shape {seq_y.shape}")

        print(f"index: {i}, ticker: {ticker}, x range: {s_begin}-{s_end}, y range: {r_begin}-{r_end}, seq_x shape: {seq_x.shape}, seq_y shape: {seq_y.shape}, seq_x_mark shape: {seq_x_mark.shape}, seq_y_mark shape: {seq_y_mark.shape}")

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # total_length = 0
        # for ticker in self.tickers:
        #     data_length = len(self.data[ticker]['data_x']) + 1 - self.seq_len - self.pred_len
        #     total_length += data_length # - self.seq_len + 1
        print("The total length is:", self.total_len)
        # return total_length
        return self.total_len


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, args, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern='*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)
