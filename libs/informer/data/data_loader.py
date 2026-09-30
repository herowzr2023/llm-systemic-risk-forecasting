import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler, MinMaxScaler
from utils.timefeatures import time_features

import warnings

warnings.filterwarnings('ignore')

scaler_dic = {'std': StandardScaler, 'minmax': MinMaxScaler, }


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv', target='OT', scale=True,
                 inverse=False, timeenc=0, freq='h', cols=None, scaler_type="std"):
        # size [seq_len, label_len, pred_len]
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
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.scaler_type = scaler_type

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = scaler_dic[self.scaler_type]()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

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
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTm1.csv', target='OT', scale=True,
                 inverse=False, timeenc=0, freq='t', cols=None, scaler_type="std"):
        # size [seq_len, label_len, pred_len]
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
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.scaler_type = scaler_type

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = scaler_dic[self.scaler_type]()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

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
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv', target='OT', scale=True,
                 inverse=False, timeenc=0, freq='h', cols=None, scaler_type="std", is_class=False):
        # size [seq_len, label_len, pred_len]
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
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.scaler_type = scaler_type
        self.is_class = is_class
        self.__read_data__()

    def __read_data__(self):
        self.scaler = scaler_dic[self.scaler_type]()
        self.scaler_y = scaler_dic[self.scaler_type]()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        # 获取除 'date' 外的所有列名
        num_cols = [col for col in df_raw.columns if col != 'date']

        # 对这些列统一执行 to_numeric
        df_raw[num_cols] = df_raw[num_cols].apply(pd.to_numeric, errors='coerce')
        df_raw.ffill(inplace=True)

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns);
            cols.remove(self.target);
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

            train_data_x = train_data.drop(columns=[self.target])
            df_data_x = df_data.drop(columns=[self.target])
            self.scaler.fit(train_data_x.values)
            data_x = self.scaler.transform(df_data_x.values)

            train_data_y = train_data[[self.target]]
            df_data_y = df_data[[self.target]]
            # if not self.is_class:
            self.scaler_y.fit(train_data_y.values)
            data_y = self.scaler_y.transform(df_data_y.values)
            # else:
            #     data_y = df_data_y.values
            data = np.concatenate([data_x, data_y],
                                  axis=1)  # self.scaler.fit(train_data.values)  # data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        # if self.features=='MS':
        #     self.data_x = data[border1:border2,:-1]
        # else:
        if not self.is_class:
            self.data_x = data[border1:border2]
        else:
            self.data_x = data[border1:border2, :-1]
        if self.inverse:
            # self.data_y = df_data.values[border1:border2]
            self.data_y = np.hstack((data[border1:border2, :-1], df_data.values[border1:border2, -1:]))
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.data_stamp_date = df_stamp['date'].values

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        # if self.inverse:
        #     seq_y = np.concatenate(
        #         [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 0)
        # else:
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_y_date = self.data_stamp_date[r_begin:r_end]
        # if self.features == 'MS':
        #     seq_x = seq_x[:, :-1]
        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_y_date

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def inverse_transform_y(self, data):
        return self.scaler_y.inverse_transform(data)


class Dataset_Direct(Dataset_Custom):
    """One target at an exact trading-calendar lead after a contiguous history."""

    def __init__(self, *args, calendar_path=None, lead=1, **kwargs):
        self.calendar_path = calendar_path
        self.lead = int(lead)
        if self.lead not in (1, 5) or not calendar_path:
            raise ValueError('Direct data requires lead 1 or 5 and a calendar_path')
        super().__init__(*args, **kwargs)

    def __read_data__(self):
        if self.pred_len != 1 or not 0 < self.label_len <= self.seq_len:
            raise ValueError('Direct data requires pred_len=1 and 0<label_len<=seq_len')
        if self.features != 'MS' or self.is_class:
            raise ValueError('Direct data supports scalar MS regression only')
        raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        columns = self.cols.copy() if self.cols else [c for c in raw if c != 'date']
        columns.remove(self.target)
        raw = raw[['date'] + columns + [self.target]].copy()
        raw['date'] = pd.to_datetime(raw['date'])
        if raw['date'].isna().any() or raw['date'].duplicated().any() or not raw['date'].is_monotonic_increasing:
            raise ValueError('Input dates must be unique and increasing')
        values = raw[columns + [self.target]].apply(pd.to_numeric, errors='coerce')
        if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError('Direct input must contain finite numeric values; no future fill is allowed')
        calendar = pd.to_datetime(pd.read_csv(self.calendar_path)['date'])
        if calendar.isna().any() or calendar.duplicated().any() or not calendar.is_monotonic_increasing:
            raise ValueError('Trading calendar must have unique increasing dates')
        calendar_index = {date: i for i, date in enumerate(calendar)}
        if any(date not in calendar_index for date in raw['date']):
            raise ValueError('Input date missing from trading calendar')
        positions = np.asarray([calendar_index[d] for d in raw['date']], dtype=int)
        row_by_position = {position: row for row, position in enumerate(positions)}
        n = len(raw)
        train_end = int(n * .7)
        test_start = n - int(n * .2)
        if train_end <= self.seq_len or test_start <= train_end:
            raise ValueError('Insufficient data for target-date train/validation/test split')
        self.scaler = scaler_dic[self.scaler_type]()
        self.scaler_y = scaler_dic[self.scaler_type]()
        self.scaler.fit(values[columns].iloc[:train_end].to_numpy(dtype=float))
        self.scaler_y.fit(values[[self.target]].iloc[:train_end].to_numpy(dtype=float))
        x = self.scaler.transform(values[columns].to_numpy(dtype=float)) if self.scale else values[columns].to_numpy(dtype=float)
        y = self.scaler_y.transform(values[[self.target]].to_numpy(dtype=float)) if self.scale else values[[self.target]].to_numpy(dtype=float)
        self.data_x = np.concatenate([x, y], axis=1).astype(np.float32)
        self.data_y = values[self.target].to_numpy(dtype=np.float32) if self.inverse else y[:, 0].astype(np.float32)
        self.data_stamp = time_features(pd.DataFrame({'date': raw['date']}), timeenc=self.timeenc, freq=self.freq).astype(np.float32)
        self.data_stamp_date = raw['date'].to_numpy()
        self.windows = []
        for origin in range(self.seq_len - 1, n):
            origin_pos = positions[origin]
            history_positions = positions[origin - self.seq_len + 1:origin + 1]
            if not np.array_equal(history_positions, np.arange(origin_pos - self.seq_len + 1, origin_pos + 1)):
                continue
            target = row_by_position.get(origin_pos + self.lead)
            if target is None:
                continue
            split = 0 if target < train_end else (1 if target < test_start else 2)
            # Every scaler/training label must already be observed at validation
            # origin; every early-stopping label must be observed at test origin.
            if (split == 1 and origin < train_end - 1) or (split == 2 and origin < test_start - 1):
                continue
            if split == self.set_type:
                self.windows.append((origin, target))

    def __getitem__(self, index):
        origin, target = self.windows[index]
        start = origin - self.seq_len + 1
        prefix_start = origin - self.label_len + 1
        seq_x = self.data_x[start:origin + 1]
        prefix = self.data_x[prefix_start:origin + 1]
        # The target step carries only its label for loss; decoder input masks it.
        target_row = np.zeros((1, self.data_x.shape[1]), dtype=np.float32)
        target_row[0, -1] = self.data_y[target]
        seq_y = np.concatenate([prefix, target_row], axis=0)
        seq_x_mark = self.data_stamp[start:origin + 1]
        seq_y_mark = np.concatenate([self.data_stamp[prefix_start:origin + 1], self.data_stamp[target:target + 1]], axis=0)
        seq_dates = np.concatenate([self.data_stamp_date[prefix_start:origin + 1], self.data_stamp_date[target:target + 1]])
        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_dates

    def __len__(self):
        return len(self.windows)

    def inverse_transform_y(self, data):
        return self.scaler_y.inverse_transform(data) if self.scale else data


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, features='S', data_path='ETTh1.csv', target='OT', scale=True,
                 inverse=False, timeenc=0, freq='15min', cols=None, scaler_type="std"):
        # size [seq_len, label_len, pred_len]
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
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.scaler_type = scaler_type
        self.__read_data__()

    def __read_data__(self):
        self.scaler = scaler_dic[self.scaler_type]()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns);
            cols.remove(self.target);
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.data_stamp_date = df_stamp['date'].values

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_y_date = self.data_stamp_date[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_y_date

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
