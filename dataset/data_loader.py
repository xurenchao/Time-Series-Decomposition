import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pandas.tseries.offsets import MonthOffset

DATA_NAME = ['mackeyglass', 'international-airline-passengers', 'unemployment-rate', 'ozon-concentration-downtown']

USE_DATA = 2

DIR = os.path.dirname(__file__)
if USE_DATA == 0:
    SPOT = pd.read_csv(os.path.join(DIR, 'data', DATA_NAME[USE_DATA] + '.csv'),
                       index_col='Time', parse_dates=True).iloc[:, 0]
else:
    SPOT = pd.read_csv(os.path.join(DIR, 'data', DATA_NAME[USE_DATA] + '.csv'),
                       index_col='Month', parse_dates=True).iloc[:, 0]


def get_year_spot(data=SPOT):
    reshaped = pd.DataFrame()
    for i in range(1, 13):
        reshaped['month' + str(i)] = data[data.index.month == i].values
    reshaped.index = pd.to_datetime(data[data.index.month == i].index).year
    reshaped.index.name = 'year'
    return reshaped


class PeriodDataset(Dataset):
    def __init__(self, end='2017-12-01', N=39, W=2):
        self.end = end
        self.N = N
        self.W = W
        self.spot = get_year_spot()

        ts = pd.Timestamp(end)

        X = []
        Y = []
        years = []
        for _ in range(N):
            x, y = self._sliding_window(ts)
            X.append(x)
            Y.append(y)
            years.append(ts)
            ts -= MonthOffset(months=12)

        start_year = years[-1]

        for _ in range(W):
            years.append(ts)
            ts -= MonthOffset(months=12)

        self.X = np.array(list(reversed(X)))
        self.Y = np.array(list(reversed(Y)))
        self.years = np.array(list(reversed(years)))

        print("Data build range: [window(%s) - %s, %s]" %
              (self.years[0], start_year, self.years[-1]))

    def _sliding_window(self, month):
        a = pd.Timestamp(month) - MonthOffset(months=self.W*12)
        b = pd.Timestamp(month)
        _sliding = self.spot.loc[a.year:b.year].values
        x = _sliding[:self.W]
        y = _sliding[-self.W:]
        return x, y

    def get_io(self, start_month, end_month):
        _sliding = self.spot.loc[(pd.Timestamp(start_month) - MonthOffset(months=self.W*12)).year:
                                 pd.Timestamp(end_month).year].values
        i_stream = _sliding[:-1]
        o_stream = _sliding[1:]
        return torch.from_numpy(i_stream).float(), torch.from_numpy(o_stream).float()

    def __getitem__(self, item):
        x = torch.from_numpy(self.X[item]).float()
        y = torch.from_numpy(self.Y[item]).float()#.clamp(-2, 5)
        return x, y, item

    def __len__(self):
        return len(self.X)

    @classmethod
    def get_loader(cls, batch_size=3, shuffle=True, *args, **kwargs):
        dataset = cls(*args, **kwargs)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader
