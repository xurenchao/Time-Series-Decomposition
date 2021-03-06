import os

import numpy as np
import pandas as pd
from pandas.tseries.offsets import Day
import statsmodels.api as sm

import torch
from torch.utils.data import Dataset, DataLoader

DIR = os.path.dirname(__file__)

SPOT = pd.read_csv(os.path.join(DIR, 'data/price_demand_qld.csv'),
                   index_col='datetime', parse_dates=True).iloc[:, 0]

# SEASONAL = pd.read_csv(os.path.join(DIR, 'seasonal_qld.csv'),
#                    index_col='datetime', parse_dates=True) # no mean & std


TOTAL_STD = 864.283
TOTAL_MEAN = 5964.189


# cared = SPOT['2010':'2016']
# hour_mean = cared.groupby(cared.index.hour).mean()
# hour_std = cared.groupby(cared.index.hour).std()


def normalize(x):
    return (x - TOTAL_MEAN) / TOTAL_STD


# def reduce_hour_mean():
#     return (SPOT - SPOT.index.map(lambda x: hour_mean[x.hour])) / TOTAL_STD


# def to_stardard(x):
#     return x * TOTAL_STD + hour_mean.values  # broadcast


def get_daily_spot(data=SPOT):
    reshaped = pd.DataFrame()
    for i in range(48):
        reshaped['half' + str(i)] = data[(data.index.hour*2+data.index.minute//30) == i].values
    reshaped.index = pd.to_datetime(data[(data.index.hour*2+data.index.minute//30) == i].index.date)
    reshaped.index.name = 'date'
    return reshaped


def get_decomposed_spot(data=SPOT):
    daily_spot = get_daily_spot(data)
    trends = dict()
    seasonals = dict()
    residuals = dict()
    for key, col in daily_spot.T.iterrows():
        res = sm.tsa.seasonal_decompose(col, freq=7)
        trends[key] = res.trend
        seasonals[key] = res.seasonal
        residuals[key] = res.resid
    return (pd.DataFrame(trends)[['half%s' % i for i in range(48)]],
            pd.DataFrame(seasonals)[['half%s' % i for i in range(48)]],
            pd.DataFrame(residuals)[['half%s' % i for i in range(48)]])


def get_loader(dataset, batch_size=64, shuffle=True, num_workers=2):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader


class Holiday:

    def __init__(self, path):
        with open(path) as f:
            start = f.readline()
            assert start.startswith('START: ')
            self.start = pd.to_datetime(start[7:]).date()
            end = f.readline()
            assert end.startswith('END: ')
            self.end = pd.to_datetime(end[5:]).date()
            _ = f.readline()  # skip
            self.core = set()
            for line in f:
                self.core.add(pd.to_datetime(line).date())

    def query(self, date):
        dt = pd.to_datetime(date).date()
        assert dt >= self.start
        assert dt < self.end
        return dt in self.core


HOLIDAY = Holiday(os.path.join(DIR, 'data/holiday_qld.txt'))


class SlidingWindow:

    '''
    end_date: the last day for training
    N: training set size
    W: sliding window size for 1 step
    '''

    def __init__(self, end_date='2016-12-31', N=2000, W=7, W_forward=1):
        self.W = W
        self.W_forward = W_forward
        dt = pd.to_datetime(end_date)
        X = []
        Y = []
        for _ in range(N):
            x = self.get_features(dt)
            y = self.get_output(dt)
            dt -= Day(1)
            X.append(x)
            Y.append(y)
        self.X = np.array(X)
        self.Y = np.array(Y)

    def get_lagged_price(self, date, return_values=True):
        lagged = SPOT[pd.to_datetime(date) - Day(self.W):pd.to_datetime(date)]
        if return_values:
            lagged = lagged.values
        return lagged

    def get_holiday(self, date):
        return HOLIDAY.query(date)

    def get_seasonal(self, date, ohe=True):
        weekday = pd.to_datetime(date).weekday()
        if ohe:
            ret = [0] * 7
            ret[weekday] = 1
            return ret
        return weekday

    def get_features(self, date, price=None):
        if price is None:
            price = self.get_lagged_price(date)
        holiday = self.get_holiday(date)
        seasonal = self.get_seasonal(date)
        # seasonal = []
        return np.array(list(price) + [holiday] + list(seasonal))
        # return np.array(list(price) + list(seasonal))

    def get_output(self, date, return_values=True):
        date = pd.to_datetime(date)
        ret = SPOT[date:date + Day(self.W_forward)]
        if return_values:
            ret = ret.values
        return ret

    def update_training(self, date):
        x = self.get_features(date)
        y = self.get_output(date)
        self.X = np.array([x] + list(self.X[:-1, :]))
        self.Y = np.array([y] + list(self.Y[:-1, :]))


class DailyDataset(Dataset):
    '''
    TODO: clean the dummy ops.
    '''

    def __init__(self, end_date='2016-12-31', N=2000, W=14, seasonal=False):
        self.W = W
        self.seasonal = seasonal
        self.spot = get_daily_spot(normalize(SPOT))
        # self.spot = get_daily_spot(reduce_hour_mean())
        # self.spot = get_daily_spot(SPOT.diff(48))


        dt = pd.to_datetime(end_date)
        X = []
        Y = []
        dates = []
        for _ in range(N):
            x, y = self._sliding_window(dt)
            X.append(x)
            Y.append(y)
            dates.append(dt)
            dt -= Day(1)

        start_date = dates[-1]

        for _ in range(W):
            dates.append(dt)
            dt -= Day(1)

        self.X = np.array(list(reversed(X)))
        self.Y = np.array(list(reversed(Y)))
        self.dates = np.array(list(reversed(dates)))

        print("Data build range: [window(%s) - %s, %s]" %
              (self.dates[0], start_date, self.dates[-1]))

    @property
    def step_size(self):
        return len(self.dates)  # N + W

    def _sliding_window(self, date):
        _sliding = self.spot.loc[pd.to_datetime(date) - Day(self.W):
                                 pd.to_datetime(date)].values

        y = _sliding[-self.W:]
        if self.seasonal:
            s_sliding = SEASONAL.loc[pd.to_datetime(date) - Day(self.W):
                                 pd.to_datetime(date)].values
            _sliding = np.concatenate((_sliding, s_sliding), axis=-1)
        x = _sliding[:self.W]

        return x, y


    def get_io(self, start_date, end_date):
        _sliding = self.spot.loc[pd.to_datetime(start_date) - Day(self.W):
                                 pd.to_datetime(end_date)].values

        o_stream = _sliding[1:]
        if self.seasonal:
            s_sliding = SEASONAL.loc[pd.to_datetime(start_date) - Day(self.W):
                                 pd.to_datetime(end_date)].values
            _sliding = np.concatenate((_sliding, s_sliding), axis=-1)
        i_stream = _sliding[:-1]
        return torch.from_numpy(i_stream).float(), torch.from_numpy(o_stream).float()

    def __getitem__(self, item):
        x = torch.from_numpy(self.X[item]).float()
        y = torch.from_numpy(self.Y[item]).float()
        return x, y, item

    def __len__(self):
        return len(self.X)

class DailyDataset_pi(Dataset):
    '''
    TODO: clean the dummy ops.
    '''

    def __init__(self, end_date='2016-12-31', N=2000, W=14, seasonal=False):
        self.W = W
        self.seasonal = seasonal
        self.spot = get_daily_spot(normalize(SPOT))
        # self.spot = get_daily_spot(reduce_hour_mean())
        # self.spot = get_daily_spot(SPOT.diff(48))


        dt = pd.to_datetime(end_date)
        X = []
        Y = []
        dates = []
        for _ in range(N):
            x, y = self._sliding_window(dt)
            X.append(x)
            Y.append(y)
            dates.append(dt)
            dt -= Day(1)

        start_date = dates[-1]

        for _ in range(W):
            dates.append(dt)
            dt -= Day(1)

        self.X = np.array(list(reversed(X)))
        self.Y = np.array(list(reversed(Y)))
        self.dates = np.array(list(reversed(dates)))

        print("Data build range: [window(%s) - %s, %s]" %
              (self.dates[0], start_date, self.dates[-1]))

    @property
    def step_size(self):
        return len(self.dates)  # N + W

    def _sliding_window(self, date):
        _sliding = self.spot.loc[pd.to_datetime(date) - Day(self.W):
                                 pd.to_datetime(date)].values

        if self.seasonal:
            s_sliding = SEASONAL.loc[pd.to_datetime(date) - Day(self.W):
                                 pd.to_datetime(date)].values
            _sliding = _sliding - s_sliding
        x = _sliding[:self.W]
        y = _sliding[-self.W:]

        return x, y


    def get_io(self, start_date, end_date):
        _sliding = self.spot.loc[pd.to_datetime(start_date) - Day(self.W):
                                 pd.to_datetime(end_date)].values
        
        if self.seasonal:
            s_sliding = SEASONAL.loc[pd.to_datetime(start_date) - Day(self.W):
                                 pd.to_datetime(end_date)].values
            _sliding = _sliding - s_sliding
        i_stream = _sliding[:-1]
        o_stream = _sliding[1:]
        return torch.from_numpy(i_stream).float(), torch.from_numpy(o_stream).float()

    def __getitem__(self, item):
        x = torch.from_numpy(self.X[item]).float()
        y = torch.from_numpy(self.Y[item]).float()
        return x, y, item

    def __len__(self):
        return len(self.X)


class DailyDataset_nn(Dataset):
    '''
    NN dataset for electricity price prediction
    '''
    def __init__(self, end_date='2016-12-31', N=2000, W=14):
        self.W = W
        self.spot = get_daily_spot(normalize(SPOT))
        # self.spot = get_daily_spot(reduce_hour_mean())
        # self.spot = get_daily_spot(SPOT.diff(48))


        dt = pd.to_datetime(end_date)
        X = []
        Y = []
        dates = []
        for _ in range(N):
            x, y = self._sliding_window(dt)
            X.append(x)
            Y.append(y)
            dates.append(dt)
            dt -= Day(1)

        start_date = dates[-1]

        for _ in range(W):
            dates.append(dt)
            dt -= Day(1)

        self.X = np.array(list(reversed(X)))
        self.Y = np.array(list(reversed(Y)))
        self.dates = np.array(list(reversed(dates)))

        print("Data build range: [window(%s) - %s, %s]" %
              (self.dates[0], start_date, self.dates[-1]))

    @property
    def step_size(self):
        return len(self.dates)  # N + W

    def _sliding_window(self, date):
        _sliding = self.spot.loc[pd.to_datetime(date) - Day(self.W):
                                 pd.to_datetime(date)].values
        x = _sliding[:self.W]
        y = _sliding[-1]
        return x, y


    def get_io(self, start_date, end_date):
        _sliding = self.spot.loc[pd.to_datetime(start_date) - Day(self.W):
                                 pd.to_datetime(end_date)].values
        i_stream = _sliding[:-1]
        o_stream = _sliding[self.W:]
        return torch.from_numpy(i_stream).float(), torch.from_numpy(o_stream).float()

    def __getitem__(self, item):
        x = torch.from_numpy(self.X[item]).float()
        y = torch.from_numpy(self.Y[item]).float()
        return x, y, item

    def __len__(self):
        return len(self.X)

class SpotDataset(Dataset):

    def __init__(self, end_date='2016-12-31', N=2000, W=14, segment=48):
        self.W = W
        self.segment = segment

        dt = pd.to_datetime(end_date)
        X = []
        Y = []
        dates = []
        seasonal = []
        holiday = []
        for _ in range(N):
            x, y, s, h = self._sliding_window(dt)
            X.append(x)
            Y.append(y)
            seasonal.append(s)
            holiday.append(h)
            dates.append(dt)
            dt -= Day(1)

        self.X = np.array(list(reversed(X)))
        self.Y = np.array(list(reversed(Y)))
        self.dates = np.array(list(reversed(dates)))
        self.seasonal = np.array(list(reversed(seasonal)))
        self.holiday = np.array(list(reversed(holiday)))
        self.step_size = N + W

        print("Data build range:", [self.dates[0], self.dates[-1]])

    def _sliding_window(self, date):
        _sliding = SPOT[pd.to_datetime(date) - Day(self.W):
                        pd.to_datetime(date) + Day(1)]  # TODO
        sliding = _sliding.values.reshape(-1, self.segment)
        timestamps = _sliding.index
        weekdays = np.array([d.weekday()
                             for d in timestamps]).reshape(-1, self.segment)
        holidays = np.array([int(HOLIDAY.query(d))
                             for d in timestamps]).reshape(-1, self.segment)
        return (sliding[:self.W, :], sliding[-self.W:, :],
                weekdays[-self.W:, 0], holidays[-self.W:, 0])

    def __getitem__(self, item):
        x = torch.from_numpy(self.X[item]).float()
        y = torch.from_numpy(self.Y[item]).float()
        # seasonal flag of predict step
        seasonal = torch.from_numpy(self.seasonal[item])
        # holiday flag of predict step
        holiday = torch.from_numpy(self.holiday[item])
        return x, y, seasonal, holiday, item

    def __len__(self):
        return len(self.X)

    @classmethod
    def get_loader(cls, batch_size=64, shuffle=True, *args, **kwargs):
        dataset = cls(*args, **kwargs)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader
