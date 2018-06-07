import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TSDataset(Dataset):

    def __init__(self, series, sample_length):
        self.X, self.Y = get_data_samples(series, sample_length)

    def __getitem__(self, item):
        x = torch.from_numpy(self.X[item]).float()
        y = torch.from_numpy(self.Y[item]).float()
        return x, y, item

    def __len__(self):
        return self.X.shape[0]

    @classmethod
    def get_loader(cls, series, sample_length, batch_size=64, shuffle=True):
        dataset = cls(series, sample_length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader


def get_data_samples(train_data, sample_length=200):
    total_length = len(train_data)
    X = []
    Y = []
    for i in range(total_length - sample_length):
        X.append(train_data[i:i + sample_length])
        Y.append(train_data[i + 1:i + sample_length + 1])
    return np.array(X), np.array(Y)


def triangle(n=10000, nT=10, noise=0):
    assert nT % 2 == 0 and n % nT == 0
    half = nT // 2
    scope = (np.ones(half), -np.ones(half)) * (n // nT)
    gaussian_noise = np.random.normal(0, noise, n)
    return (np.hstack(scope)).cumsum() + gaussian_noise


def sine(n=10000, nT=100, delta=0.1, noise=0):
    T = nT * delta
    line = np.arange(0, n) * delta
    gaussian_noise = np.random.normal(0, noise, n)
    return np.sin(2 * np.pi * line / T) + gaussian_noise
