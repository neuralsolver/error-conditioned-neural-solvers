import math

import scipy.io
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import device


TRAIN_PATH = '/data/train.mat'
TEST_PATH  = '/data/test.mat'


def load_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    data = scipy.io.loadmat(train_path)
    print(data['u'].shape)
    data_test = scipy.io.loadmat(test_path)
    print(data_test['u'].shape)
    print(data_test['t'])

    train_u_in = torch.from_numpy(data['a']).float().unsqueeze(1)
    train_u_out = torch.from_numpy(data['u']).permute(0, 3, 1, 2).float()
    print(train_u_out.shape)

    test_u_in = torch.from_numpy(data_test['a']).float().unsqueeze(1)
    test_u_out = torch.from_numpy(data_test['u']).permute(0, 3, 1, 2).float()
    print(test_u_out.shape)

    return train_u_in, train_u_out, test_u_in, test_u_out


def build_forcing(N=128):
    a = torch.linspace(0, 1, N + 1, device=device)
    a = a[0:-1]

    X, Y = torch.meshgrid(a, a)
    f = -4.0 * torch.cos(2 * math.pi * 4 * Y)  # KF
    return f


def build_train_loader(train_u_out, train_u_in, batch_size=24, num_workers=4):
    train_dataset = TensorDataset(train_u_out, train_u_in)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader
