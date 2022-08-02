
# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class My_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(My_Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class MyDataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''

    def __init__(self, x, y):
        self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = np.tanh(self.y[idx])
        return x, y

    def __len__(self):
        return len(self.y)


class ML():
    def __init__(using_pretrain = False):
        