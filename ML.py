
# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import json
import os

class My_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(My_Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16, bias=False),
            nn.ReLU(),
            nn.Linear(16, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 16, bias=False),
            nn.ReLU(),
            nn.Linear(16, output_dim, bias=False),
            # nn.Tanh(),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class My_Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''

    def __init__(self, x, y):
        self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        # y = np.tanh(self.y[idx])
        return x, y

    def __len__(self):
        return len(self.y)


class ML():
    def __init__(self, PRETRAIN_MODEL=False, TRAIN=False, input_dim=0, output_dim=1):
        self.model =  My_Model(input_dim, output_dim)
        self.PRETRAIN_MODEL = PRETRAIN_MODEL
        self.TRAIN = TRAIN

        if self.PRETRAIN_MODEL and os.path.exists("My_Model"):
            self.model.load_state_dict(torch.load("My_Model"))

        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

    def train(self, i, x_train, y_train, loss_plot):
        # with open("dataset.json", "w") as outfile:
        #     data = {}
        #     data["x_train"] = list(x_train)
        #     data["y_train"] = list(y_train)
        #     json.dump(data, outfile)

        train_dataset = My_Dataset(x_train, y_train)
        bs = min(1024, 8*(i+1))
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        self.model.train()
        if i < 3: epoch_n = 100
        else: epoch_n = 50
        loss_record = []

        for epoch in range(epoch_n):
            for x, y in train_loader:
                output = self.model(x)
                loss = self.criterion(output, y)
                # Compute gradient(backpropagation).
                loss.backward()
                # Update parameters.
                self.optimizer.step()
                loss_record.append(loss.detach().item())
            
            if (epoch+1) % 10 == 0:
                mean_train_loss = sum(loss_record)/len(loss_record)
                loss_plot.update([mean_train_loss])  # plot loss

    

        