import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import numpy as np


# ========================Data generation==============
class FinanceData(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        index = self.x[index]
        return index


# ========================Model definition==============
model = torchvision.models.resnet18(pretrained=True)

# ========================Training Pipeline==============
x = np.random((500, 1000, 100))
trainiter = FinanceData(x)
loss_function = torch.nn.MSELoss()


def train(config, model, train_iter):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.num_epochs):
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            loss = loss_function(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), "return_resnet.pth")

