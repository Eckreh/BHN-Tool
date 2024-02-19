# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:56:08 2024

@author: Marcel
"""


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from os import listdir
from os.path import isfile, join
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


def load_data(folder_path):
    
    files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    
    data = [np.load(join(folder_path, f), allow_pickle=True) for f in files]
    
    arrays = [np.array(timeseries) for timeseriescollection in data for timeseries in timeseriescollection]
    
    return arrays


class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(2 * 4096, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 2 * 4096)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
    
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

good_folder_path = './Good'
bad_folder_path = './Bad'

good_data = load_data(bad_folder_path)

X = torch.tensor(good_data, dtype=torch.float32)

model = DeepNN()

model.load_state_dict(torch.load("test.pth"))

model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    outputs = model(X)
    predictions = (outputs >= 0.5).float()
    
    
print(predictions)
print("exit")

