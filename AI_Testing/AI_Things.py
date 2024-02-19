# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:56:08 2024

@author: av179
"""


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from os import listdir
from os.path import isfile, join
import torch.nn as nn
import torch.optim as optim

def load_data(folder_path):
    
    files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    data = [np.load(join(folder_path, f), allow_pickle=True) for f in files]
    arrays = [np.array(timeseries) for timeseriescollection in data for timeseries in timeseriescollection]
    
    return arrays

good_folder_path = './Good'
bad_folder_path = './Bad'

good_data = load_data(good_folder_path)
bad_data = load_data(bad_folder_path)

np.random.shuffle(bad_data)

print(len(good_data))

if len(bad_data) > len(good_data):
    bad_data = bad_data[:len(good_data)]


# Sample data, replace this with your own data
X = np.concatenate([good_data, bad_data], axis=0)
y = np.asarray([1]*len(good_data)+[0]*len(bad_data), dtype="long")  # Labels


X = X.reshape(X.shape[0], -1)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


dataset = TensorDataset(X, y)
train_data, test_data = random_split(dataset, [0.9, 0.1])


# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


model = nn.Sequential(
    nn.Linear(2*4096, 512), 
    nn.ReLU(),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Forward pass through the LSTM layer
        lstm_out, _ = self.lstm(x)

        # Extract the output from the last time step
        last_output = lstm_out[:, -1, :]

        # Fully connected layer for classification
        output = self.fc(last_output)

        return output

#model = nn.LSTM(2, 512, num_layers=4, batch_first=True)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

# Training loop
epochs = 200

for epoch in range(epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


model.eval()

# Evaluate the model
with torch.no_grad():
    outputs = model(dataset.tensors[0][test_data.indices])
    predictions = (outputs >= 0.5).float()
    accuracy = torch.mean((predictions == dataset.tensors[1][test_data.indices].view(-1, 1)).float())
    print(f'Accuracy: {accuracy.item():.4f}')


torch.save(model.state_dict(), "test.pth")
