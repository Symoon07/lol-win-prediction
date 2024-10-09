import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data.csv')
data = data.drop(['gameId'], axis=1)

features = ['blueWardsPlaced', 'blueWardsDestroyed', 'blueFirstBlood', 'blueKills', 'blueDeaths',
            'blueAssists', 'blueEliteMonsters', 'blueDragons', 'blueHeralds', 'blueTowersDestroyed',
            'blueTotalGold', 'blueAvgLevel', 'blueTotalExperience', 'blueTotalMinionsKilled',
            'blueTotalJungleMinionsKilled', 'blueGoldDiff', 'blueExperienceDiff', 'blueCSPerMin',
            'blueGoldPerMin',
            'redWardsPlaced', 'redWardsDestroyed', 'redFirstBlood', 'redKills', 'redDeaths',
            'redAssists', 'redEliteMonsters', 'redDragons', 'redHeralds', 'redTowersDestroyed',
            'redTotalGold', 'redAvgLevel', 'redTotalExperience', 'redTotalMinionsKilled',
            'redTotalJungleMinionsKilled', 'redGoldDiff', 'redExperienceDiff', 'redCSPerMin',
            'redGoldPerMin']
x_train, x_test, y_train, y_test = train_test_split(data[features].values, data['blueWins'].values, test_size=0.2, random_state=42)

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

mean = x_train_tensor.mean(dim=0)
std = x_train_tensor.std(dim=0)
x_train_tensor_normalized = (x_train_tensor - mean) / std
mean = x_test_tensor.mean(dim=0)
std = x_test_tensor.std(dim=0)
x_test_tensor_normalized = (x_test_tensor - mean) / std

dataset = TensorDataset(x_train_tensor_normalized, y_train_tensor)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(38, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Sequential(self.fc1, self.relu, nn.BatchNorm1d(64), self.dropout,
                                self.fc2, self.relu, nn.BatchNorm1d(32), self.dropout,
                                self.fc3)

    def forward(self, x):
        return self.fc(x)


model = FNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

train_loss_history = []
test_loss_history = []
train_accuracy_history = []
test_accuracy_history = []

epochs = 100
for epoch in range(epochs):
    model.train()
    loss = 0
    accuracy = 0
    avg_loss = 0
    avg_accuracy = 0
    num_batches = 0
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()

        output = model(x_batch)

        loss = criterion(output, y_batch)
        pred = (output.sigmoid() >= 0.5).float()
        accuracy = (pred == y_batch).float().mean()

        avg_loss += loss.item()
        avg_accuracy += accuracy.item()
        num_batches += 1

        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = ((model(x_test_tensor_normalized)).sigmoid() >= 0.5).float()
    test_accuracy = (pred == y_test_tensor).float().mean().item()
    test_loss = criterion(model(x_test_tensor_normalized), y_test_tensor).item()

    avg_loss /= num_batches
    avg_accuracy /= num_batches
    train_loss_history.append(avg_loss)
    test_loss_history.append(test_loss)
    train_accuracy_history.append(avg_accuracy)
    test_accuracy_history.append(test_accuracy)

    scheduler.step(test_loss)

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: Train {avg_loss:.4f} | Test {test_loss:.4f}, Accuracy: Train {avg_accuracy:.4f} | Test {test_accuracy:.4f}')

model.eval()
with torch.no_grad():
    pred = ((model(x_test_tensor_normalized)).sigmoid() >= 0.5).float()
accuracy = (pred == y_test_tensor).float().mean()
print(f'Final Accuracy: {accuracy:.4f}')