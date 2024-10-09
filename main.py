import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

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
x_train = data[features].values.astype(np.float32)
y_train = data['blueWins'].values.astype(np.float32)
x_tensor = torch.tensor(x_train)
y_tensor = torch.tensor(y_train).unsqueeze(1)
dataset = TensorDataset(x_tensor, y_tensor)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(38, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

        self.fc = nn.Sequential(self.fc1, self.fc2, self.fc3, self.fc4, self.fc5)

    def forward(self, x):
        return self.fc(x)


model = FNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_history = []
accuracy_history = []

epochs = 20
for epoch in range(epochs):
    model.train()
    loss = 0
    accuracy = 0
    avg_loss = 0
    avg_accuracy = 0
    num_batches = 0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()

        output = model(x_tensor)

        loss = criterion(output, y_tensor)
        avg_loss += loss.item()
        predicted = (output.sigmoid() >= 0.5).float()
        accuracy = (predicted == y_tensor).float().mean()
        avg_accuracy += accuracy.item()
        num_batches += 1

        loss.backward()
        optimizer.step()
    loss_history.append(avg_loss / num_batches)
    accuracy_history.append(avg_accuracy / num_batches)

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}')