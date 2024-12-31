import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc_input_size = 64 * 7 * 7
        
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, out_size)
        
        self.optimizer = optim.SGD(self.parameters(), lr=lrate)
    
    def forward(self, x):
        x = x.view(-1, 3, 31, 31)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def step(self, x, y):
        self.optimizer.zero_grad()
        yhat = self.forward(x)
        loss = self.loss_fn(yhat, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

def fit(train_set, train_labels, dev_set, epochs, batch_size=100):
    train_set = torch.tensor(train_set, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    dev_set = torch.tensor(dev_set, dtype=torch.float32)

    net = NeuralNet(lrate=0.015, loss_fn=nn.CrossEntropyLoss(), in_size=31*31*3, out_size=4)
    
    losses = []

    train_data = torch.utils.data.TensorDataset(train_set, train_labels)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        net.train()
        total_loss = 0.0
        
        for i, (x_batch, y_batch) in enumerate(train_loader):
            loss = net.step(x_batch, y_batch)
            total_loss += loss
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

    net.eval()
    dev_predictions = net(dev_set).argmax(dim=1).numpy()

    return losses, dev_predictions, net
