import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        h = 150 
        self.layer1 = nn.Linear(in_size, h)
        self.layer2 = nn.Linear(h, out_size)
        self.activation = nn.ReLU()
        self.optimizer = optim.SGD(self.parameters(), lr=lrate)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x

    def step(self, x, y):
        self.optimizer.zero_grad()
        yhat = self.forward(x)
        loss = self.loss_fn(yhat, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

def fit(train_set, train_labels, dev_set, epochs, batch_size=100):
    losses = []
    predicted_labels = []

    train_set = (train_set - torch.mean(train_set, axis=0)) / torch.std(train_set, axis=0)
    dev_set = (dev_set - torch.mean(dev_set, axis=0)) / torch.std(dev_set, axis=0)

    dataset = get_dataset_from_arrays(train_set.numpy(), train_labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = NeuralNet(lrate=0.01, loss_fn=torch.nn.CrossEntropyLoss(), in_size=train_set.shape[1], out_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            x_batch = batch['features']
            y_batch = batch['labels']
            optimizer.zero_grad()
            yhat = model(x_batch)
            loss = model.loss_fn(yhat, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / len(train_loader))

    dev_tensor = torch.tensor(dev_set, dtype=torch.float32).clone().detach()
    dev_predictions = model(dev_tensor)
    _, predicted_labels = torch.max(dev_predictions, 1)

    return losses, predicted_labels.numpy(), model
