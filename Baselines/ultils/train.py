import random
import copy
import torch
import torch.nn as nn
from torch.optim import SGD

seed_value = 42
random.seed(seed_value)
torch.manual_seed(seed_value)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(net, trainloader, learning_rate: float):
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=learning_rate)
    net.train()
    running_loss, running_corrects = 0.0, 0    
    for images, labels in trainloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = net(images).to(DEVICE)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predicted = torch.argmax(outputs, dim=1)
        running_loss += loss.item() * images.shape[0]
        running_corrects += torch.sum(predicted == labels).item()

    running_loss /= len(trainloader.sampler)
    acccuracy = running_corrects / len(trainloader.sampler)
    return running_loss, acccuracy