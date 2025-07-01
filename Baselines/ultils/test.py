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

def test(net, testloader):
    criterion = nn.CrossEntropyLoss()
    corrects, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images).to(DEVICE)
            predicted = torch.argmax(outputs, dim=1)
            loss += criterion(outputs, labels).item() * images.shape[0]
            corrects += torch.sum(predicted == labels).item()
    loss /= len(testloader.sampler)
    accuracy = corrects / len(testloader.sampler)
    return loss, accuracy