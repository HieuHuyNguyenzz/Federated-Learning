import random
import torch
import torch.nn as nn
from torch.optim import SGD
import copy

seed_value = 42
random.seed(seed_value)
torch.manual_seed(seed_value)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(net: nn.Module, trainloader, learning_rate: float, proximal_mu: float = 0.0):
    """
    Train a model for one epoch on the given dataloader.

    Args:
        net (nn.Module): The neural network to train.
        trainloader (DataLoader): The dataloader for training data.
        learning_rate (float): Learning rate for the optimizer.
        proximal_mu (float): Proximal term coefficient for FedProx, default is 0.0.

    Returns:
        Tuple[float, float]: Tuple of average training loss and accuracy.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=learning_rate)
    net.train()

    running_loss = 0.0
    running_corrects = 0
    global_params = copy.deepcopy(net).parameters()
    total_samples = len(trainloader.sampler)

    for images, labels in trainloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = net(images)

        proximal_term = 0.0
        for local_weights, global_weights in zip(net.parameters(), global_params):
            proximal_term += (local_weights - global_weights).norm(2)
        loss = criterion(outputs, labels) + (proximal_mu / 2) * proximal_term

        loss.backward()
        optimizer.step()

        predicted = torch.argmax(outputs, dim=1)
        running_loss += loss.item() * images.size(0)
        running_corrects += (predicted == labels).sum().item()

    avg_loss = running_loss / total_samples
    accuracy = running_corrects / total_samples
    return avg_loss, accuracy
