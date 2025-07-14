import random
import torch
import torch.nn as nn
from torch.optim import SGD

seed_value = 42
random.seed(seed_value)
torch.manual_seed(seed_value)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_feddyn(
    net: nn.Module,
    trainloader,
    learning_rate: float,
    alpha: float,
    global_model: nn.Module
):
    """
    Train the model using FedDyn's dynamic regularization for one epoch.

    Args:
        net (nn.Module): Local model.
        trainloader (DataLoader): Training data loader.
        learning_rate (float): Learning rate.
        alpha (float): FedDyn regularization coefficient.
        global_model (nn.Module): Global model received from server.

    Returns:
        Tuple[float, float]: Training loss and accuracy.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=learning_rate)
    net.train()

    total_samples = len(trainloader.sampler)
    running_loss, running_corrects = 0.0, 0

    global_params = [p.detach().clone() for p in global_model.parameters()]

    for images, labels in trainloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        outputs = net(images)
        loss_ce = criterion(outputs, labels)

        # FedDyn regularization: alpha/2 * ||w - w_global||^2
        loss_reg = 0.0
        for p, p_global in zip(net.parameters(), global_params):
            loss_reg += torch.sum((p - p_global) ** 2)
        loss = loss_ce + (alpha / 2.0) * loss_reg

        loss.backward()
        optimizer.step()

        predicted = torch.argmax(outputs, dim=1)
        running_loss += loss.item() * images.size(0)
        running_corrects += (predicted == labels).sum().item()

    avg_loss = running_loss / total_samples
    accuracy = running_corrects / total_samples
    return avg_loss, accuracy
