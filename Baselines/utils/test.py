import random
import torch
import torch.nn as nn

seed_value = 42
random.seed(seed_value)
torch.manual_seed(seed_value)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(net: nn.Module, testloader):
    """
    Evaluate the model on the test set.

    Args:
        net (nn.Module): Trained neural network.
        testloader (DataLoader): Dataloader for the test set.

    Returns:
        Tuple[float, float]: Average loss and accuracy on the test set.
    """
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = len(testloader.sampler)

    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss = criterion(outputs, labels)
            predicted = torch.argmax(outputs, dim=1)
            total_loss += loss.item() * images.size(0)
            total_correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy