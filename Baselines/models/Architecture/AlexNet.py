import torch
import torch.nn as nn
import torchvision.models as models

class AlexNet(nn.Module):
    def __init__(self, num_classes=100, input_shape = [3, 224, 224]):
        super(AlexNet, self).__init__()
        self.alexnet = models.AlexNet(weights=None)

        self.alexnet.features[0] = nn.Conv2d(input_shape[0], 64, kernel_size=11, stride=4, padding=2)

        self.alexnet.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.alexnet(x)
        return x