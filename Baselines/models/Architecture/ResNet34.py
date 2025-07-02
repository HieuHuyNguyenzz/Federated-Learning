import torch
import torch.nn as nn
import torchvision.models as models

class ResNet34(nn.Module):
    def __init__(self, num_classes=100, channel_size=3):
        super(ResNet34, self).__init__()
        self.resnet = models.ResNet34(pretrained=False)

        self.resnet.conv1 = nn.Conv2d(channel_size, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        return x