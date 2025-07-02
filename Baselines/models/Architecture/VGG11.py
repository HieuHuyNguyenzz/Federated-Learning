import torch
import torch.nn as nn
import torchvision.models as models

class VGG11(nn.Module):
    def __init__(self, num_classes=100, channel_size=3):
        super(VGG11, self).__init__()
        self.vgg11 = models.vgg11(weights=None)

        # Adjust the first convolutional layer to accept the specified number of input channels
        self.vgg11.features[0] = nn.Conv2d(channel_size, 64, kernel_size=3, stride=1, padding=1)

        # Adjust the final fully connected layer to output the specified number of classes
        self.vgg11.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vgg11(x)
        return x