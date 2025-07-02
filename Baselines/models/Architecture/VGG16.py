import torch
import torch.nn as nn
import torchvision.models as models

class VGG16(nn.Module):
    def __init__(self, num_classes=100, channel_size=3):
        super(VGG16, self).__init__()
        self.vgg16 = models.vgg16(weights=None)

        # Adjust the first convolutional layer to accept the specified number of input channels
        self.vgg16.features[0] = nn.Conv2d(channel_size, 64, kernel_size=3, stride=1, padding=1)

        # Adjust the final fully connected layer to output the specified number of classes
        self.vgg16.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vgg16(x)
        return x