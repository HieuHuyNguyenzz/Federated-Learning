import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_shape = [1, 28, 28], num_classes=10) -> None:
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()
        self._feature_dim = None 

        self.fc1 = None
        self.fc2 = None 
        self.num_classes = num_classes

    def _initialize_fc(self, x):
        feature_dim = x.view(x.size(0), -1).size(1)
        self.fc1 = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(512, self.num_classes)
        self._feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        if self.fc1 is None or self.fc2 is None:
            self._initialize_fc(x)
            if torch.cuda.is_available():
                self.fc1 = self.fc1.cuda()
                self.fc2 = self.fc2.cuda()
        x = self.fc1(x)
        x = self.fc2(x)
        return x