import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_shape = [1, 28, 28], classes = 10) -> None:
        super(MLP, self).__init__()
        input_neural = input_shape[0] * input_shape[1] * input_shape[2]
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(input_neural, 200)
        self.layer2 = nn.Linear(200, 200)
        self.layer3 = nn.Linear(200, classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x