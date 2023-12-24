import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


class CVModel(nn.Module):
    """
    https://github.com/pytorch/examples/blob/main/mnist/main.py
    MNIST
    Param: 1,204,715
    """

    VERSION = "mnist"

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)  # 9216 / 64 = 144
        self.fc2 = nn.Linear(128, 43)

    def forward(self, x) -> Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, 12)  # Changed: Adaptive pooling
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
        return F.log_softmax(x, dim=1)
