import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


class CVModel(nn.Module):
    """
    https://medium.com/analytics-vidhya/resnet-understand-and-implement-from-scratch-d0eb9725e0db
    ResNet-18
    Param:
    """

    VERSION = "resnet18"

    def __init__(self):
        super().__init__()

        # (128 + 2 * 1 - 7) / 2 + 1 = 62
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # (62 + 2 * 1 - 3) / 2 + 1 = 31
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # (31 + 2 * 1 - 3) / 1 + 1 = 31
        self.cnn2_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.cnn2_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.cnn2_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.cnn2_4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        # (31 + 2 * 1 - 3) / 2 + 1 = 16
        self.cnn3_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.cnn3_2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.cnn3_2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.cnn3_2_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Projection
        # (31 + 2 * 0 - 1) / 2 + 1 = 16
        self.cnn3_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=2, padding=0)

        # (16 + 2 * 1 - 3) / 2 + 1 = 8
        self.cnn4_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.cnn4_2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.cnn4_2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.cnn4_2_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        # Projection
        # (16 + 2 * 0 - 1) / 2 + 1 = 8
        self.cnn4_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0)

        # (8 + 2 * 1 - 3) / 2 + 1 = 4
        self.cnn5_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.cnn5_2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.cnn5_2_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.cnn5_2_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        # Projection
        # (8 + 2 * 0 - 1) / 2 + 1 = 4
        self.cnn5_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0)

        self.cam = nn.Identity()

        # 4 -> 1
        self.avg6_1 = nn.AdaptiveAvgPool2d(1)
        # Flatten -> (256, 1, 1)
        self.fc6_2 = torch.nn.Linear(256, 128)
        self.bn6_3 = nn.BatchNorm1d(128)
        self.fc6_4 = torch.nn.Linear(128, 43)

        self.dropout = nn.Dropout(p=0.50)

    def forward(self, x: Tensor):
        x = self.cnn1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.mp1(x)
        y = x.clone()

        # Layer 1
        x = self.cnn2_1(x)
        x = self.cnn2_2(x)
        x += y
        x = F.relu(x)
        y = x.clone()
        x = self.cnn2_3(x)
        x = self.cnn2_4(x)
        x += y
        x = F.relu(x)
        y = x.clone()

        # Layer 2
        x = self.cnn3_1(x)
        x = self.cnn3_2_1(x)
        y = self.cnn3_3(y)
        x += y
        x = F.relu(x)
        y = x.clone()
        x = self.cnn3_2_2(x)
        x = self.cnn3_2_3(x)
        x += y
        x = F.relu(x)
        y = x.clone()

        # Layer 3
        x = self.cnn4_1(x)
        x = self.cnn4_2_1(x)
        y = self.cnn4_3(y)
        x += y
        x = F.relu(x)
        y = x.clone()
        x = self.cnn4_2_2(x)
        x = self.cnn4_2_3(x)
        x += y
        x = F.relu(x)
        y = x.clone()

        # Layer 4
        x = self.cnn5_1(x)
        x = self.cnn5_2_1(x)
        y = self.cnn5_3(y)
        x += y
        x = F.relu(x)
        y = x.clone()
        x = self.cnn5_2_2(x)
        x = self.cnn5_2_3(x)
        x += y
        x = self.cam(x)
        x = F.relu(x)

        x = self.avg6_1(x)
        # Flatten : (batch_size, channel, height_dim, width_dim) -> (batch_size, flatten_image_size)
        x = x.view(-1, 256)
        x = self.fc6_2(x)
        x = self.bn6_3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc6_4(x)

        return x
