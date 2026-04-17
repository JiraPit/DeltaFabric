import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)
        self.norm = nn.BatchNorm2d(out_channels)
        self.pool_layer = nn.MaxPool2d(2, 2) if pool else None

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        if self.pool_layer is not None:
            x = self.pool_layer(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(1, 16, 3, pool=True)
        self.conv2 = ConvBlock(16, 16, 3, pool=True)
        self.fc1 = nn.Linear(16 * 5 * 5, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
