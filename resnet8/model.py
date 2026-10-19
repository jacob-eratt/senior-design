"""Floating-point ResNet8, following the MLPerf Tiny image-classification reference.

Topology (CIFAR-10, 32x32x3 input, ~78k parameters, ~12.5M MACs/image):

    stem    conv3x3(3->16)  + BN + ReLU                       32x32x16
    stack1  conv3x3(16->16) + BN + ReLU, conv3x3 + BN,
            identity shortcut, add, ReLU                      32x32x16
    stack2  conv3x3/s2(16->32) + BN + ReLU, conv3x3 + BN,
            conv1x1/s2 shortcut (bias, no BN), add, ReLU      16x16x32
    stack3  same as stack2 with 32->64                         8x8x64
    head    8x8 average pool, dense(64->10)

The network takes ``pixel / 255`` with no mean/std normalisation, so an 8-bit
camera pixel is already the quantized input of the accelerator (scale 1/255).
"""

import torch
from torch import nn


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding=kernel_size // 2, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = ConvBN(in_channels, out_channels, 3, stride)
        self.conv2 = ConvBN(out_channels, out_channels, 3, 1)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride, bias=True)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        y = torch.relu(self.conv1(x))
        y = self.conv2(y)
        return torch.relu(y + self.shortcut(x))


class ResNet8(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = ConvBN(3, 16, 3, 1)
        self.stack1 = ResidualBlock(16, 16, 1)
        self.stack2 = ResidualBlock(16, 32, 2)
        self.stack3 = ResidualBlock(32, 64, 2)
        self.pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.stem(x))
        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def load_resnet8(checkpoint, device="cpu"):
    model = ResNet8()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model.to(device).eval()
