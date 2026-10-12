"""
Convolutional Block Attention Module (CBAM)
Embedded at Layer 23 of YOLOv11m for Dense Traffic Perception.
Sequential Channel Attention + Spatial Attention sub-modules.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel Attention Module:
    Explores inter-channel relationships using average and max pooling
    followed by a shared multi-layer perceptron (MLP).
    """
    def __init__(self, in_planes: int, ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_planes = max(1, in_planes // ratio)
        self.fc1 = nn.Conv2d(in_planes, reduced_planes, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module:
    Explores inter-spatial relationships using channel-wise average and max pooling
    followed by a standard 7x7 convolution to generate spatial weight map.
    """
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM):
    Applies sequential Channel Attention followed by Spatial Attention.
    """
    def __init__(self, c1: int, ratio: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.channel_gate = ChannelAttention(c1, ratio)
        self.spatial_gate = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention refinement
        x_out = self.channel_gate(x) * x
        # Spatial attention refinement
        x_out = self.spatial_gate(x_out) * x_out
        return x_out


# Alias for Ultralytics YOLO architecture parser compatibility
ULTIMATE_CBAM = CBAM
