"""
Dynamic Channel Fusion Module (DCFM_v2)
Parallel multi-scale receptive field convolutions with 1x1 fusion
designed to resolve extreme vehicle scale disparities in dense traffic.
"""

import torch
import torch.nn as nn


class DCFM_v2(nn.Module):
    """
    Dynamic Channel Fusion Module (DCFM_v2):
    Parallel receptive branches:
    - Branch 1: 3x3 Conv, dilation = 1 (Local structural details)
    - Branch 2: 3x3 Conv, dilation = 2 (Intermediate context)
    - Branch 3: 3x3 Conv, dilation = 5 (Large context & distant occlusions)
    - Branch 4: 1x1 Conv (Pointwise channel alignment)
    Followed by concatenation and 1x1 linear fusion.
    """
    def __init__(self, *args):
        super(DCFM_v2, self).__init__()
        self.c2 = args[0] if (args and isinstance(args[0], int)) else 512
        self.inner = max(1, self.c2 // 4)
        self.convs = None
        self.fusion = nn.Conv2d(self.inner * 4, self.c2, 1)

    def _build_layers(self, c1: int, device: torch.device):
        self.conv1 = nn.Conv2d(c1, self.inner, 3, padding=1, dilation=1).to(device)
        self.conv2 = nn.Conv2d(c1, self.inner, 3, padding=2, dilation=2).to(device)
        self.conv3 = nn.Conv2d(c1, self.inner, 3, padding=5, dilation=5).to(device)
        self.conv4 = nn.Conv2d(c1, self.inner, 1).to(device)
        self.fusion = self.fusion.to(device)
        self.convs = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.convs is None:
            self._build_layers(x.shape[1], x.device)
        out = torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)], dim=1)
        return self.fusion(out)
