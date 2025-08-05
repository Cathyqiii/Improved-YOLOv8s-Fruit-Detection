import numpy as np
import torch
from torch import nn
from torch.nn import init




# -------------------- 自定义模块定义 --------------------
class SE(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = int(channels)
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.channels, self.channels // self.reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels // self.reduction, self.channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)