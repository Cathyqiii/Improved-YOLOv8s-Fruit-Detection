import torch
import torch.nn as nn
import math

# 定义EVC模块类
class EVC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 全局依赖分支
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 1),
            nn.GELU(),
            nn.Conv2d(in_channels//4, in_channels, 1)
        )
        # 局部中心分支
        self.local_center = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        global_feat = self.mlp(x)
        local_feat = self.local_center(x)
        return x * (global_feat + local_feat)