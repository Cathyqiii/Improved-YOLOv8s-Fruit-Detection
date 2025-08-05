import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv

class SPPFCSPC(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):  # 强制k为元组
        super().__init__()
        # 处理YAML传入的列表参数
        if isinstance(k, list):
            k = tuple(k)
        # 通道数计算
        c_ = c2 // 2
        # 卷积层定义
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        # 多尺度池化层
        self.m = nn.ModuleList([
            nn.MaxPool2d(kernel_size=x, stride=1, padding=x//2) for x in k
        ])
        # 通道拼接计算
        cat_channels = c_ * (len(k) + 1)
        self.cv5 = Conv(cat_channels, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        pool_outputs = [m(x1) for m in self.m]
        return self.cv5(torch.cat([x1] + pool_outputs, 1))