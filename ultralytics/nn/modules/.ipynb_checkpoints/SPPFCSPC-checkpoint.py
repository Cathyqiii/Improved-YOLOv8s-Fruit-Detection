import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv

class SPPFCSPC(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        if isinstance(k, list):
            k = tuple(k)
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([
            nn.MaxPool2d(kernel_size=x, stride=1, padding=x//2) for x in k
        ])
        cat_channels = c_ * (len(k) + 1)  # 原x1 + 池化层结果
        self.cv5 = Conv(cat_channels, c_, 1, 1)  # 调整为c_，以便与cv2的输出拼接
        self.cv6 = Conv(c_ * 2, c2, 1, 1)  # 最终合并两个分支

    def forward(self, x):
        x1 = self.cv1(x)
        x1 = self.cv3(x1)
        x1 = self.cv4(x1)
        pool_outputs = [m(x1) for m in self.m]
        y1 = self.cv5(torch.cat([x1] + pool_outputs, 1))
        
        x2 = self.cv2(x)
        # 拼接两个分支的结果
        return self.cv6(torch.cat((y1, x2), 1))