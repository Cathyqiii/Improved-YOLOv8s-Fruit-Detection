import torch
import torch.nn as nn
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, c1, reduction=16):  # 参数名改为c1
        super().__init__()
        # 动态计算中间通道数
        self.mid_channels = max(8, c1 // reduction)  # 使用c1
        
        # 通道校验
        assert c1 >= self.mid_channels, \
            f"Input channels ({c1}) must >= mid_channels ({self.mid_channels})"  # 变量名修正

        # 通道注意力组件
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        # 特征转换层
        self.channel_conv = nn.Sequential(
            nn.Conv2d(c1, self.mid_channels, 1, bias=False),  # 使用c1
            nn.BatchNorm2d(self.mid_channels),
            h_swish()
        )
        
        # 空间注意力层
        self.spatial_conv_h = nn.Conv2d(self.mid_channels, c1, 1)  # 使用c1
        self.spatial_conv_w = nn.Conv2d(self.mid_channels, c1, 1)  # 使用c1

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        # 自动获取输出通道数
        if c != self.spatial_conv_h.out_channels:
            raise ValueError(f"Input channels mismatch: expected {self.spatial_conv_h.out_channels}, got {c}")

        # 高度和宽度注意力
        x_h = self.pool_h(x)  # [n,c,h,1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [n,c,w,1]
        
        # 特征融合与转换
        combined = torch.cat([x_h, x_w], dim=2)  # [n,c,h+w,1]
        transformed = self.channel_conv(combined)
        
        # 分离空间分量
        x_h, x_w = torch.split(transformed, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # 恢复维度顺序

        # 生成注意力图
        att_h = self.spatial_conv_h(x_h).sigmoid()
        att_w = self.spatial_conv_w(x_w).sigmoid()

        return identity * att_h * att_w