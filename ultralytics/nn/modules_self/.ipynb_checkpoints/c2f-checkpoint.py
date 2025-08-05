
import torch
from torch import nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.SE import SE

class SE_Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.se = SE(c2, 16)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.se(self.cv2(self.cv1(x))) if self.add else self.se(self.cv2(self.cv1(x)))


class C2f_SE(nn.Module):
    def __init__(self, c1, c2, shortcut = False, g = 1, n = 1, e = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(SE_Bottleneck(self.c, self.c, shortcut, g, k=((3,3),(3,3)), e = 1.0) for _ in range(n))


    def forward(self, x):
        y = list(self.cv1(x).chunk(2,1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

