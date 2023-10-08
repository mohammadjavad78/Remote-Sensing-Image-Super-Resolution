import torch.nn as nn 

class BicubicUpsampling(nn.Module):
    def __init__(self, scale_factor):
        super(BicubicUpsampling, self).__init__()
        self.upsample2 = nn.Upsample(scale_factor=scale_factor//2, mode='bicubic', align_corners=False)
        self.upsample4 = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)

    def forward(self, x):
      return x, self.upsample2(x), self.upsample4(x)