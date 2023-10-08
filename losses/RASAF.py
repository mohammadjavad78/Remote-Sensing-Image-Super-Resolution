import torch
from torch import nn as nn

class RASAFLoss(nn.Module):
    def __init__(self):
        super(RASAFLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, output1,output2,output3, target1,target2,target3, epchs=None):
        w1, w2, w4 = 1, 1, 1
        if (not epchs is None) and epchs > 6:
          w1, w2, w4 = 1, 2, 2

        cost=self.loss(target1,output1)*w1
        cost+=self.loss(target2,output2)*w2
        cost+=self.loss(target3,output3)*w4

        return cost