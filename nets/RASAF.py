import torch
from torch import nn as nn

class AFF(nn.Module):
  def __init__(self, c, ratio):
    super(AFF, self).__init__()

    self._gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    self.attention_block = nn.Sequential(
        nn.Linear(c, c//ratio),
        nn.ReLU(),
        nn.Linear(c//ratio, c),
        nn.Sigmoid()
    )

  def forward(self, in_cs_l, in_cs_r):
    input_sum = in_cs_l + in_cs_r
    # print(f'AFF_input {input_sum.shape}')
    gaps = self._gap(input_sum).squeeze(-1).squeeze(-1)
    # print(f'AFF_gpas {gaps.shape}')
    weights = self.attention_block(gaps)

    weights = weights.unsqueeze(dim=-1).unsqueeze(dim=-1)
    left_weights = weights
    right_weights = 1-weights

    return left_weights*in_cs_l + right_weights*in_cs_r

class SAFFB(nn.Module):
  def __init__(self, c=16, n=8):
    super(SAFFB, self).__init__()
    self.n = n
    self.c = c
    self.extention = nn.Sequential(
      nn.Conv2d(c, c, 3, padding=1, stride=1),
      nn.ReLU(),
      nn.Conv2d(c, c, 3, padding=1, stride=1),
      nn.Conv2d(c, c*2*n, 1)
    )

    bnches = []
    for _ in range(n):
        bnches.append(AFF(c, 2))
    self.branches = nn.ModuleList(bnches)

  def forward(self, input):
    # print(f'SAFFB_input {input.shape}')
    n2channels = self.extention(input)
    # print(f'SAFFB_n2channels {n2channels.shape}')
    output = self.branches[0](n2channels[:, 0:(0+1)*self.c], n2channels[:, (1)*self.c:(1+1)*self.c])
    for i in range(1, self.n):
      l_idx, r_idx = i*2*self.c, (i*2+1)*self.c
      output += self.branches[i](n2channels[:, l_idx:l_idx+self.c], n2channels[:, r_idx:r_idx+self.c])

    return output + input

class RFAB(nn.Module):
  def __init__(self, c=16):
    super(RFAB, self).__init__()

    self.fst_blk = SAFFB(c, 6)
    self.snd_blk = SAFFB(c, 6)
    self.trd_blk = SAFFB(c, 6)

    self.aggr_conv = nn.Conv2d(3*c, c, 1, padding=0, stride=1)

  def forward(self, input):
    out_mid1 = self.fst_blk(input)
    out_mid2 = self.snd_blk(out_mid1 + input)
    out_mid3 = self.trd_blk(out_mid2 + out_mid1 + input)

    concat_mid4 = torch.cat([out_mid1, out_mid2, out_mid3], axis=1)
    agr_mid5 = self.aggr_conv(concat_mid4)

    return agr_mid5 + input
  




class InitBlock(nn.Module):
  def __init__(self, in_c=3, out_c=16):
    super(InitBlock, self).__init__()
    self.block = nn.Sequential(
      nn.Conv2d(in_c, out_c, 3, padding=1, stride=1),
      nn.Upsample(scale_factor=4, mode='bicubic')
    )

  def forward(self, input_lr):
    return self.block(input_lr)

class DownsamplingBlock(nn.Module): # در هر دو لایه که کانولوشنی که نداریم ساب سمپلینگ درسته؟
  def __init__(self, in_c=16, out_c=32, downscale=2):
    super(DownsamplingBlock, self).__init__()
    self.block = nn.Sequential(
      nn.Conv2d(in_c, out_c, 3, bias=True, stride=1, padding=1),
      nn.LeakyReLU(),
      nn.Conv2d(out_c, out_c, 3, bias=True, stride=2, padding=1),
    )

  def forward(self, fm):
    return self.block(fm)

class UpsamplingBlock(nn.Module):
  def __init__(self, c, t=2, upscale=2):
    super(UpsamplingBlock, self).__init__()
    self.t = t
    blocks = []
    for _ in range(self.t):
      blocks.append(RFAB(c))
    self.RFABs = nn.ModuleList(blocks)
    self.pixelShuffle = nn.PixelShuffle(upscale)

  def forward(self, fm):
    output = fm
    for i in range(0, self.t):
      output = self.RFABs[i](output)
    return self.pixelShuffle(output)

class RASAF(nn.Module):
  def __init__(self):
    super(RASAF, self).__init__()

    self.initBlock = InitBlock(in_c=3, out_c=16)

    self.downsampling_f1 = DownsamplingBlock(in_c=16, out_c=32, downscale=2)
    self.downsampling_f2 = DownsamplingBlock(in_c=32, out_c=64, downscale=2)

    self.upsampling_f3 = UpsamplingBlock(c=64, t=2, upscale=2) # c_out 16
    self.upsampling_f4 = UpsamplingBlock(c=48, t=2, upscale=2) # c_out 12

    self.conv_sr1 = nn.Conv2d(64, 3, 3, padding=1)
    self.conv_sr2 = nn.Conv2d(48, 3, 3, padding=1)
    self.conv_sr3 = nn.Conv2d(28, 3, 3, padding=1)

  def forward(self, input_lr):
    # head
    f0 = self.initBlock(input_lr)
    # print(f'f0: {f0.shape}')

    # downsamplings
    f1 = self.downsampling_f1(f0)
    # print(f'f1: {f1.shape}')

    f2 = self.downsampling_f2(f1)
    # print(f'f2: {f2.shape}')

    # upsamplings
    sr1_in = f2
    f3 = self.upsampling_f3(sr1_in)
    # print(f'f3: {f3.shape}')

    sr2_in = torch.cat([f3, f1], axis=1)
    f4 = self.upsampling_f4(sr2_in)

    # print(f'f4: {f4.shape}')
    sr3_in = torch.cat([f4, f0], axis=1)

    # reconstructions
    sr1_out = self.conv_sr1(sr1_in)
    sr2_out = self.conv_sr2(sr2_in)
    sr3_out = self.conv_sr3(sr3_in)

    return sr1_out, sr2_out, sr3_out