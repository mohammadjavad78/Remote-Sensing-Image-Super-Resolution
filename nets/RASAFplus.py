import torch
from torch import nn as nn
import matplotlib.pyplot as plt
from utlis.ScoreFunc import *


def rotat(input, k):
  return torch.stack([torch.rot90(channel, k) for channel in input])

def flip(input):
  return torch.stack([torch.flip(channel, dims=[0]) for channel in input])

def show_eights(sr4s):
  for sr4 in sr4s:
    plt.imshow(sr4.transpose(0, 2))
    plt.show()

def mutate_8s(input):
  inputs = [input[0], flip(input[0])]

  for i in range(1, 4):
    mutat_inpt = rotat(input[0], i)
    flpd_mutat_inpt = flip(mutat_inpt)
    inputs.extend([mutat_inpt, flpd_mutat_inpt])

  return torch.stack(inputs)

def revert_mutats_8s(input):
  inverteds = [input[0], flip(input[1])]

  for i in range(1, 4):
    invert_rot = rotat(input[i*2], -1*i)
    invert_flip = flip(input[i*2+1])
    invert_flip_rot = rotat(invert_flip, -1*i)
    inverteds.extend([invert_rot, invert_flip_rot])

  return torch.stack(inverteds)

class RASAF_plus(nn.Module):
  def __init__(self, model):
    super(RASAF_plus, self).__init__()
    self.model = model

  def _forward_single(self, input):
    inputs = mutate_8s(input)
    sr1s, sr2s, sr4s = self.model(inputs)

    # If you want to show results of all rotated and fliped inputs
    # show_eights(sr4s)

    # Invert
    revrts_4s = revert_mutats_8s(sr4s)
    revrts_2s = revert_mutats_8s(sr2s)
    revrts_1s = revert_mutats_8s(sr1s)

    # show_eights(revrts_2s)

    # Ensembling (using mean of eights)
    sr1, sr2, sr4 = torch.mean(revrts_1s, dim=0),\
                    torch.mean(revrts_2s, dim=0),\
                    torch.mean(revrts_4s, dim=0)
    return sr1, sr2, sr4

  def forward(self, input):
    if input.ndim == 4:
      sr1s, sr2s, sr4s = [], [], []
      for idx in range(input.shape[0]):
        sr1, sr2, sr4 = self._forward_single(input[idx].unsqueeze(dim=0))
        sr1s.append(sr1); sr2s.append(sr2); sr4s.append(sr4)

      return torch.stack(sr1s, dim=0),\
               torch.stack(sr2s, dim=0),\
               torch.stack(sr4s, dim=0)

    else:
      return self._forward_single(input)

