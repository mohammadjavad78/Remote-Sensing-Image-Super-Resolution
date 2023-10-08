import torch
from torch import nn as nn
import matplotlib.pyplot as plt
from utlis.ScoreFunc import *


def plot_image(ax, tensor, caption):
    ax.imshow(tensor)
    ax.tick_params(axis='both', which='both', length=0, width=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel(caption)


def plot_loss_visual(outputs, inputs, idx):
    lv_ls = nn.L1Loss()

    fig = plt.figure(figsize=(10, 6), )
    axs = fig.subplots(2, 3)
    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)

    single_outputs = [output[idx] for output in outputs]
    single_inputs = [input[idx] for input in inputs]
    for idx, (output, input) in enumerate(zip(single_outputs, single_inputs)):
        _, w, h = input.shape
        ls_out = lv_ls(output, input)

        trgt = input.unsqueeze(dim=0)*255

        normed_output = normalize_batch(output)
        ssim, psnr = calc_ssim(normed_output.unsqueeze(dim=0)*255, trgt, False, data_range=255),\
            calc_psnr(normed_output.unsqueeze(dim=0)*255, trgt, data_range=255)

        plot_image(axs[0, idx], input.transpose(2, 0).cpu(), '')
        plot_image(axs[1, idx], normed_output.transpose(2, 0).cpu(),
                   f'({w}, {h})\nls:{ls_out:.5f} psnr:{psnr:.2f} ssim:{ssim:.2f}')

    plt.show()


def show_results(hr, sr, srp, srbic):
    fig = plt.figure(figsize=(16, 5))
    axs = fig.subplots(1, 4)

    size_text = f'({hr.shape[1]}, {hr.shape[2]})'
    plot_image(axs[0], hr.transpose(0, 2), f'Original {size_text}')

    trgt = hr.unsqueeze(dim=0)*255

    normed_output = normalize_batch(srp)
    ssim, psnr = calc_ssim(normed_output.unsqueeze(dim=0)*255, trgt, False, data_range=255),\
        calc_psnr(normed_output.unsqueeze(dim=0)*255, trgt, data_range=255)
    plot_image(axs[1], normed_output.transpose(
        0, 2), f'RASAF+ {psnr:.03f}/{ssim:.03f}')

    normed_output = normalize_batch(sr)
    ssim, psnr = calc_ssim(normed_output.unsqueeze(dim=0)*255, trgt, False, data_range=255),\
        calc_psnr(normed_output.unsqueeze(dim=0)*255, trgt, data_range=255)
    plot_image(axs[2], normed_output.transpose(
        0, 2), f'RASAF {psnr:.03f}/{ssim:.03f}')

    normed_output = normalize_batch(srbic)
    ssim, psnr = calc_ssim(normed_output.unsqueeze(dim=0)*255, trgt, False, data_range=255),\
        calc_psnr(normed_output.unsqueeze(dim=0)*255, trgt, data_range=255)
    plot_image(axs[3], normed_output.transpose(
        0, 2), f'Bicubic {psnr:.03f}/{ssim:.03f}')

    plt.show()

def plot_final_stats(train_stats):
  # np.unravel_index(idx, shape=shape)
  labels = ['train', 'valid']
  fig = plt.figure(figsize=(13, 4))
  axs = fig.subplots(1, len(train_stats)-1)
  x = range(1, train_stats['epoch']+1)
  cnt = 0
  for key, value in train_stats.items():
    if type(value) == list:
      for idx, stat in enumerate(value):
        axs[cnt].plot(x, stat, label=labels[idx])
        axs[cnt].spines[['right', 'top']].set_visible(False)
        axs[cnt].set_xlabel('Epochs')
        axs[cnt].set_ylabel(key)
        axs[cnt].legend()
      cnt+=1
  plt.legend()
  plt.show()

