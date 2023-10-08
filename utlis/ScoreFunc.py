import torch
import torch.nn.functional as F
from pytorch_msssim import ssim

def normalize_batch(input):
    if input.ndim == 4:
        batch, channels, h, w = input.shape
        channel_flattened = input.view(batch, channels, h*w)
    elif input.ndim == 3:
        channels, h, w = input.shape
        channel_flattened = input.view(channels, h*w)

    mins = torch.min(channel_flattened, dim=-1, keepdim=True)[0]
    maxes = torch.max(channel_flattened, dim=-1, keepdim=True)[0]

    channel_flattened[...] = (channel_flattened - mins) / (maxes - mins)
    return input

# Target images (Ground Truths) are in range [0, 1]
# but output could have any value float32, no normalization
# should be nomalized
def calc_ssim(out, trgt, normalization=True, data_range=1):
    if normalization:
        normalize_batch(out)
    return ssim(out, trgt, data_range, size_average=True)


def calc_psnr(img1, img2, data_range=1):
    if data_range == 1:
        max_noise = 0
    else:
        max_noise = 20 * \
            torch.log10(torch.tensor([data_range], device=img1.device))
    mse = F.mse_loss(img1, img2)
    psnr = max_noise - 10 * torch.log10(mse)

    return psnr if psnr.ndim == 0 else psnr[0]


def calc_metrics(outputs, inputs, loss):
    hr1, hr2, hr4 = inputs
    ls = loss(outputs[0], outputs[1], outputs[2], hr1, hr2, hr4)

    normed_output = normalize_batch(outputs[2])
    ssim, psnr = calc_ssim(normed_output*255, hr4*255, False, data_range=255),\
        calc_psnr(normed_output*255, hr4*255, data_range=255)

    return ls, ssim, psnr


def calc_acc(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy


def calc_topk_acc(outputs, labels, k):
    _, predicted = torch.topk(outputs, k, dim=1)
    correct = predicted.eq(labels.view(-1, 1).expand_as(predicted)).sum()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy


def get_incorrects(outputs, labels):
    probabilities = F.softmax(outputs, dim=1)
    val, predicted = torch.max(probabilities, 1)
    correct = (predicted == labels)
    return torch.logical_not(correct), correct, predicted, val
