from nets.RASAFplus import *
from utlis.Resultplots import *
from utlis.Resultplots import *

import torchvision.transforms as transforms
import PIL.Image as Image

from utlis.Resultplots import * 

def visual_model_comparison(dataloader, sr_model, bic_model, rasafp_model):
    rasaf_model = sr_model.cpu()
    rasafp_model = RASAF_plus(rasaf_model)

    cnt = 0
    num_samples = 100
    for input in dataloader:
        hr1, _, hr4, _ = input

        with torch.no_grad():
            sr1p, sr2p, sr4p = rasafp_model(hr1)
            sr1, sr2, sr4 = sr_model(hr1)
            bicsr1, bicsr2, bicsr4 = bic_model(hr1)

            # show_results(hr1[0], sr1[0], sr1p)

            show_results(hr4[0], sr4[0], sr4p[0], bicsr4[0])

        cnt += 1
        if cnt >= num_samples:
            break


def objetive_model_comparison(dataloader, sr_model, bic_model, rasafp_model, device):
    sr_model.eval()

    rasaf_model = sr_model.cpu()
    rasafp_model = RASAF_plus(rasaf_model)

    stats = []
    for input in dataloader:
        hr1 = input[0].to(device)
        hr2 = input[1].to(device)
        hr4 = input[2].to(device)

        inp = [hr1, hr2, hr4]

        with torch.no_grad():
            sr1p, sr2p, sr4p = rasafp_model(hr1)
            sr1, sr2, sr4 = rasaf_model(hr1)
            bicsr1, bicsr2, bicsr4 = bic_model(hr1)

            rsf_mtrs = calc_metrics([sr1, sr2, sr4], inp)
            rsfp_mtrs = calc_metrics([sr1p, sr2p, sr4p], inp)
            bic_mtrs = calc_metrics([bicsr1, bicsr2, bicsr4], inp)

            stats.append([[*rsf_mtrs], [*rsfp_mtrs], [*bic_mtrs]])

        sts_tens = torch.tensor(stats)
        torch.mean(sts_tens, dim=0)


def loss_visualization(loader, device, model):
    model.eval()

    cnt = 0
    num_samples = 10
    with torch.no_grad():
        for input in loader:
            rand_idx = torch.randint(0, input[0].shape[0], (1, ))[0].item()

            hr1 = input[0].to(device)
            hr2 = input[1].to(device)
            hr4 = input[2].to(device)

            outputs = model(hr1)

            plot_loss_visual(outputs, [hr1, hr2, hr4], rand_idx)

            cnt += 1
            if cnt >= num_samples:
                break


def paper_patch_results(dataset_UCMerced_path, model):
    model.eval()

    hr4_size = 64
    images = [{'address': f'{dataset_UCMerced_path}/airplane/airplane81.tif', 'up_left': (65, 62), 'size': hr4_size},
              {'address': f'{dataset_UCMerced_path}/tenniscourt/tenniscourt71.tif', 'up_left': (124, 99), 'size': hr4_size}]

    cnt = 0
    num_samples = 10
    with torch.no_grad():
        resize_mode = transforms.InterpolationMode.BICUBIC
        hr1_resizer = transforms.Resize(
            (hr4_size//4, hr4_size//4), interpolation=resize_mode)
        # hr2_resizer = transforms.Resize((hr4_size//4, hr4_size//4), interpolation = resize_mode)

        for image in images:
            to_tensor = transforms.ToTensor()
            img = to_tensor(Image.open(image['address']))

            tl_x, tl_y = image['up_left']
            hr4 = img[:, tl_x:tl_x+hr4_size, tl_y:tl_y+hr4_size]
            hr1 = hr1_resizer(hr4)

            sr1, _, sr4 = model(hr1.unsqueeze(dim=0))

            plt.imshow(hr1.squeeze().transpose(0, 1).transpose(1, 2))
            plt.show()

            plt.imshow(sr4.squeeze().transpose(0, 1).transpose(1, 2))
            plt.show()

            plt.imshow(hr4.squeeze().transpose(0, 1).transpose(1, 2))
            plt.show()

            cnt += 1
            if cnt >= num_samples:
                break


def plot_incorrect_predictions(sr_out, hr_out, tgts, sr, hr, lr, lbl_dict):
    sr_inc, sr_ct, sr_pdcts, sr_vals = get_incorrects(sr_out, tgts)
    hr_inc, hr_cr, hr_pdcts, hr_vals = get_incorrects(hr_out, tgts)
    hr_cr_sr_inc = torch.logical_and(sr_inc, hr_cr)
    sr_incs, hr_crs, cr_hr_pdcts, inc_sr_pdcts, cr_hr_vals, inc_sr_vals, lrs\
        = sr[hr_cr_sr_inc], hr[hr_cr_sr_inc], hr_pdcts[hr_cr_sr_inc],\
        sr_pdcts[hr_cr_sr_inc], hr_vals[hr_cr_sr_inc], sr_vals[hr_cr_sr_inc],\
        lr[hr_cr_sr_inc]

    cnt = torch.count_nonzero(hr_cr_sr_inc)
    if cnt != 0:

        for idx in range(cnt):
            fig = plt.figure(figsize=(12, 5))
            axs = fig.subplots(1, 3)

            trgt = hr_crs[idx].unsqueeze(dim=0)*255
            hr_cl, hr_conf = lbl_dict[cr_hr_pdcts[idx].item()], cr_hr_vals[idx]
            plot_image(axs[0], hr_crs[idx].transpose(0, 2).cpu(),
                       f'Ref cls: {hr_cl}, cnf: {hr_conf:.2f}')

            plot_image(axs[1], lrs[idx].transpose(0, 2).cpu(), f'LR input')

            normed_output_sr = normalize_batch(sr_incs[idx])
            ssim, psnr = calc_ssim(normed_output_sr.unsqueeze(dim=0)*255, trgt, False, data_range=255),\
                calc_psnr(normed_output_sr.unsqueeze(
                    dim=0)*255, trgt, data_range=255)

            sr_cl, sr_conf = lbl_dict[inc_sr_pdcts[idx].item(
            )], inc_sr_vals[idx]
            plot_image(axs[2], normed_output_sr.transpose(0, 2).cpu(
            ), f'SR out cls: {sr_cl}, cnf: {sr_conf:.2f}, {psnr:.2f}/{ssim:.2f}')
            plt.show()


def evaluate_classification(cl_model, sr_model, ldr, device, datset, label=None):
    cl_loss = torch.nn.CrossEntropyLoss()

    hr_acc_top3_acm = 0
    sr_acc_top3_acm = 0
    hr_acc_acm = 0
    sr_acc_acm = 0

    inverse_labelsdict = {v: k for k, v in datset.labelsdict.items()}

    cnt = 0
    with torch.no_grad():
        for dat in ldr:
            tgts = dat[3].to(device)
            hr = dat[2].to(device)
            lr = dat[0].to(device)

            if not label is None:

                in_cls = tgts == label
                tgts = tgts[in_cls]
                hr = hr[in_cls]
                lr = lr[in_cls]

                if torch.numel(lr) == 0:
                    continue

                cnt += 1
                sr = sr_model(lr)[2]

                hr_out = cl_model(hr)
                hr_ls = cl_loss(hr_out, tgts)

                sr_out = cl_model(sr)
                sr_ls = cl_loss(sr_out, tgts)

                hr_acc_top3_acm += calc_topk_acc(hr_out, tgts, k=3)
                sr_acc_top3_acm += calc_topk_acc(sr_out, tgts, k=3)
                hr_acc_acm += calc_acc(hr_out, tgts)
                sr_acc_acm += calc_acc(sr_out, tgts)

                plot_incorrect_predictions(
                    sr_out, hr_out, tgts, sr, hr, lr, inverse_labelsdict)

        print(f'top3_hr: {hr_acc_top3_acm/cnt}, ' +
              f'top3_sr: {sr_acc_top3_acm/cnt}, ' +
              f'acc_hr: {hr_acc_acm/cnt}, ' +
              f'acc_sr: {sr_acc_acm/cnt} ')


def class_based_classification_evaluation(ldr, dataset, cl_model, sr_model, device):
    classes = ['airplane', 'mobilehomepark', 'runway', 'river']
    num_clses = [dataset.labelsdict[labl] for labl in classes]
    for idx, num_cls in enumerate(num_clses):
        print(f'------ {classes[idx]} ------')

        evaluate_classification(cl_model, sr_model, ldr,
                                device, dataset, num_cls)
