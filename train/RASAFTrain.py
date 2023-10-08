from fastprogress import master_bar, progress_bar
from utlis.ScoreFunc import *
import numpy as np

def batch_iter(model, optimizer, scheduler, input, criterion, device):
  hr1 = input[0].to(device)
  hr2 = input[1].to(device)
  hr4 = input[2].to(device)

  out = model(hr1)
  loss = criterion(out[0], out[1], out[2], hr1, hr2, hr4, glob_epoch)

  ssim_mes = calc_ssim(out=out[2], trgt=hr4)
  psnr_mes = calc_psnr(out[2], hr4)

  return loss, ssim_mes, psnr_mes

def train_iter(model, optimizer, scheduler, input, criterion, device):
  optimizer.zero_grad()

  loss, ssim_mes, psnr_mes = batch_iter(model, optimizer, scheduler, input, criterion, device)

  loss.backward()
  # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

  optimizer.step()

  return loss, ssim_mes, psnr_mes

def dataset_iter(model, optimizer, scheduler, criterion, ldr, btch_iter, prnt_bar, device):
  accum_epch_ls = 0
  accum_epch_psnr = 0
  accum_epch_ssim = 0

  cnt = 0
  for batch_input in progress_bar(ldr, parent=prnt_bar):

    btch_ls, ssim_mes, psnr_mes = btch_iter(model, optimizer, scheduler, batch_input, criterion, device)

    accum_epch_ls += btch_ls.item()
    accum_epch_psnr += psnr_mes.item()
    accum_epch_ssim += ssim_mes.item()
    cnt += 1

    prnt_bar.child.comment = f'bt_ls: {btch_ls:.3f}, ' +\
                              f'mn_ls: {(accum_epch_ls/cnt):.3f}, ' +\
                              f'mn_ssim: {(accum_epch_ssim/cnt):.3f}, '+\
                              f'mn_psnr: {(accum_epch_psnr/cnt):.3f}  '

  return accum_epch_ls/len(ldr), accum_epch_ssim/len(ldr), accum_epch_psnr/len(ldr)

def plot_loss_update(epoch, epochs, mb, train_loss, valid_loss):
  if len(train_loss) != 0:
    x = range(1, epoch+1)
    y = np.concatenate((train_loss, valid_loss))
    graphs = [[x,train_loss], [x,valid_loss]]
    x_margin = 0.2
    y_margin = 0.05
    x_bounds = [1-x_margin, epochs+x_margin]
    y_bounds = [np.min(y)-y_margin, np.max(y)+y_margin]
    mb.update_graph(graphs, x_bounds, y_bounds)

def save_state(model, optimizer, scheduler, epoch, losses, ssims, psnrs, train_res_path):
  state = {
    'train_stats': {
        'epoch': epoch,
        'losses': losses,
        'ssims': ssims,
        'psnrs': psnrs
    },
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict()
  }

  torch.save(state, train_res_path)
  return state['train_stats']

def training(model, optimizer, scheduler, loaders, criterion, epochs, train_stats, save_step, train_res_path, device):
  train_loader = loaders[0]
  test_loader = loaders[1]

  train_loss, test_loss = ([], []) if train_stats is None else train_stats['losses']
  train_ssim, test_ssim = ([], []) if train_stats is None else train_stats['ssims']
  train_psnr, test_psnr = ([], []) if train_stats is None else train_stats['psnrs']
  last_epoch = 0 if train_stats is None else train_stats['epoch']

  epchs_mstr_bar = master_bar(range(last_epoch+1, epochs+1))
  plot_loss_update(last_epoch, epochs, epchs_mstr_bar, train_loss, test_loss)

  global glob_epoch
  for epoch in epchs_mstr_bar:
    glob_epoch = epoch

    # Iterate through train data set
    model.train()
    epch_trn_ls, epch_trn_ssim, epch_trn_psnr = dataset_iter(model, optimizer, scheduler, criterion,
                                train_loader, train_iter, epchs_mstr_bar, device)
    scheduler.step()
    train_loss.append(epch_trn_ls)
    train_ssim.append(epch_trn_ssim)
    train_psnr.append(epch_trn_psnr)

    # Iterate through test data set
    model.eval()
    with torch.no_grad(): # If not using 'with no_grads' then overflow comes after
      epch_tst_ls, epch_tst_ssim, epch_tst_psnr = dataset_iter(model, optimizer, scheduler, criterion, test_loader, batch_iter, epchs_mstr_bar, device)
      test_loss.append(epch_tst_ls)
      test_ssim.append(epch_tst_ssim)
      test_psnr.append(epch_tst_psnr)

    # Finally update learning plot
    plot_loss_update(epoch, epochs, epchs_mstr_bar, train_loss, test_loss)
    epchs_mstr_bar.write(f'train( loss; {epch_trn_ls:.3f}, ssim; {epch_trn_ssim:.3f}, psnr; {epch_trn_psnr:.3f} ), ' +\
                          f'test( loss: {epch_tst_ls:.3f}, ssim; {epch_tst_ssim:.3f}, psnr; {epch_tst_psnr:.3f} )')

    if epoch % save_step == 0:
        train_stats = save_state(model, optimizer, scheduler, epoch,\
                                [train_loss, test_loss],\
                                [train_ssim, test_ssim],\
                                [train_psnr, test_psnr],\
                                train_res_path)
  return train_stats