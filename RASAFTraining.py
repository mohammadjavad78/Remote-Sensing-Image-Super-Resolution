import torch 

import torch.optim.lr_scheduler as schedule

from torch import optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler

from dataloaders.DataLoader import *

from utlis.Resultplots import *
from utlis.Read_yaml import *
from losses.RASAF import *

from train.RASAFTrain import *

from nets.RASAFplus import *
from nets.RASAF import *
from nets.Bicubic import *

yamls=Getyaml()
batch_size=yamls['batch_size']
lr=yamls['lr']
betas0=yamls['betas0']
betas1=yamls['betas1']
step_size=yamls['step_size']
gamma=yamls['gamma']
epochs=yamls['epochs'] 
save_step=yamls['save_step'] 
dataset_saved=yamls['dataset_saved'] 
pretrained=yamls['pretrained'] 
model_name=yamls['model_name'] 
train_res_path=yamls['train_res_path']
dataset_path=yamls['dataset_path']
saved_dataset_path=yamls['saved_dataset_path']

if not dataset_saved:
  dataset_train=MyDataset(
                    train=True,
                    test=False,
                    dir=dataset_path,
                    rand=5)

  dataset_test=MyDataset(
                    train=False,
                    test=True,
                    dir=dataset_path,
                    rand=5)

  torch.save(dataset_train, f'{saved_dataset_path}/train_set.tar')
  torch.save(dataset_test, f'{saved_dataset_path}/test_set.tar')

else:
  dataset_train = torch.load(f'{saved_dataset_path}/train_set.tar')
  dataset_test = torch.load(f'{saved_dataset_path}/test_set.tar')


train_loader = DataLoader(dataset_train,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset_test,batch_size=batch_size,shuffle=True)
print(f'{len(train_loader)} {len(test_loader)}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RASAF().to(device)
loss = RASAFLoss()
optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(betas0,betas1)) # 0.01, 0.005 -> loss explosion
scheduler = schedule.StepLR(optim, step_size=step_size, gamma=gamma)
if pretrained:
  state = torch.load(f'{train_res_path}/{model_name}', map_location=torch.device(device))
  model.load_state_dict(state['model'])

  optim.load_state_dict(state['optimizer'])
  scheduler.load_state_dict(state['scheduler'])

  train_stats = state['train_stats']
else:
  train_stats = None



# Training
train_stats = training(model=model,
                optimizer=optim,
                scheduler=scheduler,
                loaders=[train_loader, test_loader],
                criterion=loss,
                epochs=epochs,
                train_stats=train_stats,
                save_step=save_step,
                train_res_path=f'{train_res_path}/{model_name}',
                device=device)

# Plot final stats
plot_final_stats(train_stats)