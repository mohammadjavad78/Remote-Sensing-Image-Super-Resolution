import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models

from torch import optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler

from dataloaders.DataLoader import *

from utlis.Resultplots import *
from utlis.Read_yaml import *

from train.ResNetTrain import *

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

epochs = 8
save_step = 2
learning_rate = 0.256/32
cl_batch_size = 8
momentum = 0.875
pretrained = True
cl_datatset_saved = False

cl_train_path = f'./SavedDatasets/cl_train_set.tar'
cl_test_path = f'./SavedDatasets/cl_test_set.tar'
cl_model_name = 'classification_train_state_23_class.pt' # train_state.pt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not cl_datatset_saved:
  cl_dataset_train=MyDataset(train=True,
                          test=False,
                          dir=f'./Datasets/datasets',
                          rand=5,
                          output_size=240,
                          justin='UCMerced_LandUse',
                          augh=False)

  cl_dataset_test=MyDataset(train=False,
                        test=True,
                        dir=f'./Datasets/datasets',
                        rand=5,
                        output_size=240,
                        justin='UCMerced_LandUse',
                        augh=False)

  torch.save(cl_dataset_train, cl_train_path)
  torch.save(cl_dataset_test, cl_test_path)

else:
  cl_dataset_train=torch.load(cl_train_path)
  cl_dataset_test= torch.load(cl_test_path)
  

cl_train_loader = DataLoader(cl_dataset_train,batch_size=cl_batch_size,shuffle=True)
cl_test_loader = DataLoader(cl_dataset_test,batch_size=cl_batch_size,shuffle=False)
print(f'len_train_ldr: {len(cl_train_loader)}, len_test_ldr: {len(cl_test_loader)}')

num_classes =  len(cl_dataset_train.labelsdict) # 23
cl_model_2 = models.resnet50(pretrained=True)
cl_model_2.fc = torch.nn.Linear(cl_model_2.fc.in_features, num_classes)
cl_model_2.to(device)

cl_loss = torch.nn.CrossEntropyLoss()
cl_optimizer = optim.SGD(cl_model_2.parameters(), lr=learning_rate, momentum=momentum)
cl_scheduler = lr_scheduler.CosineAnnealingLR(cl_optimizer, T_max=epochs)

if pretrained:
  state = torch.load(f'{train_res_path}/{cl_model_name}', map_location=torch.device(device))
  cl_model_2.load_state_dict(state['model'])

  cl_optimizer.load_state_dict(state['optimizer'])
  cl_scheduler.load_state_dict(state['scheduler'])

  cl_train_stats = state['train_stats']
else:
  cl_train_stats = None
  
# Training
cl_train_stats = training(model=cl_model_2,
                optimizer=cl_optimizer,
                scheduler=cl_scheduler,
                loaders=[cl_train_loader, cl_test_loader],
                criterion=cl_loss,
                epochs=epochs,
                train_stats=cl_train_stats,
                save_step=save_step,
                train_res_path=f'{train_res_path}/{cl_model_name}',
                device=device)

plot_final_stats(cl_train_stats)