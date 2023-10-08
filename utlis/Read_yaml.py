
import yaml

# Load config file
def Getyaml(filename='config.yml'):
    with open(filename) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config_dict={}
    config_dict['batch_size'] = config['batch_size']
    config_dict['lr'] = config['lr']
    config_dict['betas0'] = config['betas0']
    config_dict['betas1'] = config['betas1']
    config_dict['step_size'] = config['step_size']
    config_dict['gamma'] = config['gamma']
    config_dict['epochs'] = config['epochs']
    config_dict['save_step'] = config['save_step']
    config_dict['dataset_saved'] = config['dataset_saved']
    config_dict['pretrained'] = config['pretrained']
    config_dict['model_name'] = config['model_name']
    config_dict['train_res_path'] = config['train_res_path']
    config_dict['dataset_path'] = config['dataset_path']
    config_dict['saved_dataset_path'] = config['saved_dataset_path']
    return config_dict