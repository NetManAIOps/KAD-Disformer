import torch
import numpy as np
import re

def config2str(config, epoch=None):
    str_ls = []

    str_ls.append(config['model']['name'])
    if epoch is not None:
        str_ls.append(f"epoch:{epoch}")

    for key in config['model']:
        if key == 'name' or key == 'device':
            continue
        str_ls.append(f"{key}:{config['model'][key]}")

    for key in config['train']:
        if key == 'epochs':
            continue
        str_ls.append(f"{key}:{config['train'][key]}")

    return '__'.join(str_ls)


def str2config(config_str: str):
    model_config = config_str.split('/')[-1][:-3]
    model_name, parameters = model_config.split('__', maxsplit=1)
    parameters = parameters.split('__')

    config_dict = {}
    config_dict['name'] = model_name

    for param in parameters:
        key, value = param.split(':')
        config_dict[key] = int(value) if value.isdigit() else value

    return config_dict


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True