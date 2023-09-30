import numpy as np
import logging


def get_logger(name, path):
    logger = logging.getLogger(name)

    if len(logger.handlers) > 0:
        return logger  # Logger already exists

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=path)

    console.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.INFO)

    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger


import yaml
from easydict import EasyDict
import itertools

def flatten_dict(nested_dict, parent_key='', separator='_'):
    items = {}
    for key, value in nested_dict.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, separator))
        else:
            items[new_key] = value
    return items


def get_config(configs:dict)->EasyDict:
    configs = flatten_dict(configs)
    args = EasyDict(configs)
    return args


def get_args_from_yaml(yaml_path):
    with open("train_configs/common_configs.yaml") as f:
        common_configs = yaml.load(f, Loader=yaml.FullLoader)
        common_configs = flatten_dict(common_configs)

    with open(yaml_path) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
        configs = flatten_dict(configs)
    
    #get key from yaml file name 
    configs["key"] = yaml_path.split("/")[-1].split(".")[0]

    for k, v in common_configs.items():
        if k not in configs:
            configs[k] = v

    multiple_config_keys = []
    multiple_config_values = []
    for k, v in configs.items():
        if isinstance(v, list):
            multiple_config_keys.append(k)
            multiple_config_values.append(v)
    
    config_combinations = list(itertools.product(*multiple_config_values))    
    configs_list = []
    for config_combination in config_combinations:
        configs_ = configs.copy()
        for k, v in zip(multiple_config_keys, config_combination):
            configs_[k] = v
        configs_ = EasyDict(configs_)
        configs_list.append(configs_)

    return configs_list


import numpy as np
import torch as th


def evaluate(model, loader, device):
    # Evaluate RMSE
    model.eval()
    mse = 0.0
    for batch in loader:
        with th.no_grad():
            preds = model(batch[0].to(device))
        labels = batch[1].to(device)
        mse += ((preds - labels) ** 2).sum().item()
    mse /= len(loader.dataset)
    return np.sqrt(mse)
