import os.path as osp
import copy
import yaml

from rpdiff.utils import path_util

# General config
def load_config(path, default_path=None, demo_train_eval='train'):
    ''' Loads config file.
    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    assert demo_train_eval in ['demo', 'train', 'eval'], f'Invalid value for "demo_train_eval" ({demo_train_eval}) in load_config!'

    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.safe_load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        if demo_train_eval == 'train':
            cfg = load_config(osp.join(path_util.get_train_config_dir(), inherit_from))
        elif demo_train_eval == 'eval':
            cfg = load_config(osp.join(path_util.get_eval_config_dir(), inherit_from))
        elif demo_train_eval == 'demo':
            cfg = load_config(osp.join(path_util.get_demo_config_dir(), inherit_from))
        else:
            raise ValueError(f'Argument "demo_train_eval" must be either "train", "demo", or "eval", value {demo_train_eval} not recognized')
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1: dict, dict2: dict) -> dict:
    ''' Update two config dictionaries recursively.
    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used
    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


class AttrDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__


# def recursive_attr_dict(in_dict):
#     out_dict = {}
#     for v in in_dict.values():
#         if isinstance(v, dict):
#             v = recursive_attr_dict(v)
#     return AttrDict(in_dict)
    

def recursive_attr_dict(in_dict: dict) -> AttrDict:
    out_dict = AttrDict(in_dict)
    for k, v in out_dict.items():
        if isinstance(v, dict):
            out_dict[k] = recursive_attr_dict(v)
    return out_dict
    

def recursive_dict(in_dict: dict) -> dict:
    out_dict = dict(in_dict)
    for k, v in out_dict.items():
        if isinstance(v, dict):
            out_dict[k] = recursive_dict(v)
    return out_dict


def copy_attr_dict(in_dict: dict) -> dict:
    d = copy.deepcopy(recursive_dict(in_dict))
    out_dict = recursive_attr_dict(d)
    return out_dict 
