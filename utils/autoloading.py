"""
source: Latent Diffusion Model source code
"""

import importlib
from omegaconf import OmegaConf
import lightning as L
import os
import glob
import torch


def get_obj_from_str(string, reload=False):
    """
    instantiate an object from a string module path
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    """
    instantiate an object from a config
    target: the module path to instantiate the object from
    params: the params to provide to the object constructor
    """
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_model(dir_trained_model: str, checkpoint: str):
    """
    dir_trained_model: The Directory path of the trained model. It where his config and weights are located.
    checkpoint: The Checkpoint filename of the trained model.
    """
    config_model = OmegaConf.load(glob.glob(os.path.join(dir_trained_model, 'configs', '*-project.yaml'))[0])
    model: L.LightningModule = get_obj_from_str(config_model.model.target)(**config_model.model.params)
    checkpoint = torch.load(os.path.join(dir_trained_model, 'checkpoints', checkpoint))
    model.load_weights(checkpoint['state_dict'])
    return model
