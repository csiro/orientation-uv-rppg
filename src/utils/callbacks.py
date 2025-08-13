'''
CALLBACKS


'''
from typing import *
from omegaconf import DictConfig
from hydra.utils import instantiate
from lightning import Callback

def instantiate_callbacks(cfg: DictConfig) -> Dict[str, Callback]:
    return {key: instantiate(_cfg) for (key, _cfg) in cfg.items()}