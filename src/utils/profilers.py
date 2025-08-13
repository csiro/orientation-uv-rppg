'''
PROFILERS


'''
from typing import *
from omegaconf import DictConfig
from hydra.utils import instantiate
from lightning.pytorch.profilers import Profiler

def instantiate_profiler(cfg: DictConfig) -> Profiler:
    return instantiate(cfg)