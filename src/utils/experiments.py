import logging
import omegaconf
from omegaconf import OmegaConf
from abc import ABC, abstractmethod

from typing import *
from omegaconf import DictConfig





# class ExperimentManager(ABC):
#     """_summary_

#     Class defines specific logic relating to handling of the configuration file to perform
#     dependent resolution of different parts of the name-space and merging of independently
#     define namespaces to allow for modularization of the configuration file than what `hydra`
#     will allow with the strict hierarchy.

#     MODEL
#         MODEL_SPECIFIC_OUTPUT_POSTPROCESSING
        
#         MODEL_SPECIFIC_DATASET_PROCESSING
#             MODEL_SPECIFIC_DATASET_INPUT_PROCESSING
#             MODEL_SPECIFIC_DATASET_TARGET_PROCESSING

#         DATASET_SPECIFIC_PROCESSING

#         EXPERIMENT_SPECIFIC_PROCESSING
#             EXPERIMENT_SPECIFIC_CONSTRUCTION
#             EXPERIMENT_SPECIFIC_FILTERING
#             EXPERIMENT_SPECIFIC_SAMPLING 

#     EXPERIMENT : Combination of...
#         MODEL
#         DATASET
#         PROCESSING
#         EVALUATION
            
#     Args:
#         ABC (_type_): _description_
#     """
#     def __init__(self, cfg: DictConfig, *args, **kwargs) -> None:
#         super(ExperimentManager, self).__init__(*args, **kwargs)
        
