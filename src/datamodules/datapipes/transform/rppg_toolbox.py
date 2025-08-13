import torch
import numpy as np
import torchvision
from src.datamodules.datapipes import DatasetOperation


from typing import *
from numpy import ndarray
from torch import Tensor



class DiffNormalizeFrames(DatasetOperation):
    """ Perform 1-st order frame difference.
    """
    def __init__(self, key: str, *args, **kwargs) -> None:
        super(DiffNormalizeFrames, self).__init__(*args, **kwargs)
        self.key = key
    
    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        frames = data[self.key].data

        # compute frame difference
        new_frames = torch.zeros(size=(frames.shape))
        for idx in range(frames.shape[0]-1):
            new_frames[idx+1] = (frames[idx+1] - frames[idx]) / (frames[idx+1] + frames[idx] + 1e-9)
        
        # normalize data
        new_frames = new_frames / torch.std(new_frames)

        # handle nan
        new_frames[torch.isnan(new_frames)] = 0

        # re-assign
        data[self.key].data = new_frames


class DiffNormalizeLabels(DatasetOperation):
    """
    """
    def __init__(self, key: str, *args, **kwargs) -> None:
        super(DiffNormalizeLabels, self).__init__(*args, **kwargs)
        self.key = key
    
    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        labels = data[self.key].data

        # compute frame difference
        labels = torch.diff(labels, dim=0, prepend=labels[0].unsqueeze(0))

        # normalize data
        labels = labels / torch.std(labels)

        # handle nan
        labels[torch.isnan(labels)] = 0

        # re-assign
        data[self.key].data = labels