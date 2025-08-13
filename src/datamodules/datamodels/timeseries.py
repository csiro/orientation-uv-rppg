import torch
import numpy as np
from enum import Enum
from src.datamodules.datamodels import DataModel
from src.datamodules.datamodels import calculate_length

from typing import *
from numpy import ndarray
from torch import Tensor


class TimeseriesKeys(Enum):
    SPS = "sps"
    TIMESTAMPS = "timestamps"


class TimeseriesModel(DataModel):
    """ Class for interfacing with `Timeseries` data.

    frames (dataset)
        <val> : [T] in the format [time_steps]
        .<attrs>
            fps (int) : Frames per second
            format (str) : Format of the frames e.g. THWC

    NOTE: `VideoFrames` will NEVER cache their results, no reason to do this since the returned
    item has a consistent interface.

    Returns:
        _type_: _description_

    # TODO: Caution we are only handling specific configurations of slicing +int:+int and +int:-int
    """
    def __init__(self, *args, **kwargs) -> None:
        super(TimeseriesModel, self).__init__(*args, **kwargs)

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> ndarray:
        raise NotImplementedError(f"Iterator for `TimeseriesModel` is NOT implemented.")

    @property
    def sps(self) -> float:
        return self.attrs[TimeseriesKeys.SPS.value]
    
    @property
    def timestep(self) -> float:
        return 1 / self.sps
    
    @property
    def length(self) -> int:
        return self.data.shape[0] # Length
        # return calculate_length(self.data.shape[0], self.start, self.stop)
    
    @property
    def time(self) -> float:
        return self.length * self.timestep
    
    # def prepare(self, data: ndarray) -> ndarray:
    #     """ Returns slice of the `VideoFrames` array in default `torchvision` format.

    #     Args:
    #         data (ndarray): _description_

    #     Returns:
    #         ndarray: _description_
    #     """
    #     # Convert to TCHW format
    #     data = np.transpose(data, axes=(self.tdim))

    #     # Convert to contiguous `Tensor`
    #     data = torch.from_numpy(data).contiguous()

    #     return data
 