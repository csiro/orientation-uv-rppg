import cv2
import torch
import numpy as np
from enum import Enum
from src.datamodules.datamodels import DataModel
from src.datamodules.datamodels import calculate_length

from typing import *
from numpy import ndarray
from torch import Tensor


class FramesFormat(Enum):
    TORCHVISION = "torchvision"
    OPENCV = "opencv"
    

class FramesKeys(Enum):
    VIDEO_FPS = "video_sps"
    VIDEO_LENGTH = "video_length"
    TIMESTAMPS = "TIMESTAMPS"


class VideoFramesModel(DataModel):
    """ `DataModel` to handle `VideoFrames` data.

    Frames MUST always be provided in the default `torchvision` format of [TCHW] [RGB] 
    though the data type [UINT8/FP32] and scale [0-255/0-1] may vary

    frames (dataset)
        <val> : [T, H, W, C] in the format [time_steps, height, width, channels]
        .<attrs>
            fps (int) : Frames per second
            format (str) : Format of the frames e.g. THWC

    # TODO: Caution we are only handling specific configurations of slicing +int:+int and +int:-int
    """
    def __init__(self, *args, **kwargs) -> None:
        super(VideoFramesModel, self).__init__(*args, **kwargs)

    def __len__(self) -> int:
        return self.length
    
    def __iter__(self) -> Generator:
        idx = self.start if self.start is not None else 0
        stop = idx + len(self) # incorporates stop
        while True:
            try:
                if idx >= stop: raise StopIteration
                frame = self.data[idx]
                idx += 1
                yield frame
            except StopIteration:
                break

    @property
    def sps(self) -> float:
        return self.attrs[FramesKeys.VIDEO_FPS.value]

    @property
    def fps(self) -> float: # ALIAS for `sps`
        return self.sps
    
    @property
    def timestep(self) -> float:
        return 1/self.sps
    
    @property
    def tdim(self) -> int:
        return self.format.upper().find(self.format_tdim.upper())
    
    @property
    def length(self) -> int:
        return calculate_length(self.max_length, self.start, self.stop)
    
    @property
    def max_length(self) -> int:
        return self.attrs[FramesKeys.VIDEO_LENGTH.value] # Maximum length (not of slice)
    
    @property
    def time(self) -> float:
        return self.length * self.timestep
    
    @property
    def width(self) -> int: 
        return self.data.shape[3 if self.format == FramesFormat.TORCHVISION.value else 2] # [TCHW] else [THWC]
    
    @property
    def height(self) -> int:
        return self.data.shape[2 if self.format == FramesFormat.TORCHVISION.value else 1] # [TCHW] else [THWC]
    
    @property
    def channels(self) -> int:
        return self.data.shape[1 if self.format == FramesFormat.TORCHVISION.value else 3] # [TCHW] else [THWC]
    
    @property
    def isColor(self) -> bool:
        return self.channels == 3

    def prepare(self) -> None:
        """ Returns slice of the `VideoFrames` array in default `torchvision` format.

        SAVED IN: CHW > [0-255] > UINT8 > BGR
        DEFAULT: RGB > FP32 [0-1] > HWC

        Args:
            data (ndarray): _description_

        Returns:
            ndarray: _description_
        """
        self.to_torchvision() # default: torchvision
        

    def format(self, format: str) -> None:
        """ Convert from `torchvision` [TCHW] [RGB] to `opencv` [THWC] [BGR]
        """
        if format == FramesFormat.TORCHVISION.value: # cv2 to torchvision
            self.to_torchvision()
        
        elif format == FramesFormat.OPENCV.value: # torchvision to cv2
            self.to_opencv()    
        
        else:
            raise ValueError(f"Provide one of [torchvision, opencv]")
        
    def to_opencv(self) -> None:
        if self.format == FramesFormat.TORCHVISION.value:
            self.data : ndarray = np.transpose(self.data, axes=(0,2,3,1)) # TCHW to THWC
            self.data : ndarray = self.data[:,:,:,::-1] # RGB to BGR
            self.format : str = FramesFormat.OPENCV.value # CV to TV

    def to_torchvision(self) -> None:
        if self.format == FramesFormat.OPENCV.value:
            self.data : ndarray = self.data[:,:,:,::-1] # BGR to RGB
            self.data : ndarray = np.transpose(self.data, axes=(0,3,1,2)) # THWC to TCHW
            self.format : str = FramesFormat.TORCHVISION.value # TV to CV
