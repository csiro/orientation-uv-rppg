from typing import Optional
import cv2
import numpy as np
from src.datamodules.datapipes import DatasetOperation

from typing import *


class Tensor2Frames(DatasetOperation):
    def __init__(self, fkey: str, *args, **kwargs) -> None:
        super(Tensor2Frames, self).__init__(*args, **kwargs)
        self.fkey = fkey

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        """ Convert the input frames in `Tensor` [0-1] [T,H,W,C] format into `ndarray` [0-255]

        Args:
            data (Dict[str, Any]): _description_
        """
        data[self.fkey].data = (255 * data[self.fkey].data).permute(0,2,3,1).numpy().astype(np.uint8)


class OverlayDetections(DatasetOperation):
    """_summary_

    Args:
        DatasetOperation (_type_): _description_
    """
    def __init__(self, point_kwargs: Dict[str, Any], fkey: str, dkey: str, *args, **kwargs) -> None:
        super(OverlayDetections, self).__init__(*args, **kwargs)
        self.point_kwargs = point_kwargs
        self.fkey = fkey
        self.dkey = dkey

    def apply(self, data: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        """ Convert the input frames 

        Args:
            data (Dict[str, Any]): _description_
        """
        data[self.fkey].data = data[self.dkey].interface.overlay(
            data[self.fkey].data.copy(), 
            data[self.dkey].data, 
            **self.point_kwargs
        )