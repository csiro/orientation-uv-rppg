# class SegmentationMasks(Detection):
#     """

#     segmentation_masks (dataset)
#         <val> : [T, H, W, M] in same format as the input images, except
#             [
#                 T = num frames
#                 H, W = height & width of image
#                 M = mask channels
#             ]
#         .<attrs>
#             detector: Segmentation mask algorithm pipeline
#             channels: {
#                 semantic_channel_name (str) : mask_channel_index (int)
#             }
#             boxes (str): Key for the corresponding tracked bounding boxes (unique objects)

#     NOTE: Segmentation masks may also have a corresponding set of tracked bounding boxes
#     per person. 
    
#     # TODO: Determine linkage between `SegmentationMasks` and `BoundingBoxes`
#     """
#     def __init__(self, *args, **kwargs) -> None:
#         super(SegmentationMasks, self).__init__(*args, **kwargs)


import cv2
import torch
import logging
import numpy as np
from scipy.spatial import ConvexHull
import torchvision.transforms.functional as tv_f
from src.datamodules.datamodels import DetectionModel

from typing import *
from numpy import ndarray
from torch import Tensor

log = logging.getLogger(__name__)


class BooleanMaskModel(DetectionModel):
    """
    
    # TODO: Add computations for min/max/scale/offset etc.

    Args:
        DetectionModel (_type_): _description_
    """
    def __init__(self, *args, **kwargs) -> None:
        super(BooleanMaskModel, self).__init__(*args, **kwargs)

    def prepare(self, data: ndarray) -> ndarray:
        """_summary_

        Args:
            data (ndarray): _description_

        Returns:
            ndarray: _description_
        """
        # Convert to `Tensor` in contiguous format
        data = torch.from_numpy(data).contiguous()

        return data

    @staticmethod
    def scale(data: Tensor, size: int) -> Tensor:
        data = tv_f.resize(img=data, size=(size[0], size[1]), antialias=None)
        return data
    
    # @staticmethod
    # def min(data: Tensor) -> Tensor:
