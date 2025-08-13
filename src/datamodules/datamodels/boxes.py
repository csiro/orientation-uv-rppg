import torch
import numpy as np
import torchvision
from src.datamodules.datamodels import DetectionModel

from typing import *
from numpy import ndarray
from torch import Tensor


class BoundingBoxesModel(DetectionModel):
    """
    
    bounding_boxes (dataset)
        <val> : [N, 8] in the format
            [
                N = bounding boxes
                8 = [frame_index, object_index, x1, y1, x2, y2, conf, class]
        .<attrs>
            detector: Object detection algorithm.
            tracker: Object tracking algorithm.
    
    """
    # Keys
    format_f : str = "frame_id"
    format_o : str = "object_id"
    format_x1 : str = "x1"
    format_y1 : str = "y1"
    format_x2 : str = "x2"
    format_y2 : str = "y2"

    def __init__(self, *args, **kwargs) -> None:
        super(BoundingBoxesModel, self).__init__(*args, **kwargs)

    @property
    def f_idx(self) -> None:
        self.format.index(self.format_f)

    @property
    def format_x1(self) -> None:
        self.format.index(self.format_x1)

    @property
    def format_y1(self) -> None:
        self.format.index(self.format_y1)

    @property
    def format_x2(self) -> None:
        self.format.index(self.format_x2)

    @property
    def format_y2(self) -> None:
        self.format.index(self.format_y2)

    @property
    def max_length(self) -> int:
        return self.data.shape[0]
    
    @property
    def length(self) -> None:
        start = self.start if self.start is not None else 0
        stop = self.stop if self.stop is not None else self.max_length
        return stop - start
    
    def prepare(self, data: ndarray) -> ndarray:
        data = torch.from_numpy(data).contiguous()

    @staticmethod
    def frame_indexes(data: ndarray) -> ndarray:
        return data[:,0].astype(int) # [FXYZ]
    
    @staticmethod
    def min(data: Tensor) -> Tensor:
        return torch.min(data[:,2:6], dim=1).values

    @staticmethod
    def max(data: Tensor) -> Tensor:
        return torch.max(data[:,2:6], dim=1).values
    
    @staticmethod
    def height(data: Tensor) -> Tensor:
        return data[:,5] - data[:,3] # y2 - y1
    
    @staticmethod
    def width(data: Tensor) -> Tensor:
        return data[:,4] - data[:,2] # x2 - x1
    
    @staticmethod
    def center_x(data: Tensor) -> Tensor:
        return (data[:,2] + data[:,4]) / 2
    
    @staticmethod
    def center_y(data: Tensor) -> Tensor:
        return (data[:,3] + data[:,5]) / 2
    
    @staticmethod
    def center(data: Tensor) -> Tensor:
        cx = BoundingBoxesModel.center_x(data)
        cy = BoundingBoxesModel.center_y(data)
        return torch.cat([cx, cy], dim=-1)
    
    @staticmethod
    def offset(data: Tensor, offset: List[float], idx: Optional[int] = None) -> Tensor:
        """_summary_

        Args:
            data (Tensor): _description_
            offset (List[float]): [x_offset, y_offset]
            idx (Optional[int], optional): _description_. Defaults to None.

        Returns:
            Tensor: _description_
        """
        if idx is None:
            data[:,2] += offset[0] # x1
            data[:,3] += offset[1] # y1
            data[:,4] += offset[0] # x2
            data[:,5] += offset[1] # y2
        else:
            data[idx,2] += offset[0] # x1
            data[idx,3] += offset[1] # y1
            data[idx,4] += offset[0] # x2
            data[idx,5] += offset[1] # y2
        return data
    
    @staticmethod
    def scale(data: Tensor, scale: List[float], idx: Optional[int] = None) -> Tensor:
        """_summary_

        Args:
            data (Tensor): _description_
            scale (List[float]): [x_scale, y_scale]
            idx (Optional[int], optional): _description_. Defaults to None.

        Returns:
            Tensor: _description_
        """
        if idx is None: # Scale all frames.
            data[:,2] = scale[0] * data[:,2] # x1
            data[:,3] = scale[1] * data[:,3] # y1
            data[:,4] = scale[0] * data[:,4] # x2
            data[:,5] = scale[1] * data[:,5] # y2
        else: # Scale a given frame
            data[idx,2] = scale[0] * data[idx,2]
            data[idx,3] = scale[1] * data[idx,3]
            data[idx,4] = scale[0] * data[idx,4]
            data[idx,5] = scale[1] * data[idx,5]
        return data
    
    @staticmethod
    def clip(data: Tensor, size: List[float], idx: Optional[int] = None) -> Tensor:
        """ Clip to the size provided 

        Args:
            data (Tensor): _description_
            size (List[float]): [x_min, y_min, x_max, y_max]
            idx (Optional[int], optional): _description_. Defaults to None.

        Returns:
            Tensor: _description_
        """
        x_min, y_min, x_max, y_max = tuple(size)
        data[:,2:6] = torchvision.ops.clip_boxes_to_image(data[:,2:6], [y_max, x_max])



        # if idx is None:
        #     data[data[:,2] < x_min,2] = x_min


        #     data[data[:,2] < size[0],2] = size[0] # x1 < x_min
        #     data[data[:,2] > size[2],2] = size[2] # x1 > x_max

        #     data[data[:,3] < size[1],3] = size[1] # y1 < y_min
        #     data[data[:,3] > size[3],3] = size[3] # y1 > y_max

        #     data[data[:,4] < size[0],4] = size[0] # x2 < x_min
        #     data[data[:,4] > size[2],4] = size[2] # x2 > x_max

        #     data[data[:,5] < size[1],5] = size[1] # y2 < y_min
        #     data[data[:,5] > size[3],5] = size[3] # y2 > y_max
        # else:
        #     raise NotImplementedError
        return data

    @staticmethod
    def scale_sides(data: Tensor, scale: List[float]) -> Tensor:
        # Compute fixed center point
        cx = BoundingBoxesModel.center_x(data)
        cy = BoundingBoxesModel.center_y(data)

        # Compute scaled width and height deltas
        dw = scale[0] * (BoundingBoxesModel.width(data) / 2)
        dh = scale[1] * (BoundingBoxesModel.height(data) / 2)

        # Convert to area scaled coordinates
        data[:,2] = cx - dw # x1
        data[:,3] = cy - dh # y1
        data[:,4] = cx + dw # x2
        data[:,5] = cy + dh # y2

        return data

    @staticmethod
    def convert_to_bounding_boxes(data: Tensor) -> Tensor:
        return data