import h5py
import numpy as np
from enum import Enum
from src.datamodules.datasources.files import DatasetFile
from src.datamodules.datasources.paths.detections import DetectionsSourceFile
from src.datamodules.datamodels.boxes import BoundingBoxesModel
from src.datamodules.datamodels.landmarks import LandmarksModel
from abc import abstractmethod

from typing import *
from numpy import ndarray


class DetectionKeys(Enum):
    # BOUNDING_BOXES = "" # Defined by the key provided
    TYPE = "detection_type"
    ALGORITHM = "algorithm"
    FORMAT = "detection_format"
    FILENAME = "filename"
    CREATED = "created"


class DetectionsFile(DatasetFile):
    def __init__(self, *args, **kwargs) -> None:
        super(DetectionsFile, self).__init__(*args, **kwargs)

    def data(self, start: Optional[int] = 0, stop: Optional[int] = None) -> ndarray:
        """
        NOTE: DetectionModel results are associated with a given frame e.g. bounding boxes, landmarks,
        and segmentation masks. Obtained start/stop indexes are the respective frame indexes, hence the
        retrieve method should determine the suitable subset to use rather than just indexing as this
        may be invalid.
        """
        with h5py.File(self.path, "r") as fp:
            # Dataset
            ds = fp["data"]

            # Frame indexes
            frame_indexes = self.frame_indexes(ds)
            frame_mask = (frame_indexes >= start) & (frame_indexes < stop) if stop is not None else (frame_indexes >= start)

            # Slice data (I/O)
            data = ds[frame_mask] 
            data = data.astype(np.float32)

            # Attributes : Load ALL available
            attrs = {key: val for key, val in ds.attrs.items()}

        return data, attrs
    
    def detections(self, start: Optional[int] = 0, stop: Optional[int] = None) -> ndarray:
        with h5py.File(self.path, "r") as fp:
            # Dataset
            ds = fp["data"]

            # Frame indexes
            frame_indexes = self.frame_indexes(ds)
            frame_mask = (frame_indexes >= start) & (frame_indexes < stop) if stop is not None else (frame_indexes >= start)

            # Slice data (I/O)
            data = ds[frame_mask] 
            data = data.astype(np.float32)
        return data

    @property
    def algorithm(self) -> str:
        with h5py.File(self.path, "r") as fp:
            val = fp["data"].attrs[DetectionKeys.ALGORITHM.value]
        return val
    
    @property
    def format(self) -> str:
        with h5py.File(self.path, "r") as fp:
            val = fp["data"].attrs[DetectionKeys.FORMAT.value]
        return val
    
    @property
    def type(self) -> str:
        with h5py.File(self.path, "r") as fp:
            val = fp["data"].attrs[DetectionKeys.TYPE.value]
        return val

    @property
    def created_time(self) -> str:
        with h5py.File(self.path, "r") as fp:
            val = fp["data"].attrs[DetectionKeys.CREATED.value]
        return val

    @abstractmethod
    def frame_indexes(self, data: ndarray) -> ndarray:
        pass

    

class BoundingBoxKeys(Enum):
    TYPE = "bounding_boxes"
    FILENAME = DetectionsSourceFile.BOUNDING_BOXES.value


class BoundingBoxFormats(Enum):
    XYXY = ["frame_id", "object_id", "x1", "y1", "x2", "y2"]
    XYWH = ["frame_id", "object_id", "x1", "y1", "w", "h"]
    CXCYWH = ["frame_id", "object_id", "cx", "cy", "w", "h"]


class BoundingBoxesFile(DetectionsFile):
    """ 

    All `BoundingBoxes` files contain the following structure:
        <root>/bounding_boxes.HDF5
            <algorithm_0>
            ...
            <algorithm_N>

    """
    def __init__(self, *args, **kwargs) -> None:
        super(BoundingBoxesFile, self).__init__(*args, **kwargs)

    def data(self, start: Optional[int] = 0, stop: Optional[int] = None) -> ndarray:
        # Extract data and attributes
        data, attrs = super(BoundingBoxesFile, self).data(start, stop)

        # Extract data format
        format = attrs[DetectionKeys.FORMAT.value]

        return BoundingBoxesModel(data=data, format=format, attrs=attrs)

    def detections(self, start: Optional[int] = 0, stop: Optional[int] = None) -> BoundingBoxesModel:
        boxes = super(BoundingBoxesFile, self).detections(start, stop)
        boxes_format = self.format()
        return BoundingBoxesModel(data=boxes, format=boxes_format)

    def frame_indexes(self, data: ndarray) -> ndarray:
        return BoundingBoxesModel.frame_indexes(data)
    

class LandmarksKeys(Enum):
    TYPE = "landmarks"
    FILENAME = DetectionsSourceFile.LANDMARKS.value


class LandmarksFormats(Enum):
    """ Format of the last dimension.

    All `Landmarks` have the following shape: [N, 478, 4]

    """
    XYZ = ["frame_id", "x", "y", "z"]


class LandmarksFile(DetectionsFile):
    """
    
    All `Landmarks` files contain the following structure:
        <root>/landmarks.HDF5
            <algorithm_0>
            ...
            <algorithm_N>
    
    """
    def __init__(self, *args, **kwargs) -> None:
        super(LandmarksFile, self).__init__(*args, **kwargs)

    def data(self, start: Optional[int] = 0, stop: Optional[int] = None) -> ndarray:
        # Extract data and attributes
        data, attrs = super(LandmarksFile, self).data(start, stop)

        # Extract data format
        format = attrs[DetectionKeys.FORMAT.value]

        return LandmarksModel(data=data, format=format, attrs=attrs)

    def detections(self, start: Optional[int] = 0, stop: Optional[int] = None) -> LandmarksModel:
        landmarks = super(LandmarksFile, self).detections(start, stop)
        landmarks_format = self.format()
        return LandmarksModel(data=landmarks, format=landmarks_format)

    def frame_indexes(self, data: ndarray) -> ndarray:
        return LandmarksModel.frame_indexes(data)


class KeyPoints(DatasetFile):
    pass


class SegmentationMasks(DatasetFile):
    pass





# class DetectionModel(DataModel):
#     """_summary_



#     Args:
#         DatasetModel (_type_): _description_
#     """
#     # Keys
#     type_loc : str = "type"
#     name_loc : str = "name"
#     format_loc : str = "format"
#     algorithm_loc : str = "algorithm"
#     created_loc : str = "create"

#     def __init__(self, *args, **kwargs) -> None:
#         super(DetectionModel, self).__init__(*args, **kwargs)
    
#     @property
#     def retrieve(self) -> ndarray:
#         """ Access a slice of the underlying dataset and conditionally cache the result.

#         

#         Returns:
#             ndarray: _description_
#         """
#         