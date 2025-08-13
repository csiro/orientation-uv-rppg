from enum import Enum

from typing import *


class DatasetModel:
    """ Provides a `DatasetModel` to interact with standard `DatasetFile` items across
    datasets. E.g. detections.

    # TODO: Return standard interfaces to specific datamodels such as landmarks/detections
    """
    def __init__(self, data: Dict[str, Any]) -> None:
        self.data = data


# NOTE: All dataset models 


# from pathlib import Path
# from abc import ABC, abstractmethod
# from functools import cached_property
# from src.datamodules.datamodels import DetectionModel
# from src.datamodules.datamodels.landmarks import LandmarksModel
# from src.datamodules.datamodels.masks import BooleanMaskModel
# from src.datamodules.datamodels.boxes import BoundingBoxesModel

# from typing import *




# # class DatasetModel(ABC):
# #     """_summary_

# #     Args:
# #         BaseModel (_type_): _description_
# #     """
# #     # TODO: Determine better method of splitting data at run-time
# #     stage : Optional[str] = None

# #     # Locations
# #     detections_loc      : str = "/detections"

# #     def __init__(self, data: Any, detections: Optional[str] = None, start: Optional[int] = 0, stop: Optional[int] = None, *args, **kwargs) -> None:
# #         super(DatasetModel, self).__init__(*args, **kwargs)

# #         # Data
# #         self.data = data

# #         # Detection
# #         self._detections = detections
        
# #         # Slicing
# #         self.start = start
# #         self.stop = stop

# #         # Caching
# #         ...

# #     @staticmethod
# #     def location(loc: str) -> Tuple[str, str]:
# #         """ Typically only useful for HDF5 datasets... Maybe shouldn't be here...

# #         Args:
# #             loc (str): _description_

# #         Returns:
# #             Tuple[str, str]: _description_
# #         """
# #         loc = Path(loc)
# #         return str(loc.parent), loc.name
    
# #     @cached_property
# #     def detections(self) -> Any:
# #         assert self._detections is not None, f"Provide a detection key when requesting detections. Available keys are ({list(self.data[self.detections_loc].keys())})"
# #         name = f"{self.detections_loc}/{self._detections}"
# #         assert name in self.data, f"Provided detection key ({name}) does NOT exist in detections. Available keys are ({list(self.data[self.detections_loc].keys())})"
# #         data = self.data[name]
# #         if data.attrs[DetectionModel.type_loc] == "landmarks":
# #             return LandmarksModel(data, start=self.start, stop=self.stop)
# #         elif data.attrs[DetectionModel.type_loc] == "bounding_boxes":
# #             return BoundingBoxesModel(data, start=self.start, stop=self.stop)
# #         elif data.attrs[DetectionModel.type_loc] == "boolean_mask":
# #             return BooleanMaskModel(data, start=self.start, stop=self.stop)
# #         else:
# #             raise ValueError(f"Detection {self._detections} has unsupport type ({data.attrs[DetectionModel.type_loc]}).")
