from src.datamodules.datasources.files import DatasetFile
from src.datamodules.datasources.paths.detections import DetectionsSourceFile
from src.datamodules.datasources.files.detections import BoundingBoxesFile, LandmarksFile

from typing import *


class DatasetSample(DatasetFile):
    """ `DatasetSample` provides the interface to interact with `DatasetFile` slices.

    Minimum information necessary to index a slice of the dataset should be stored, 
    
    Accessing overheads e.g I/O should be occured only when required to support lazy loading
    and to reduce the memory consumption overhead per sample.

    """
    def __init__(self, 
        source: str, 
        start: Optional[int] = 0, 
        stop: Optional[int] = None,
        detections: Optional[str] = None,
        *args, **kwargs
    ) -> None:
        super(DatasetSample, self).__init__(source)        
        # Slice bounds
        self.start = start
        self.stop = stop

        # Detections
        self.detections_filename = detections

    def bounding_boxes(self, filename: str, *args, **kwargs) -> BoundingBoxesFile:
        return BoundingBoxesFile(path=self.path.joinpath(filename), *args, **kwargs)

    def landmarks(self, filename: str, *args, **kwargs) -> LandmarksFile:
        return LandmarksFile(path=self.path.joinpath(filename), *args, **kwargs)

    def segmentation_masks(self) -> None:
        raise NotImplementedError(f"`DataModel` for segmentation_masks does not currently exist.")

    def detections(self, filename: Optional[int] = None, *args, **kwargs) -> Any:
        # Select filename to link
        use_filename = filename if filename is not None else self.detections_filename

        # Dynamically return `DetectionModel` according to filename.
        if "landmarks" in use_filename:
            return self.landmarks(use_filename, *args, **kwargs)

        elif "bounding_boxes" in use_filename:
            return self.bounding_boxes(use_filename, *args, **kwargs)

        elif "segmentation_masks" in use_filename:
            return self.segmentation_masks(use_filename, *args, **kwargs)

        else:
            raise ValueError(f"{self.detections} is not a recognized detection type.")