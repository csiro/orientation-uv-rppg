from typing import Any, List, Optional
import cv2
import torch
import urllib
import logging
import numpy as np
from pathlib import Path

from src.datamodules.datapipes.extract import ExtractorModel, InterpolateDetections
from src.datamodules.datamodels.boxes import BoundingBoxesModel
from src.datamodules.datapipes import DistributedDataOperation
from src.datamodules.datasources.paths.detections import DetectionsSourceFile
from src.datamodules.datasources.files.detections import BoundingBoxKeys, BoundingBoxFormats


from typing import *
from numpy import ndarray
from torch import Tensor

log = logging.getLogger(__name__)


class BoundingBoxDetector(ExtractorModel):
    """ Detection class for Bounding Boxes.
    """
    # Default attributes for Bounding Boxes
    detection_type : str = BoundingBoxKeys.TYPE.value
    detection_format : str = BoundingBoxFormats.XYXY.value
    filename: str = DetectionsSourceFile.BOUNDING_BOXES.value

    def __init__(self, *args, **kwargs) -> None:
        super(BoundingBoxDetector, self).__init__(*args, **kwargs)


class HAARCascadeDetector(BoundingBoxDetector):
    """
    """
    # Attributes
    algorithm : str = "haarcascade_frontalface_default"

    # Static
    MODEL_URL = "https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml"

    def __init__(self, path: str, attrs: Optional[Dict[str, Any]] = {}, margs: Optional[Dict[str, Any]] = {}, *args, **kwargs) -> None:
        # Resolve
        path = str(Path(path).absolute().resolve())
        
        # Download HAAR-Cascade model
        if not Path(path).exists():
            _, headers = urllib.request.urlretrieve(self.MODEL_URL, filename=path)

        # Create classifier
        model = cv2.CascadeClassifier(path)
        self.margs = margs

        super(HAARCascadeDetector, self).__init__(model, attrs)

    def preprocess(self, frame: ndarray) -> ndarray:
        """ Convert the frame from CV2 format into the required model format.
        """
        # frame = (255*frame[:,:,::-1]).astype(np.uint8) # [HWC, RGB, 0-1, fp32] -> [HWC, BGR, 0-255, uint8]
        # frame = np.transpose(frame, axes=(1,2,0))
        return frame

    def update(self, frame: ndarray) -> Any:
        boxes = self.model.detectMultiScale(frame, **self.margs) # Box format: [x,y,w,h]
        return boxes

    def postprocess(self, results: Any, frame: ndarray, index: int) -> ndarray:
        """
        """
        outputs = []
        height, width = frame.shape[0], frame.shape[1]

        if type(results) in [tuple]:
            return outputs

        if results.shape[0] < 1:
            log.warning(f"No bounding box found for frame {index}")
            return outputs

        elif results.shape[0] > 1:        
            # Use largest box (could apply tracking?)
            largest_box_idx = np.argmax([box[2] * box[3] for box in results])
            x, y, w, h = tuple(results[largest_box_idx])

        else:
            x, y, w, h = tuple(results[0])

        # Format
        output = np.array([index, -1, x, y, x+w, y+h])
        outputs.append(output)

        return outputs
    

from src.datamodules.datamodels.boxes import BoundingBoxesModel


class InterpolateBoundingBoxes(InterpolateDetections):
    """ Bounding box data format is [[frame, 0, x1, y1, x2, y2], ...] = [N,4]
    """
    detection_model : BoundingBoxesModel = BoundingBoxesModel

    def __init__(self, *args, **kwargs) -> None:
        super(InterpolateBoundingBoxes, self).__init__(*args, **kwargs)

    def update_frame_index(self, data: ndarray, index: int) -> ndarray:
        """ Update the frame index for a single bounding box
        data: [4]
        """
        data[0] = index
        return data
