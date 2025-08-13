'''
Object Trackers:
    Single:
        Only track a single object even if multiple are present in the frame; typically work by
        initializing the position of an object in the frame and then tracking it throughout the
        sequence of frames.
            - dlib.correlation_tracker
    Multi:
        Track multiple objects present in a frame.
            - DeepSORT
            - CenterTrack

with Detection:
    Object detector detects objects in the frames and then performs data associated accross frames
    to generate trajectories, hence tracking. Can help track and identify objects even if the
    object detection fails.
without Detection:
    Coordinates are manually initialized and the object is tracker in further frames.

'''
from typing import Any, Dict, List
import numpy as np
import logging
from functools import partial
from abc import ABC, abstractmethod

#
from submodules.yolo_tracking.boxmot import OCSORT

from typing import *
from numpy import ndarray
from src.datamodules.datapipes.extract import BoundingBox, DetectedObject, TrackedObject

log = logging.getLogger(__name__)


class Tracker(ABC):
    """_summary_

    Returns:
        _type_: _description_
    """
    def __init__(self, tracker: Callable, attrs: Optional[Dict[str, Any]] = {}) -> None:
        self.attrs = attrs
        self.tracker = tracker
        self.name = self.__class__.__name__ # Alg. class descriptor

    def __call__(self, detections: List[DetectedObject], detector: str, *args, **kwargs) -> List[TrackedObject]:
        if not len(detections) > 0: return detections
        data = self.preprocess(detections)
        results = self.update(data, *args, **kwargs)
        outputs = self.postprocess(results, detector)
        return outputs
    
    @property
    @abstractmethod
    def attributes(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def preprocess(self, detections: List[DetectedObject]) -> ndarray:
        pass

    @abstractmethod
    def update(self, data: ndarray) -> Any:
        pass

    @abstractmethod
    def postprocess(self, results: Any) -> List[TrackedObject]:
        pass


class OCSORTTracker(Tracker):
    """ Observation-centric SORT

    See: https://github.com/noahcao/OC_SORT

    Args:
        Tracker (_type_): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self, attrs: Optional[Dict[str, Any]] = {}, *args, **kwargs) -> None:
        super(OCSORTTracker, self).__init__(OCSORT(*args, **kwargs), attrs)

    def attributes(self) -> Dict[str, Any]:
        return {**{
            "tracker": self.name
        }, **self.attrs}

    def preprocess(self, detections: List[DetectedObject]) -> ndarray:
        return np.stack([det.object.array for det in detections])
    
    def update(self, data: ndarray, *args, **kwargs) -> List[List[float]]:
        return self.tracker.update(data, None)
    
    def postprocess(self, results: List[List[float]], detector: str) -> List[TrackedObject]:
        outputs = []
        for result in results:
            output = TrackedObject(
                object = BoundingBox(box=result[:4], format="xyxy"),
                detector = detector,
                tracker = self.name,
                id = result[4]
            )
            outputs.append(output)
        return outputs



