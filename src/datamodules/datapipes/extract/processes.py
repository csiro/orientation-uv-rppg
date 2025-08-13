import logging 

from typing import *
from src.datamodules.datapipes.extract import TrackedObject

log = logging.getLogger(__name__)


class SelectTrackerID:
    """ Return a single `TrackedObject` based on the provided ID (referencing the ID of a
    tracked object)
    """
    def __init__(self, id: Optional[int] = None) -> None:
        self.id = id

    def __call__(self, detections: List[TrackedObject]) -> List[TrackedObject]:
        if self.id is None:
            self.id = detections[0].id
            log.warning(f"Found {len(detections)} IDs: no ID set, setting to {self.id}")
        return [
            det 
            for det in detections 
            if hasattr(det, "id") and (det.id == self.id)
        ]