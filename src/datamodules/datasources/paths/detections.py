from enum import Enum


class DetectionsSourceFile(Enum):
    """
    """
    BOUNDING_BOXES = "bounding_boxes.HDF5"
    LANDMARKS = "landmarks.HDF5"
    SEGMENTATION_MASKS = "segmentation_masks.HDF5"
