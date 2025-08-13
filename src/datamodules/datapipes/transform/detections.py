import torch
import torchvision
import numpy as np
from src.datamodules.datapipes import DatasetOperation

from typing import *


class ClipBoxes(DatasetOperation):
    def __init__(self, fkey: str, dkey: str, *args, **kwargs) -> None:
        super(ClipBoxes, self).__init__(*args, **kwargs)
        self.fkey = fkey
        self.dkey = dkey

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        # retrieve data
        boxes = data[self.dkey].data

        # size
        frame = data[self.fkey].data[0]
        height, width = frame.size(1), frame.size(2)

        #
        data[self.dkey].data = torchvision.ops.clip_boxes_to_image(boxes, [height, width])