from typing import Dict, List
import torch
from src.models import ModelModule
from src.datamodules.datamodels import DataModel
from src.datamodules.datamodels.timeseries import TimeseriesModel

from typing import *
from torch import Tensor


class PhysNet(ModelModule):
    """ LightningModule wrapper for `PhysNet` style models.

    Args:
        ModelModule (_type_): _description_
    """
    def __init__(self, *args, **kwargs) -> None:
        super(PhysNet, self).__init__(*args, **kwargs)

    def forward(self, inputs: Dict[str, DataModel]) -> Dict[str, DataModel]:
        """_summary_

        Args:
            inputs (Dict[str, DataModel]): Frames of shape [B,T,C,H,W]

        Returns:
            Dict[str, DataModel]: Predicted BVP of shape [B,T]
        """
        data = self.network(inputs["frames"].data.permute(0,2,1,3,4)) # BTCHW to BCTHW to convolve over T
        results = {"signal": TimeseriesModel(data=data, attrs=inputs["frames"].attrs)}
        results["signal"].attrs["sps"] = inputs["frames"].attrs["video_sps"]
        return results
