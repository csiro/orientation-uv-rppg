import logging
from pathlib import Path
from functools import partial
from src.datamodules.datasets import Dataset
from src.datamodules.datasources.mmpd import MMPD_DatasetSample
from src.datamodules.datasources.files.mmpd import MMPD_Video, MMPD_BVP, MMPD_Metadata

from typing import *
from src.datamodules.datamodels.frames import VideoFramesModel
from src.datamodules.datamodels.timeseries import TimeseriesModel
from src.datamodules.datamodels import DetectionModel

from src.datamodules.datasources.files.mmpd import MMPD_Keys

log = logging.getLogger(__name__)


class MMPDDataset(Dataset):
    """ Mobile Muti-domain Physiological Dataset

    Links:
        DOI: doi.org/10.48550/arXiv.2302.03840
        Available at: https://github.com/McJackTang/MMPD_rPPG_dataset
    
    """
    name : str = "MMPD"

    def __init__(self, *args, **kwargs) -> None:
        super(MMPDDataset, self).__init__(*args, **kwargs)
