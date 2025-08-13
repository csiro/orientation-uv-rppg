import h5py
import numpy as np
from src.datamodules.datasources.files import DatasetFile
from src.datamodules.datamodels import DataModel, LossModel, MetadataModel
from src.datamodules.datamodels import DatasetSampleKeys
from abc import abstractmethod

from typing import *


class DatasetSampleFile(DatasetFile):
    """ Class for interfacing with exported `DatasetSample` results.

    NOTE: Can include any number of `DataModel` style exports.

    """
    def __init__(self, *args, **kwargs) -> None:
        super(DatasetSampleFile, self).__init__(*args, **kwargs)

    def __len__(self) -> int:
        with h5py.File(self.path, "r") as fp:
            keys = fp.keys()
        return len(keys)

    def __iter__(self) -> Generator:
        """ Generator function to iteratively return exported `DataModel` samples from disk.
        """
        with h5py.File(self.path, "r") as fp:
            for sample_key, sample_val in fp.items(): # samples (e.g. 0, 1, 2, ..)
                sample = {}
                for group_key, group_val in sample_val.items(): # groups in sample (e.g. intputs)
                    for entry_key, entry_val in group_val.items(): # entries in groups (e.g. frames)
                        model = DataModel(data=None).load(entry_val, True) # load into `DataModel`
                        sample[f"{group_key}/{entry_key}"] = model
                yield sample

    def data(self):
        raise NotImplementedError

    def _load_group(self, group_key: str) -> List[Dict[str, Any]]:
        with h5py.File(self.path, "r") as fp:
            for sample_key, sample_val in fp.items(): # samples (e.g. 0, 1, 2, ..)
                sample = {}
                if group_key in sample_val.keys():
                    group_val = sample_val[group_key]
                    for entry_key, entry_val in group_val.items():
                        model = MetadataModel(data=None).load(entry_val, True)
                        sample[f"{group_key}/{entry_key}"] = model
                yield sample
                    