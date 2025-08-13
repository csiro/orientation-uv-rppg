import h5py
import numpy as np
from abc import ABC, abstractmethod

from typing import *
from numpy import ndarray
from torch import Tensor
from h5py._hl.dataset import Dataset as H5PYDataset
from h5py._hl.group import Group as H5PYGroup

    
def calculate_length(max_length: int, start: Optional[int] = 0, stop: Optional[int] = None) -> int:
    """ Calculate the length of the video in frames from the provided indexes.

    Args:
        start (Optional[int], optional): _description_. Defaults to 0.
        stop (Optional[int], optional): _description_. Defaults to None.

    Returns:
        int: _description_
    """
    start = start if start is not None else 0
    stop = stop

    # Handle underflow of indexing
    if (stop is not None) and (stop < 0): stop += max_length
    if (start is not None) and (start < 0): start += max_length
    
    # return computed length
    return stop - start if stop is not None else max_length - start


class DataModel(ABC):
    """
    """
    def __init__(self, data: ndarray, format: Optional[str] = None, attrs: Optional[Dict[str, Any]] = {}) -> None:
        self.data = data
        self.format = format
        self.attrs = attrs

    def pin_memory(self) -> None:
        """ Pin `DataModel` items to paged-memory
        """
        # Pin `data to in-page memory
        if isinstance(self.data, Tensor):
            self.data = self.data.pin_memory()

        # Pin `attrs` to in-page memory
        for key, val in self.attrs.items():
            if isinstance(val, Tensor):
                self.attrs[key] = val.pin_memory()
        return self

    def to(self, device: Any) -> None:
        """ Transfer `DataModel` items to `device`.
        """
        # Move `data` to device
        if isinstance(self.data, Tensor):
            self.data = self.data.to(device)

        # Move `attrs` to device
        for key, val in self.attrs.items():
            if isinstance(val, Tensor):
                self.attrs[key] = val.to(device)
        return self

    def dump(self, file_grp: H5PYGroup, name: str, write_data: Optional[bool] = True, index: Optional[int] = None) -> None:
        """ Export `DataModel` items to disk.
        """
        # validate
        assert isinstance(file_grp, H5PYGroup), f"Provide a `h5py` group to export the dataset to."
        assert "/" not in name, f"Provided name ({name}) must not contain relative path."

        # export data
        if write_data:
            # format data for export
            if isinstance(self.data, Tensor):
                if len(self.data.shape) > 0:
                    data : Tensor = self.data[index] if index is not None else self.data
                    data : ndarray = data.detach().cpu().numpy()
                else:
                    data : ndarray = self.data.item() # loss typically (0-dim tensor)
                
            elif isinstance(self.data, ndarray):
                data : ndarray = self.data[index] if index is not None else self.data

            elif isinstance(self.data, list):
                if len(self.data) > 0:
                    data : List = self.data[index] if index is not None else self.data
                    data : ndarray = np.array(data)
                else:
                    data : List[bool] = [False] # empty data

            else:
                raise ValueError(f"Cannot export data type ({type(self.data)})")

            # create dataset with data
            ds = file_grp.create_dataset(name, data=data)
        else:
            # create dataset with invalid data
            ds = file_grp.create_dataset(name, data=[False])

        # format attributes for export
        for key, val in self.attrs.items():
            # if indexable
            if any([isinstance(val, dtype) for dtype in [list, ndarray, Tensor]]):
                if index is not None:
                    val = val[index]
            else:
                val = str(val) # convert to str as back up

            # format data
            if isinstance(val, Tensor):
                val : ndarray = val.detach().cpu().numpy()

            # create attr in dataset
            ds.attrs.create(key.replace("/", "_"), val)


    def load(self, file_ds: H5PYDataset, load_state_dict: Optional[bool] = False) -> Union[Any, Tuple[ndarray, Dict[str, Any]]]:
        """ Load `DataModel` items from disk.
        """
        # validate
        assert isinstance(file_ds, H5PYDataset), f"Provide a `h5py` dataset to load into the `DataModel`."

        # load data from dataset
        data : ndarray = file_ds[:] if len(file_ds.shape) > 0 else file_ds[()] # slice vs. scalar ds

        # load attrs from dataset
        attrs : Dict[str, Any] = {k: v for (k, v) in file_ds.attrs.items()}

        # conditionally update self : can just return (use to load into another model type e.g. timeseries)
        if load_state_dict:
            self.data = data
            self.attrs = attrs
            return self
        else:
            return data, attrs

    
class DetectionModel(DataModel):
    """
    We really only do this because we want a unified way of dealing with different types of
    detection outputs within consistent interfaces, such as with
    
    Consider we want to define a bounding box, we can defined this from `BoundingBox`,
    `Landmarks`, or `SegmentationMasks` in different ways: many-to-one mapping.

    Down-side of this is since we interface with the underlying data through the wrappers
    property and bound methods we need to maintain an internal state of the data during the
    processes being applied to the data.

    NOTE: `Detection` objects WILL cache their results internally : you should only
    ever interface with a detection object through its `Model`.
    """
    def __init__(self, *args, **kwargs) -> None:
        super(DetectionModel, self).__init__(*args, **kwargs)

    @property
    @abstractmethod
    def frame_indexes(self) -> ndarray:
        pass


class MetadataModel(DataModel):
    def __init__(self, *args, **kwargs) -> None:
        super(MetadataModel, self).__init__(*args, **kwargs)


class LossModel(DataModel):
    def __init__(self, *args, **kwargs) -> None:
        super(LossModel, self).__init__(*args, **kwargs)

    @property
    def group(self) -> str:
        return "/".join(self.attrs["name"].split("/")[:-1])

    @property
    def loss(self) -> str:
        return self.attrs["name"].split("/")[-1]


from enum import Enum

class DatasetSampleKeys(Enum):
    # source
    SOURCE = "source"

    # data
    INPUTS = "inputs"
    TARGETS = "targets"

    # results
    OUTPUTS = "outputs"
    PREDICTIONS = "predictions"
    
    # summary
    LOSSES = "losses"
    METRICS = "metrics"


def stack_datamodels(models: Iterable) -> DataModel:
    data = []
    attrs = {}

    # accumulate results
    for model in models:
        data.append(model.data)
        for key, val in model.attrs.items():
            if key in attrs:
                attrs[key].append(val)
            else:
                attrs[key] = [val]
    
    # stack results
    data = np.stack(data, axis=0)
    for key, val in attrs.items():
        attrs[key] = np.stack(val, axis=0)

    return DataModel(data=data, attrs=attrs)


def unstack_datamodel(model: DataModel) -> List[DataModel]:
    models = []

    for idx in range(model.data.shape[0]):
        # extract data
        data = model.data[idx]
        attrs = {k: v[idx] for (k, v) in model.attrs.items()}

        # accumulate models
        models.append(DataModel(data=data, attrs=attrs))

    return models