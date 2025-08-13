import copy
import torch
import numpy as np
from src.datamodules.datamodels import DataModel
from src.datamodules.datapipes import DatasetOperation
from abc import abstractmethod

from typing import *
from numpy import ndarray
from torch import Tensor


class ConvertDataModel(DatasetOperation):
    """ Convert `DataModel` into another `DataModel` e.g. `TimeSeriesModel`.
    """
    def __init__(self, key: str, model: Callable, *args, **kwargs) -> None:
        super(ConvertDataModel, self).__init__(*args, **kwargs)
        self.key = key
        self.model = model

    def apply(self, data: Dict[str, Any]) -> None:
        data[self.key] = self.model(data=data[self.key].data, attrs=data[self.key].attrs)


class ToPrepared(DatasetOperation):
    """ Perform in-place conversion of `DataModel` data to default format for training.
    """
    def __init__(self, *args, **kwargs) -> None:
        super(ToPrepared, self).__init__(*args, **kwargs)
        
    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        for key, val in data.items():
            if hasattr(val, "prepare"):
                val.prepare() # convert into default format


class ToTensor(DatasetOperation):
    """ Perform in-place conversion of ndarrays in `DataModel`s to Tensors.
    """
    def __init__(self, exclude_keys: Optional[List[str]] = [], *args, **kwargs) -> None:
        super(ToTensor, self).__init__(*args, **kwargs)
        self.exclude_keys = exclude_keys

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        for key, val in data.items():
            if key not in self.exclude_keys:
                if isinstance(val, DataModel):
                    if isinstance(val.data, ndarray): # skip pre-defined tensors
                        val.data = torch.from_numpy(val.data.copy()).contiguous() # convert from numpy


class DetachTensor(DatasetOperation):
    def __init__(self, keys: List[str], *args, **kwargs) -> None:
        super(DetachTensor, self).__init__(*args, **kwargs)
        self.keys = keys

    def apply(self, data: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        for key in self.keys:
            data[key].data = data[key].data.detach()


class MoveTensorToCpu(DatasetOperation):
    """ Move `DataSample` from primary device (determined by .data device) to the
        CPU and store the original device.
    """
    def __init__(self, keys: List[str], *args, **kwargs) -> None:
        super(MoveTensorToCpu, self).__init__(*args, **kwargs)
        self.keys = keys

    def apply(self, data: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        for key in self.keys:
            data[key].attrs["device"] = data[key].data.device # store original device for later use
            data[key] = data[key].to(torch.device("cpu"))


class MoveTensorToDevice(DatasetOperation):
    """ Move `Tensor` to device defined in the `DataModel`.
    """
    def __init__(self, keys: List[str], *args, **kwargs) -> None:
        super(MoveTensorToDevice, self).__init__(*args, **kwargs)
        self.keys = keys

    def apply(self, data: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        for key in self.keys:
            if "device" in data[key].attrs:
                device = data[key].attrs["device"]
                data[key] = data[key].to(device)


class CopyKey(DatasetOperation):
    """ Copy `DataModel` into a new key in the `DataSample`.

    NOTE: Moving items between devices before copying is really only an issue when we 
    want to copy output items from the graph which we want to compute metrics on, since
    they will typically be stored on a GPU.

    NOTE: You will need to move data from the `device` to the `cpu` before copying
    if the data exists on a device, and then move it back to the device.
    """
    def __init__(self, old_key: str, new_key: str, *args, **kwargs) -> None:
        super(CopyKey, self).__init__(*args, **kwargs)
        self.old_key = old_key
        self.new_key = new_key

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> Any:
        data[self.new_key] = copy.deepcopy(data[self.old_key]) # perform copy


class MoveKey(DatasetOperation):
    """ Move `DataModel` from one key to another WITHOUT retaining the old key.
    """
    def __init__(self, old_key: str, new_key: str, *args, **kwargs) -> None:
        super(MoveKey, self).__init__(*args, **kwargs)
        self.old_key = old_key
        self.new_key = new_key

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> Any:
        data[self.new_key] = copy.deepcopy(data[self.old_key])
        del data[self.old_key]


class DeleteKeys(DatasetOperation):
    """ Delete a `DataModel` from the `DataSample`.
    """
    def __init__(self, keys: Union[str, List[str]], *args, **kwargs) -> None:
        super(DeleteKeys, self).__init__(*args, **kwargs)
        self.keys = [keys] if type(keys) == str else keys

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> Any:
        for key in self.keys:
            del data[key]


class Difference(DatasetOperation):
    """ Compute the n-th order difference of the target label signal.

    signal[i] = signal[i+1] - signal[i]

    """
    def __init__(self, key: str, order: Optional[int] = 1, prepend: Optional[bool] = None, *args, **kwargs) -> None:
        super(Difference, self).__init__(*args, **kwargs)
        self.key = key
        self.order = order
        self.prepend = prepend

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> Any:
        # kwargs = ({"prepend": data[self.key].data[0]} if self.prepend else {"append": data[self.key].data[-1]}) if self.prepend is not None else {}
        _data = data[self.key].data
        if self.prepend is not None:
            _data = torch.cat([_data[0].unsqueeze(0), _data] if self.prepend else [_data, _data[0].unsqueeze(0)], dim=0) 
        
        data[self.key].data = torch.diff(_data, n=self.order, dim=0, **kwargs)


class Normalize(DatasetOperation):
    """ Normalize the frames by minimum/maximum.

    frame[i] = (frame[i] - min) / (max - min)
    
    """
    def __init__(self, key: str, mode: str, *args, **kwargs) -> None:
        super(Normalize, self).__init__(*args, **kwargs)
        self.key = key
        self.mode = mode.lower()
        assert self.mode in ["minmax", "std"]

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> Any:
        if self.mode == "minmax":
            minimum = torch.min(data[self.key].data).values
            maximum = torch.max(data[self.key].data).values
            data[self.key].data = (data[self.key].data - minimum) / (maximum - minimum)
        elif self.mode == "std":
            std = torch.std(data[self.key].data) # population correction
            data[self.key].data = data[self.key].data / std
        else:
            pass
        data[self.key].data[torch.isnan(data[self.key].data)] = .0


class Standardize(DatasetOperation):
    """ Standardize the frames such that mean=0, var=1.
    """
    def __init__(self, key: str, *args, **kwargs) -> None:
        super(Standardize, self).__init__(*args, **kwargs)
        self.key = key

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> Any:
        std = torch.std(data[self.key].data)
        mean = torch.mean(data[self.key].data)
        data[self.key].data = (data[self.key].data - mean) / std
        data[self.key].data[torch.isnan(data[self.key].data)] = .0


class NormalizedDifference(DatasetOperation):
    """ Compute the normalized frame difference between frames.
    
    frame[i] = (frame[i+1] - frame[i]) / (frame[i+1] + frame[i])
    
    """
    def __init__(self, key: str, prepend: Optional[bool] = None, *args, **kwargs) -> None:
        super(NormalizedDifference, self).__init__(*args, **kwargs)
        self.key = key
        self.prepend = prepend

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        frames = data[self.key].data
        # frames = (torch.cat([frames[0], frames]) if self.prepend else torch.cat([frames, frames[0]])) if self.prepend is not None else frames
        
        if self.prepend is not None:
            frames = torch.cat([frames[0].unsqueeze(0), frames] if self.prepend else [frames, frames[0].unsqueeze(0)], dim=0) 
        
        # data[self.key].data = torch.diff(frames, n=self.order, dim=0, **kwargs)
        
        for idx in range(frames.shape[0] -1):
            frames[idx] = (frames[idx+1] - frames[idx]) / (frames[idx+1] + frames[idx])
        frames[torch.isnan(frames)] = .0
        data[self.key].data = frames[:-1]



