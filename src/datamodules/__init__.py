import copy
import torch
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from abc import ABC, abstractmethod

from typing import *


class DataModule(LightningDataModule):
    """

    What splits did you use?
    What transforms did you use?
    What normalization did you use?
    What preparation/tokenization did you use?

    NOTE: If you need information from the datasets to build your model, then run `prepare_data` and
    `setup` manually (lightning will ensure the methods run on the correct devices).

    NOTE: `DataModules` support hyperparameters with the same API as `LightningModules`.

    # TODO: Validate whether BytesIO object can be pickled for checkpointing resuming...
    
    """
    def __init__(self, dataset: Callable, dataloaders: Dict[str, Callable], name: Optional[str] = None) -> None:
        super(DataModule, self).__init__()

        self.name = name
        self.dataset = dataset
        self.dataloaders = DataLoaders(dataloaders)

    def prepare_data(self) -> None:
        '''
        Allows for safe preparation of the data on a single CPU process, as downloading and saving
        data with multiple processes (distributed setting) can result in corrupted data. 

        Utilize this function to download data to disk, tokenize the data, etc.

        However, there are often cases, where you will want to perform heavy pre-processing of the
        data before creating the datasets. In this case it is recommended to utilize feature stores
        and specific pre-processing jobs to reduce the train-time processing overhead. Often the case
        where you cannot fit the data into memory e.g. tokenizing a large dataset, when streaming of
        the data is required.

        NOTE: `setup` is called after `prepare_data` and there is a barrier between them to ensure
        synchronization across all processes.

        WARNING: `prepare_data` is called from a single process on the CPU, as such, setting states 
        here is NOT recommended, as it will not be reflected across all processes on all nodes.
        '''
        pass

    def setup(self, stage: Optional[str] = None, *args, **kwargs) -> None:
        '''
        Perform data operations on every GPU such as
            counting the number of classes
            building the vocabulary
            performing train/val/test splits
            creating datasets
            applying transforms
            etc.

        NOTE: `setup` is called from every process across all the nodes, it is recommended to set
        states here.

        NOTE: As stochastic processes such as sampling/indexing are done on construction we create a copy to ensure a
        valid copy of each dataset.

        # TODO: Current implementation will create 3 different h5py file references X num processes : may cause issues???
        '''        
        assert stage in ["fit", "test", None], f"Provided stage={stage} must be either (fit/test for training) or (None for inference)"

        # Fitting Stage
        if stage in ["fit"]:
            if not hasattr(self, "train"):
                self.train = copy.deepcopy(self.dataset)
                self.train.filter(stage="train")
            if not hasattr(self, "valid"):
                self.valid = copy.deepcopy(self.dataset)
                self.valid.filter(stage="valid")

        # Testing Stage
        if stage in ["test"]:
            if not hasattr(self, "test"):
                self.test = copy.deepcopy(self.dataset)
                self.test.filter(stage="test")

        if stage is None:
            if not hasattr(self, "data"):
                self.all = copy.deepcopy(self.dataset)


    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        '''
        Wrap the dataset defined in `setup` with a `DataLoader` which will be used during the
        `trainer.fit` method.

        DataLoader's represent Python iterables over a dataset with support for
            map/iterable-style datasets
            data loading order
            automatic batching
            single/multi-process data loading
            memory pinning

        NOTE: In PyTorch you should use `DistributedSampler` to ensure samples are not replicated
        across devices with `DistributedDataParallel` to ensure each accelerator see's an appropriate
        part of your data. Lightning will automatically add the correct samplers when required.

        If `use_distributed_sampler` is `False` then will wrap the returned `DataLoader` sampler 
        `Sampler` with `DistributedSampler` automatically when required. By default will add 
        `shuffle=True` for train sampler and `shuffle=False` for validation/test/predict sampler. 

        Can override with `false` and add your own distributed sampler into the dataloader hooks, if
        `true` but already added, will NOT override the existing one.

        For iterable-style datasets NONE of the wrapping will be done.

        NOTE: You can disable the automatic `DistributedSampler` with `Trainer(replace_sampler_ddp=False)`
        if desired.

        NOTE: By default will define `shuffle=True` for training and `shuffle=False` for val/test
        sampling, and `drop_last` will be set to the default of PyTorch.

        NOTE: If you called `seed_everything` then Lightning will have assigned the seed of the RNG
        generator provided to the sampler.

        '''
        assert hasattr(self, "train"), f"DataLoader for stage (train) has not been initialized."
        return self.dataloaders.stage(self.train, "train", *args, **kwargs)
    
    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        assert hasattr(self, "valid"), f"DataLoader for stage (valid) has not been initialized."
        return self.dataloaders.stage(self.valid, "valid", *args, **kwargs)
    
    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        assert hasattr(self, "test"), f"DataLoader for stage (test) has not been initialized."
        return self.dataloaders.stage(self.test, "test", *args, **kwargs)
    
    def predict_dataloader(self, *args, **kwargs) -> DataLoader:
        """_summary_

        Returns:
            DataLoader: _description_
        """
        assert hasattr(self, "all"), f"DataLoader for stage (predict) has not been initialized."
        return self.dataloaders.stage(self.all, "predict", *args, **kwargs)

    def teardown(self, stage: str) -> None:
        """
        Clean up when run is finished

        NOTE: `teardown` is called from every process across all the nodes, it is recommended to
        clean up the state here.
        """
        pass

    def information(self, *args, **kwargs) -> Dict[str, Any]:
        """ Compute DataModule specific information for logging and instantiation purposes, may
        include information such as 'num_classes' for classification tasks or `num_conditions` for
        other group-level population metrcis.

        Returns:
            Dict[str, Any]: _description_
        """
        attributes = {}
        stages = {
            "train": "fit",
            "test": "test"
        }
        
        for stage, dl_stage in stages.items():
            # Perform `DataLoader` setup for the stage
            self.setup(dl_stage)

            # Validate whether specified stage has data
            if len(getattr(self, stage)) > 0:

                # Access `Dataset` attributes
                attrs = getattr(self, stage).attributes(*args, **kwargs)

                # Update extracted attributes
                for key, val in attrs.items():
                    attributes[f"{key}_{stage}"] = val

        # hacky workaround
        if "length_train" not in attributes:
            # log.warning(f"Training length not in attributes, assigning to be 0 as a workaround")
            attributes["length_train"] = 0

        return attributes

    def transfer_batch_to_device(self, batch: Any, device: Any, dataloader_idx: Optional[int] = 0) -> Any:
        """ Transfer custom batch type `BatchEntry` to device.
        """
        if isinstance(batch, Batch):
            batch.to(device)
        else:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch


class DataLoaders:
    """ Factory object for creating PyTorch `DataLoader` for a required stage.

    Allows for configuration defined instantiation across different stages.

    """
    def __init__(self, dataloaders: Dict[str, Any]) -> None:
        self.dataloaders = dataloaders

    def stage(self, data: Any, stage: str, *args, **kwargs) -> DataLoader:
        if stage in self.dataloaders.keys():
            return self.dataloaders[stage](data, *args, **kwargs)
        raise ValueError(f"{stage} is not an available `DataLoader` ({self.dataloaders.keys()})")


from torch import Tensor
from src.datamodules.datamodels import DataModel



class Batch:
    """ Custom `Batch` class to interface with `DataPipe` outputs.
    """
    transfer_keys: List[str] = ["inputs", "targets"] # pin/transfer

    def __init__(self, batch: List[Any], only_collate_tensor_attributes: Optional[bool] = False) -> None:
        self.collated_batch = self.collate(batch, only_collate_tensor_attributes)
    
    def collate(self, batch: List[Dict[str, Any]], only_collate_tensor_attributes: Optional[bool] = False):
        """ Collate the `batch` items

        """
        # Accumulate entries across samples into nested dictionary structure as per below
        ''' 
        We obtain the following input structure
        [
            {
                "inputs/frames": `DataModel` : .data (Tensor) & .attrs (Dict[str,Any]),
                ...
            }
        ]
        
        We format into the following output structure e.g.
        {
            "inputs": {
                "frames": `DataModel`: torch.Tensor([T,C,H,W]),
                ...
            }
            "targets": {...}
            "source": {...}
        }
        '''
        collated_batch = {}
        for sample in batch:
            for name, entry in sample.items():
                if "/" in name:
                    dest, key = tuple(name.split("/"))
                    
                    if dest not in collated_batch: # create nested dict
                        collated_batch[dest] = {}

                    if key not in collated_batch[dest]: 
                        collated_batch[dest][key] = DataModel(data=[], attrs={}) # generic container

                    if hasattr(entry, "data"): # append data to container
                        if isinstance(entry.data, Tensor):
                            collated_batch[dest][key].data.append(entry.data)

                    if hasattr(entry, "attrs"): # append attrs to container
                        for k, v in entry.attrs.items():
                            if only_collate_tensor_attributes:
                                if not isinstance(v, Tensor): continue
                            if k in collated_batch[dest][key].attrs.keys():
                                collated_batch[dest][key].attrs[k].append(v)
                            else:
                                collated_batch[dest][key].attrs[k] = [v]
        
        # Stack `List` of items into a single `Tensor` if possible
        for _, src in collated_batch.items():
            for key, val in src.items():
                if len(val.data) > 0: # stack tensor data
                    val.data = torch.stack(val.data, dim=0).contiguous()
                for k, v in val.attrs.items(): # stack tensor attrs
                    if len(v) > 0:
                        if isinstance(v[0], Tensor):
                            val.attrs[k] = torch.stack(v, dim=0).contiguous()

        # Define un-packed data to be returned : Dict[str, DataModel(data,attrs)]
        collated_batch["batch_size"] = len(batch)

        return collated_batch

    def pin_memory(self):
        """ Perform custom memory pinning of inputs/targets data (conditionally tensors)
        """
        for transfer_key in self.transfer_keys:
            for key, val in self.collated_batch[transfer_key].items():
                # Pin `DataModel`
                val = val.pin_memory()
        return self
        
    def to(self, device: Any) -> None:
        """ Perform transfer of custom inputs/targets data structure to `device`.
        """
        for transfer_key in self.transfer_keys:
            for key, val in self.collated_batch[transfer_key].items():
                # Move `DataModel` to device
                val = val.to(device)
    
    @property
    def unpack(self) -> Any:
        """ Extract and return unpacked batch data : just a convenient interface
        """
        inputs = self.collated_batch["inputs"]
        targets = self.collated_batch["targets"]
        sources = self.collated_batch["source"]
        batch_size = self.collated_batch["batch_size"]
        return inputs, targets, sources, batch_size
        

class BatchCollater:
    """ Custom `BatchCollator` class to provide to `DataLoader`.
    """
    def __init__(self) -> None:
        pass

    def __call__(self, batch: List[Any]) -> Tuple[Dict[str,Tensor], Dict[str,Tensor], List[str], int]:
        return Batch(batch)
