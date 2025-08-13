import os
import pickle
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from torch.utils.data import Dataset as TorchDataset
from src.datamodules.datasources import DatasetSample

from typing import *

log = logging.getLogger(__name__)



class Dataset(TorchDataset):
    """ Base implementation for `Dataset`s

    Classes which inherit this class should override the `format` method and NOT the `__getitem__` method as is
    traditional with `torch` style datasets.

    Args:
        TorchDataset (_type_): _description_
    """
    def __init__(self,
        source: Iterable[str],
        construct: Callable,
        stage: Optional[str] = None,
        process: Optional[Callable] = None,
        augment: Optional[Callable] = None,
        detections: Optional[str] = None,
        profiler: Optional[Callable] = None,
        filenames: Optional[Dict[str,List[str]]] = {},
        cache: Optional[str] = None,
        *args, **kwargs
    ) -> None:
        """_summary_

        Args:
            source (Iterable[str]): Iterable inputs to be provided to the constructor.
            construct (Callable): Callable function which accumulated `DatasetModel` into a Dict.
            context (Callable): Callable function to access the returned data
            datasetmodel (Callable): `DatasetModel` used to interface to the underlying data retrieved from the source.
            stage (Optional[str], optional): Whether to only use specific stages of the dataset. Defaults to None.
            process (Optional[Callable], optional): Apply processes to the formatted samples (inputs & targets). Defaults to None.
        """
        super(Dataset, self).__init__()

        # Construct `DatasetSample` index
        self.samples : Dict[str, Any] = construct(source)["samples"] # load `DatasetModel` entries here.
        self.index = dict()

        # Processing & Augmentation
        self.process = process
        self.augment = augment

        # Detections
        self.detections = detections
        self.filenames = filenames

        # Profiler : `DataModule` profiler
        self.profiler = profiler

        # Enable temporary storing (caching)
        self.cache = Path(cache) if cache is not None else cache
        if self.cache is not None: self.cache.mkdir(exist_ok=True, parents=True)

        # Filter samples for current stage AND construct index AND cache
        self.filter(stage)
        if len(self) == 0: log.warning(f"Dataset ({stage}) does NOT contain any data.")

    def filter(self, stage: str) -> None:
        """ Filter the contained `DatasetSample`s by `stage` and construct the `Dataset` index.
        
        NOTE: Necessary to call  to construct the index, otherwise dataset will not work.
        """
        # Remove non-relevant stages from file
        if stage is not None:
            for key in list(self.samples.keys())[::-1]:
                sample = self.samples[key]
                if hasattr(sample, "stage"):
                    if stage not in sample.stage:
                        del self.samples[key]
                else:
                    del self.samples[key]

        # Index keys
        self.index = {idx: key for idx, key in enumerate(self.samples.keys())}

    def __len__(self) -> int:
        """ Return the number of indexable `DatasetSample`s in the `Dataset`
        """
        return len(self.index.keys())
    
    def __getitem__(self, index: int) -> Any:
        """_summary_

        each of the objects can be individually cached or not as desired e.g. can cache the ppg labels
        during each transform we can check if the object we're overriding is cached or not and only
        override that object : but since processes have dependencies between the data, e.g. cropping
        video also impacts the bounding boxes so if you cache one and not the other then it doesnt
        make much sense. 
        
        Deal with the holistic inputs and outputs of the results e.g. cache the processed entries e.g.
        after static processing has been applied then cache those results. hence each dataset should
        individually handle the caching of the formatted dataset.

        NOTE: What about applying processes multiple times to the data: split into random and non-random?
        """
        # Stop iteration
        if index >= len(self): raise IndexError("End of Dataset")

        # Acces key to ID sample
        key : int = self.index[index]
        
        # Load sample based on index
        if self.cache is None:
            # Load data and process
            # NOTE: 28/10/2023 Copy-on-read (<512B/sample : 19.33MB RAM per instance of a worker : acceptable)
            sample : DatasetSample = self.samples[key]

            # Load and format data (dataset-specific) : we incur I/O operations here
            data = dict()
            for attr, filenames in self.filenames.items():
                for loc, filename in filenames.items():
                    data[loc] = getattr(sample, attr)(filename=filename).data(start=sample.start, stop=sample.stop)
                    if "metadata" in attr:
                        data[loc].attrs["dataset_name"] = self.name
                        data[loc].attrs["index"] = index
                        data[loc].attrs["key"] = key
                    if "sample" in loc: 
                        data[loc].data = f"{key}" # index as reference

            # Apply in-place data operations to the resultant items
            if self.process is not None:
                data = self.process(data)["data"]            
        else:
            # Check if exists
            # video_name = data["source/sample"].attrs["video_name"]
            # TODO: Add in ID/start index for each window to allow for unique definition w.r.t video_name (generalize across filtering etc.)
            path = self.cache.joinpath(f"{key}.pkl")
            if path.exists():
                # Load cached file
                with open(path, "rb") as fp:
                    data = pickle.load(fp)

            else:
                # Load data and process
                sample : DatasetSample = self.samples[key]

                # Load and format data (dataset-specific) : we incur I/O operations here
                data = dict()
                for attr, filenames in self.filenames.items():
                    for loc, filename in filenames.items():
                        data[loc] = getattr(sample, attr)(filename=filename).data(start=sample.start, stop=sample.stop)
                        if "metadata" in attr:
                            data[loc].attrs["dataset_name"] = self.name
                            data[loc].attrs["index"] = index
                            data[loc].attrs["key"] = key
                        if "sample" in loc: 
                            data[loc].data = f"{key}" # index as reference

                # Apply in-place data operations to the resultant items
                if self.process is not None: 
                    data = self.process(data)["data"]

                # Conditionally cache the results of the static processing
                with open(path, "wb") as fp:
                    pickle.dump(data, fp)

        # Apply stochastic augmentations
        if self.augment is not None:
            data = self.augment(data)["data"]

        return data
    
    def attributes(self, *args, **kwargs) -> Dict[str, Any]:
        """ Load dataset attributes stored across `DatasetSample` metadata files.
        """
        # Attributes
        samples_attributes = {}

        # Create results
        for sample in self.samples.values():
            sample_attrs : Dict[str, Any] = sample.metadata().data().attrs

            # Add to attributes
            for key, val in sample_attrs.items():
                if key in samples_attributes:
                    samples_attributes[key].append(val)
                else:
                    samples_attributes[key] = [val]

        # Compact results
        for key, vals in samples_attributes.items():
            samples_attributes[key] = list(set(vals))

        # Dataset Attributes
        dataset_attributes = {
            "length": len(self)
        }

        # Combine additional attributes
        return {**samples_attributes, **dataset_attributes}