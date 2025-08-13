import logging
from typing import Any, Optional
from torch.utils.data.dataset import random_split
from src.datamodules.datapipes.sample.filters import Filter
from src.datamodules.datapipes import DataOperation
from src.datamodules.datasources import DatasetSample
from src.datamodules.datapipes.sample import DatasetStage

from typing import *

log = logging.getLogger(__name__)


class AssignSplit(DataOperation):
    def __init__(self, stages: List[str], exclude: Optional[List[str]] = [], *args, **kwargs) -> None:
        super(AssignSplit, self).__init__(*args, **kwargs)
        self.stages = stages
        self.exclude = exclude

    def __call__(self, samples: Dict[int, DatasetSample], *args, **kwargs) -> Dict[str, DatasetSample]:
        # Iterate over samples
        for sample in samples.values():
            # Per Stage
            for stage in self.stages:

                # Assign stage if un-assinged
                if not hasattr(sample, "stage"):
                    sample.stage = stage

                else:
                    # Conditionally append stages
                    if not any([excl_stage in sample.stage for excl_stage in self.exclude]):
                        sample.stage = f"{sample.stage}_{stage}"
        return samples


class AttributeSplit(DataOperation):
    """ Split the attributes into `training`, `validation`, and `testing` sets based on their
    attributes by assigning each sample into a category via. HDF5 attribute data.

    Args:
        DataOperation (_type_): _description_
    """
    def __init__(self, train: Optional[Filter] = None, valid: Optional[Filter] = None, test: Optional[Filter] = None, assign_remaining: Optional[bool] = False, allow_multiple_stages: Optional[bool] = False, *args, **kwargs) -> None:
        """_summary_

        Args:
            train (Filter): Pipe containing multiple filters returning `True` if it should be in `train`.
            valid (Filter): _description_
            test (Optional[Filter], optional): _description_. Defaults to None.
        """
        super(AttributeSplit, self).__init__(*args, **kwargs)

        # Filters (shouldUse?)
        self.train = train
        self.valid = valid
        self.test = test

        # Assign remaining
        self.assign_remaining = assign_remaining
        self.allow_multiple_stages = allow_multiple_stages

    def __call__(self, samples: Dict[int, DatasetSample], *args, **kwargs) -> Dict[str, DatasetSample]:
        # Process stages iteratively
        for stage in [DatasetStage.TRAIN.value, DatasetStage.VALID.value, DatasetStage.TEST.value]:

            # Obtain the stage-specific filter
            stage_filter = getattr(self, stage)

            # Apply filter
            if stage_filter is not None:

                # Process each item in the dataset
                for sample in samples.values():

                    # Use sample?
                    # TODO: Current implementation only allows for filtering of metadata and not data attributes; this may be beneficial in ths future.
                    if stage_filter.filter(sample.metadata()):

                        # Assign stage
                        if not hasattr(sample, "stage"):
                            sample.stage = stage
                        else: # conditionally assign
                            if self.allow_multiple_stages:
                                sample.stage = f"{sample.stage}_{stage}"

            else:
                # Conditionally assign items if not associated with a filter
                if self.assign_remaining:
                    
                    # Process each item in the dataset
                    for sample in samples.values():

                        # Assign stage
                        if not hasattr(sample, "stage"):
                            sample.stage = stage
                        else: # conditionally assign
                            if self.allow_multiple_stages:
                                # ONLY allow duplicate stages if NOT associated with a defined filter.
                                if all([getattr(self, f_stage) is None for f_stage in sample.stage.split("_")]):
                                    sample.stage = f"{sample.stage}_{stage}"
        
        return samples


class RandomSplit(DataOperation):
    """ Randomly split the dataset based on the fractional amounts specified.

    NOTE: You can use this with other filters e.g. use `AttributeSplit` to define the training data and
    then randomly split the validation/testing data.

    Args:
        DataOperation (_type_): _description_
    """
    def __init__(self, split: Union[float, List[float]], *args, **kwargs) -> None:
        """_summary_

        Args:
            split (Union[float, List[float]]): _description_
        """
        super(RandomSplit, self).__init__(*args, **kwargs)
        self.split = [split] if type(split) == float else list(split)
        assert len(self.split) <= 3, f"split must have at MOST 3 entries."
        
        # Fractional amount of dataset per stage
        self.train = self.split[0]
        self.valid = self.split[1] if len(self.split) > 1 else (1 - self.train - 1e-16) / 2 # ensure sum < 1
        self.test = self.split[2] if len(self.split) > 2 else (1 - self.train - self.valid - 1e-16)

    def __call__(self, samples: Dict[int, DatasetSample], *args, **kwargs) -> Any:
        """_summary_

        Args:
            file (Any): _description_
            context (Callable): _description_

        Returns:
            Any: _description_
        """
        # Only assign non-assigned keys
        keys = [key for (key, val) in samples.items() if not hasattr(val, "stage")]

        # Randomly split the entries in file (indexes)
        subsets = random_split(keys, [self.train, self.valid, self.test])

        # Iteratively construct stages
        for subset, stage in zip(subsets, [DatasetStage.TRAIN.value, DatasetStage.VALID.value, DatasetStage.TEST.value]):
            
            # For each dataset index assign the stage
            for idx in subset.indices:

                # Assign the stage based on the index
                key = subset.dataset[idx]
                sample = samples[key]

                if not hasattr(sample, "stage"):
                    sample.stage = stage

        return samples