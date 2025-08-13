import logging
from typing import Any
from src.datamodules.datapipes import DataOperation
from src.datamodules.datasources import DatasetSample
from src.datamodules.datasources.models import DatasetModel
from abc import ABC, abstractmethod

from typing import *

log = logging.getLogger(__name__)


class FilterDatasetSamples(DataOperation):
    pass


class FilterDatasetSamplesIn(DataOperation):
    """ Returns `True` if you should INCLUDE the item else `False`.
    """
    def __init__(self, filters: Dict[str, Callable], *args, **kwargs) -> None:
        super(FilterDatasetSamplesIn, self).__init__(*args, **kwargs)
        self.filters = filters

    def __call__(self, sample: DatasetSample) -> bool:
        use = self.filter(sample)
        return use
    
    def filter(self, sample: DatasetSample) -> bool:
        use = True
        for filter in self.filters.values():
            use &= filter.condition(sample)
        return use
    

class FilterDatasetSamplesOut(DataOperation):
    """ Returns `True` if you should EXCLUDE the item else `False`.
    """
    def __init__(self, filters: Dict[str, Callable], *args, **kwargs) -> None:
        super(FilterDatasetSamplesOut, self).__init__(*args, **kwargs)
        self.filters = filters

    def __call__(self, sample: DatasetSample) -> bool:
        skip = self.filter(sample)
        return skip
    
    def filter(self, sample: DatasetSample) -> bool:
        use = True
        for filter in self.filters.values():
            use &= filter.condition(sample)
        return not use


class Filter(DataOperation):
    """ Filter samples out of the `Dataset` if they DO NOT need a certain
    condition.

    Dynamically query values in a dataset entry and return subset of those entries.

    Args:
        ABC (_type_): _description_
    """
    def __init__(self, match: Dict[str, str], *args, **kwargs) -> None:
        """_summary_

        Args:
            match (Dict[str, str]): Key defines the attribute to read.
        """
        super(Filter, self).__init__(return_key=None)
        self.match = match

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
    
    @abstractmethod
    def condition(self, sample: DatasetSample) -> bool:
        """_summary_

        Args:
            key (str): Key corresponding to the sample entry key.
            val (Iterable[str]): Value corresponding to the entry.

        Returns:
            bool: _description_
        """
        pass


from src.datamodules.datasources.files import DatasetFile


class IncludeIfAnyIn(Filter):
    """ Filters out `DatasetModel`s where the key IS NOT IN the values specified
    for ANY of the attributes (match).

    motion: ["Stationary"] =  INCLUDE (STATIONARY)

    Args:
        Filter (_type_): _description_
    """
    def condition(self, sample: DatasetSample) -> bool:
        """ Return `True` if ANY of the val.attr == desired value

        E.g. getattr(val,attr) : val.root.subject == mval : "LED-high"

        NOTE: Specific `DatasetModel` child will depend on the `Dataset` implementation,
        hence we dynamically access attributes here.

        Args:
            key (str): Key used to index the sample in `Dataset.data`
            val (DatasetModel): Sample of type `DatasetModel`

        Returns:
            bool: `True` if any `match_val`'s in `attr_val`'s.
        """
        results = []
        data : DatasetFile = sample.metadata()

        for attr_key, attr_vals in self.match.items():
            # Dynamically get attribute (value) of `DatasetFile`
            match_val = getattr(data, attr_key) 

            # TRUE if `DatasetModel` attr is in the items : Hence keep this `DatasetModel`
            results.append(match_val in attr_vals)

        return any(results)
    

class ExcludeIfAnyIn(Filter):
    """ Filters out `DatasetModel`s where the key IS IN the values specified
    for ANY of the attributes (match).

    motion: ["Stationary"] = EXCLUDE (STATIONARY)

    motion: ["Stationary"] & light ["LED-high"] = EXCLUDE (STATIONARY OR LED-HIGH)

    Args:
        Filter (_type_): _description_
    """
    def condition(self, sample: DatasetSample) -> bool:
        """_summary_

        Args:
            key (str): _description_
            data (DatasetModel): _description_

        Returns:
            bool: `False` if any `match_val`'s in `attr_val`'s.
        """
        results = []
        data : DatasetFile = sample.metadata()

        # Iterate over attributes
        for attr_key, attr_vals in self.match.items():
            # Dynamically get attribute of the `DatasetModel`
            match_val = getattr(data, attr_key)

            # TRUE if `DatasetModel` attr is NOT IN the items : Hence keep this `DatasetModel`
            results.append(match_val in attr_vals) # TRUE if in [...]

        return not any(results)


class IncludeIfAllIn(Filter):
    """ Filters out `DatasetModel`s where the key IS IN the values specified
    for ANY of the attributes (match).

    motion: ["Stationary"] = EXCLUDE (STATIONARY)

    motion: ["Stationary"] & light ["LED-high"] = EXCLUDE (STATIONARY OR LED-HIGH)

    Args:
        Filter (_type_): _description_
    """
    def condition(self, sample: DatasetSample) -> bool:
        """_summary_

        Args:
            key (str): _description_
            data (DatasetModel): _description_

        Returns:
            bool: TRUE if all `match_val` are in the `attr_vals` wrt. `attr_key`.
        """
        results = []
        data : DatasetFile = sample.metadata()

        # Iterate over attributes
        for attr_key, attr_vals in self.match.items():
            # Dynamically get attribute of the `DatasetModel`
            match_val = getattr(data, attr_key)

            # TRUE if `DatasetModel` attr is NOT IN the items : Hence keep this `DatasetModel`
            results.append(match_val in attr_vals)

        return all(results)
    

class ExcludeIfAllIn(Filter):
    """ Filters out `DatasetModel`s where the key IS IN the values specified
    for ANY of the attributes (match).

    motion: ["Stationary"] = EXCLUDE (STATIONARY)

    motion: ["Stationary"] & light ["LED-high"] = EXCLUDE (STATIONARY OR LED-HIGH)

    Args:
        Filter (_type_): _description_
    """
    def condition(self, sample: DatasetSample) -> bool:
        results = []
        data : DatasetFile = sample.metadata()

        # Iterate over attributes
        for attr_key, attr_vals in self.match.items():
            # Dynamically get attribute of the `DatasetModel`
            match_val = getattr(data, attr_key)

            # TRUE if `DatasetModel` attr is NOT IN the items : Hence keep this `DatasetModel`
            results.append(match_val in attr_vals)

        return not all(results)