from abc import ABC, abstractmethod
from src.datamodules.datapipes import DatasetOperation

from typing import *


class Groups(ABC):
    """ Compute the group within the population based on
    some provided information.

    E.g. grouping target labels into odd/even for MNIST
    digit classification.

    Args:
        ABC (_type_): _description_
    """
    def __init__(self) -> None:
        # TODO: Re-label this to something more suitable.
        super(Groups, self).__init__()

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


class CreateGroup(DatasetOperation):
    def __init__(self, key: str, *args, **kwargs) -> None:
        super(CreateGroup, self).__init__(*args, **kwargs)
        self.key = key

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        """ Access the `DatasetSample` corresponding to the `Root` attributes and define a source identifer.
        """
        data[self.key].data = "_".join(data[self.key].attrs)
