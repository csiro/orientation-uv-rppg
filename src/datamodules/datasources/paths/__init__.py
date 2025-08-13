import logging
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod

from typing import *
from typing import Any

log = logging.getLogger(__name__)


class Paths(ABC):
    """ Base-class for creating generators for filesystem trawling.

    Args:
        ABC (_type_): _description_
    """
    def __init__(self, root: str, regex: str, process: Optional[Callable] = None, skip: Optional[Callable] = None, *args, **kwargs) -> None:
        super(Paths, self).__init__(*args, **kwargs)
        self.root = Path(root).resolve().absolute()
        self.regex = regex
        assert self.root.exists(), f"Provided a root directory ({self.root}) which does NOT exist."
        self.process = process
        self.skip = skip

    def __iter__(self) -> Generator:
        """ Iterable over globbed items in the root directory with conditionally skipping of 
        files based on a defined `self.skip` and yields processed items via. `self.process`.

        Yields:
            Generator[Any]: _description_
        """
        items = self.root.glob(self.regex)
        count = 0
        while True:
            try:
                item = self.process(next(items)) if self.process is not None else next(items)
                if self.skip is not None:
                    if self.skip(item): continue
                count += 1
                yield item
            except StopIteration:
                if count == 0: log.warning(f"Returned {count} items from {self.root} with {self.regex}")
                break


class FilterSource():
    def __init__(self, filters: Dict[str, Callable]) -> None:
        self.filters = filters

    def __call__(self, item: Any) -> bool:
        retain = True
        with item as fp:
            for filter in self.filters.values():
                retain &= filter.condition(None, fp.interface())
        skip = not retain
        return skip
                

