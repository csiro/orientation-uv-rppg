from pathlib import Path
from abc import ABC, abstractmethod

from typing import *
from _io import BytesIO


class DatasetFile(ABC):
    """ Each `DatasetFile` should implement the dataset-specific methods for reading
    and minimally formatting the data.

    NOTE: We generally implement each `DatasetFile` as an iterable sliced stream to 
    support efficient lazy loading where possible to reduce IO overhead.
    """
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path).resolve().absolute() if path is not None else None

    @abstractmethod
    def data(self, start: Optional[int]=0, stop: Optional[int] = None) -> Any:
        """ Load a reference to the data lazily if possible. """
        pass


class File(ABC):
    """ Base-class for `File` context management.

    Args:
        ABC (_type_): _description_
    """
    def __init__(self, path: str, context: Callable, *args, **kwargs) -> None:
        super(File, self).__init__()
        self.path = Path(path).resolve().absolute() if type(path) not in [BytesIO] else path
        self.context = context
        self.args, self.kwargs = args, kwargs


    def __enter__(self) -> Any:
        """_summary_

        NOTE: Conditionally apply defined `self.process` to the loaded file upon
        entering the context manager. This should be used to set any important state
        information that may be needed in the context of the file.

        Returns:
            Any: _description_
        """
        if type(self.path) not in [BytesIO]: self.path.parent.mkdir(exist_ok=True, parents=True)
        self.file = self.context(self.path, *self.args, **self.kwargs)
        if hasattr(self, "process"): self.file = self.process(self.file)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        if self.file:
            self.file.close()
        if exc_type is not None:
            raise exc_type(exc_value)
        
    @property
    @abstractmethod
    def suffix(self) -> str:
        pass

    @property
    def suffixes(self) -> List[str]:
        """ Conditionally return path suffixes or the default suffix for a file.

        Returns:
            List[str]: File suffixes
        """
        return self.path.suffixes if self.path.exists() else [self.suffix]




        