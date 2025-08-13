'''
DataPipes

Expect to transition over to utilizing `torchdata` once the API becomes more mature.

'''
import logging
from tqdm import tqdm
from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count
from pathos.multiprocessing import ProcessPool

from typing import *

log = logging.getLogger(__name__)


class DataOperation(ABC):
    """ `DataOperation` perform operations on the input data and return the result optionally wrapped
    in a `Dict` if a key is provided.
    """
    def __init__(self, return_key: Optional[str] = None) -> None:
        super(DataOperation, self).__init__()
        self.return_key = return_key

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """ Perform a specific data operation here.

        Can be either in-place or out-of-place: typically for `Dataset` processing we will utilize in-place
        operations on a specific data structure, however this is further documented in those specific
        implementations.

        NOTE: Want to maintain the flexibility of applying the process 
        """
        pass

    def compute(self, *args, **kwargs) -> Dict[str, Any]:
        """ Acts as an interface method for the data pipe to sequentially apply operations and maintain wrapping.
        """
        results = self.__call__(*args, **kwargs)
        return {self.return_key: results} if self.return_key is not None else results


class DataPipe(ABC):
    """ `DataPipe` act as containers for applying different operations to the data.
    """
    def __init__(self, operations: Dict[str, DataOperation], *args, **kwargs) -> None:
        super(DataPipe, self).__init__()
        self.operations = operations

    def __call__(self, input: Any, pbar: Optional[tqdm] = None, *args, **kwargs) -> Dict[str, Any]:
        """ Sequentially apply a set of operations to the input data.

        NOTE: Operations should return a `Dict[str,Any]` which contains relevant keyword arguments
        which will be provided to the next `Operation`
        """
        if pbar is not None: base_desc = pbar.desc
        for idx, (key, operation) in enumerate(self.operations.items()):
            if pbar is not None: pbar.set_description(f"{base_desc} : {key}...")
            results : Dict[str, Any] = operation.compute(input, *args, **kwargs) if idx == 0 else operation.compute(**results)
            if pbar is not None: pbar.update(1)
        return results



class DatasetOperation(DataOperation):
    """ Define a specific framework for `DataOperation`s for processing of `Dict[str, DataModel]`
    in a consistent manner: should only perform in-place operations on the stored data.

    Data from `compute` should always be returned wrapped in {"data":...} to facilitate
    passing through the `DataPipe`.
    """
    def __init__(self, return_key: Optional[str] = "data", *args, **kwargs) -> None:
        super(DatasetOperation, self).__init__(return_key=return_key, *args, **kwargs)

    def __call__(self, data: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        self.apply(data)
        return data
    
    @abstractmethod
    def apply(self, data: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        """ Define the in-place operation to apply to the data, possibly in terms of
        a functional interface.

        Args:
            data (Dict[str, Any]): _description_
        """
        pass

    @staticmethod
    def functional(self, *args, **kwargs) -> None:
        """ Functional method for the `DataOperation`.

        With compound methods which handle dynamic underlying datatypes this
        may not be possible as they typically require access to an interface
        class.
        """
        pass


class ThreadedDataOperation:
    pass


class DistributedDataOperation(DataOperation):
    def __init__(self, num_proc: Optional[int] = None, *args, **kwargs) -> None:
        """_summary_

        0 = no mp.
        n = n proc
        None = cpu_count proc.
        """
        super(DistributedDataOperation, self).__init__(*args, **kwargs)
        num_proc = cpu_count() if num_proc is None else num_proc
        self.num_proc = num_proc if num_proc < cpu_count() else cpu_count()

    @abstractmethod
    def __call__(self, *args, **kwargs) -> List[Any]:
        pass

    def compute(self, items: Iterable[Any], *args, **kwargs) -> Dict[str, Any]:
        if self.num_proc > 0:
            with Pool(processes=self.num_proc) as pool:
                results = pool.map_async(
                    func=self.__call__,
                    iterable=items
                )
                results = results.get()
                return {self.return_key: results} if self.return_key is not None else results
        else:
            result = [self.__call__(item) for item in items]
            return {self.return_key: result} if self.return_key is not None else result


class PathosMultiprocessOperation(DistributedDataOperation):
    """_summary_

    https://github.com/uqfoundation/pathos/tree/master

    # NOTE: Cannot pickle a file object even will `dill` which restricts multiprocessing, however utilizing `Threads` can still provide compute benefit as access is primarily IO restricted
    """
    def compute(self, items: Iterable[Any], *args, **kwargs) -> Dict[str, Any]:
        if self.num_proc > 0:
            with ProcessPool(nodes=self.num_proc) as pool:
                result = pool.amap(
                    self.__call__, 
                    items
                )
                result = result.get()
        else:
            result = [self.__call__(item) for item in items]
        return {
            self.return_key: result
        } if self.return_key is not None else result
