import logging
from src.datamodules.datapipes import DataOperation

from typing import *
from h5py import File

log = logging.getLogger(__name__)


# class ConstructIndex(DataOperation):
#     """_summary_

#     Args:
#         DataOperation (_type_): _description_
#     """
#     def __init__(self, opts: Optional[Dict[str, Any]] = {}, *args, **kwargs) -> None:
#         """_summary_

#         Args:
#             context (Any): 
#         """
#         super(ConstructIndex, self).__init__(return_key="index")
#         self.opts = opts

#     def __call__(self, file: Any, context: Callable, *args, **kwargs) -> Any:
#         """_summary_

#         Args:
#             file (Any): _description_

#         Returns:
#             Any: _description_
#         """
#         with context(file, "r+") as fp:




#     def __call__(self, file: File, *args, **kwargs) -> Dict[str, Any]:
#         """_summary_

#         Args:
#             file (File): File-based (or otherwise) context manager with an `__enter__` method.

#         Returns:
#             Dict[str, Dict[str, DatasetSample]]: _description_
#         """
#         log.debug(f"[{self.__class__.__name__}] Constructing index...")

#         # Access contained interface method
#         interface = file.__enter__().interface()
#         func = type(interface)

#         # Create `Dict` of interfaces to the data
#         return {key: func(data=val["data"], start=val.attrs["start"], stop=val.attrs["stop"], **self.opts) for (key, val) in interface.data.items()}
