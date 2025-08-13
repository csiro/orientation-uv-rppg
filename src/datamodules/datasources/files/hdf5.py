import h5py
from pathlib import Path
from src.datamodules.datasources.files import File

from typing import *
from omegaconf.listconfig import ListConfig




class HDF5File(File):
    """ HDF5-based File Context Manager

    HDF5 : https://docs.h5py.org/en/stable/index.html

    Key attributes of HDF5 file formatting includes
        - Groups (structural)
        - Dataset (chunking, compression)
        - Attributes (self-descriptive)
        - Links (soft, hard, external, virtual-sources)

    NOTE: HDF5 files perform sliced-IO access at run-time allowing for highly 
    dynamic access to the underlying data in a performant manner.

    Args:
        File (_type_): _description_
    """
    def __init__(self, path: str, *args, **kwargs) -> None:
        super(HDF5File, self).__init__(path, h5py.File, *args, **kwargs)

    def create_dataset(self, location: str, data: Any, *args, **kwargs) -> None:
        kwargs = {key: tuple(val) if type(val) == ListConfig else val for (key, val) in kwargs.items()}
        dataset = self.file.create_dataset(location, data=data, *args, **kwargs)
        return dataset

    def create_attribute(self, location: str, val: Any, *args, **kwargs):
        location = Path(location)
        parent, name = str(location.parent), str(location.name)
        self.file[parent].attrs.create(name, val)

    @staticmethod
    def suffix() -> str:
        return ".HDF5"