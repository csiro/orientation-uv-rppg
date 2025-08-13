import numpy as np
from src.datamodules.datasources.files import DatasetFile
from pathlib import Path

from typing import *
from numpy import ndarray


class CSVNumeric(DatasetFile):
    def __init__(self, *args, **kwargs) -> None:
        super(CSVNumeric, self).__init__(*args, **kwargs)

    def data(self, start: Optional[int] = 0, stop: Optional[int] = None, *args, **kwargs) -> ndarray:
        max_rows = stop - start if stop is not None else stop
        return np.loadtxt(self.path, skiprows=start, max_rows=max_rows, *args, **kwargs)
