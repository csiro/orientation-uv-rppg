from src.datamodules.datapipes import DataOperation, DatasetOperation
from src.datamodules.datamodels.timeseries import TimeseriesModel

from typing import *


class ExportBVP(DatasetOperation):
    def __init__(self, key: str, filename: str, *args, **kwargs) -> None:
        super(ExportBVP, self).__init__(*args, **kwargs)
        self.key = key
        self.filename = filename

    def apply(self, data: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        # Reference existing items
        bvp : TimeseriesModel = data[self.key]

        # Extract root
        root = bvp.attrs["root"]
        path = root.joinpath(self.filename)

        # Write to text file
        with open(path, "w") as fp:
            fp.write("BVP\n")
            for val in bvp.data:
                fp.write(f"{float(val)}\n")
