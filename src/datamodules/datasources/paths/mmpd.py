from enum import Enum
from pathlib import Path
from functools import partial
from src.datamodules.datasources.paths import Paths
from src.datamodules.datasources.files.mmpd import MMPD_Keys

from typing import *


class MMPD_Subject(Enum):
    ALL = "p*_*"


class MMPD_SourceFile(Enum):
    """
    """
    ALL = "*"
    VIDEO = f"{MMPD_Keys.VIDEO.value}.avi"
    BVP = f"{MMPD_Keys.GT_BVP.value}.HDF5"
    METADATA = f"metadata.HDF5"


class MMPD_RawFiles(Paths):
    """ Provides a structured method for iterating over MMPD files.
    
    File structure for the MMPD data is as follows:
        <root>/
            subject_??/
                p*_*.mat # Files

    """
    def __init__(self,
        subject: MMPD_Subject,
        *args, **kwargs
    ) -> None:
        # Attributes
        self.subject = subject

        # Structure
        regex = f"**/{self.subject}"

        super(MMPD_RawFiles, self).__init__(regex=regex, *args, **kwargs)


def list_mmpd_paths(root: str, regex: str) -> Dict[str, Any]:
    return {p.name: p for p in Path(root).glob(regex)}


class MMPD_Files(Paths):
    def __init__(self, *args, **kwargs) -> None:
        process = partial(list_mmpd_paths, regex="*")
        super(MMPD_Files, self).__init__(process=process, *args, **kwargs)


class MMPD_Roots(Paths):
    def __init__(self, root: str) -> None:
        super(MMPD_Roots, self).__init__(root, "*/")