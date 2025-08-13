from src.datamodules.datasources import DatasetSample
from src.datamodules.datasources.models.mmpd import MMPD_DatasetModel
from src.datamodules.datasources.paths.mmpd import MMPD_SourceFile, list_mmpd_paths
from src.datamodules.datasources.files.mmpd import MMPD_Video, MMPD_BVP, MMPD_Metadata
from src.datamodules.datasources.paths.detections import DetectionsSourceFile
from src.datamodules.datasources.files.detections import BoundingBoxesFile, LandmarksFile

from typing import *


class MMPD_DatasetSample(DatasetSample):
    """ 

    General principle follow

    Minimum required data should be
        Root directory
        Slice 

    """
    def __init__(self, *args, **kwargs) -> None:
        super(MMPD_DatasetSample, self).__init__(*args, **kwargs)

    def data(self) -> None:
        raise NotImplementedError(f"MMPD Data")
    
    def video(self, filename: Optional[str] = None) -> MMPD_Video:
        # path = self.path.joinpath(filename if filename is not None else MMPD_SourceFile.VIDEO.value)
        path = self.path.joinpath(filename) if filename is not None else self.path.joinpath("frames") # just provide root directory to sample frames
        return MMPD_Video(path)
    
    def bvp(self, filename: Optional[str] = None) -> MMPD_BVP:
        path = self.path.joinpath(filename if filename is not None else MMPD_SourceFile.BVP.value)
        return MMPD_BVP(path)
    
    def metadata(self, filename: Optional[str] = None) -> MMPD_Metadata:
        path = self.path.joinpath(filename if filename is not None else MMPD_SourceFile.METADATA.value)
        return MMPD_Metadata(path)
    

class MMPD_DatasetSource:
    def __init__(self, roots: Iterable[str]) -> None:
        self.roots = roots

    def __iter__(self) -> Generator:
        for root in self.roots:
            yield MMPD_DatasetSample(root)
