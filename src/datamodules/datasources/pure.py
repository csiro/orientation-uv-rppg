from src.datamodules.datasources import DatasetSample
from src.datamodules.datasources.paths.pure import PURE_SourceFile
from src.datamodules.datasources.models.pure import PURE_Video, PURE_BVP, PURE_Metadata, PURE_Timestamps

from typing import *


class PURE_DataSample(DatasetSample):
    """ PURE Dataset Sample

    Default structure:
        <root>/
            <subject_id>-<scenario_id>/
                video.avi
                video_timestamps.txt
                bvp.txt
                bvp_timestamps.txt
                metadata.HDF5

    """
    def __init__(self, *args, **kwargs) -> None:
        super(PURE_DataSample, self).__init__(*args, **kwargs)
        
    def data(self) -> None:
        raise NotImplementedError("PURE")

    def video(self, filename: Optional[str] = None) -> PURE_Video:
        # path = self.path.joinpath(filename if filename is not None else PURE_SourceFile.VIDEO.value)
        # return PURE_Video(path)
        # path = self.path.joinpath(filename) if filename is not None else self.path # 01-01/01-01
        return PURE_Video(path=self.path, relpath=filename)

    def bvp(self, filename: Optional[str] = None) -> PURE_BVP:
        path = self.path.joinpath(filename if filename is not None else PURE_SourceFile.BVP.value)
        return PURE_BVP(path)

    def timestamps(self, filename: Optional[str] = None) -> PURE_Timestamps:
        path = self.path.joinpath(filename if filename is not None else PURE_SourceFile.TIME.value)
        return PURE_Timestamps(path)
    
    def metadata(self, filename: Optional[str] = None) -> PURE_Metadata:
        path = self.path.joinpath(filename if filename is not None else PURE_SourceFile.METADATA.value)
        return PURE_Metadata(path)


class PURE_DataSample_Frames(PURE_DataSample):
    """ PURE Dataset implementation which uses frames folders instead of video files.
    """
    def __init__(self, *args, **kwargs) -> None:
        super(PURE_DataSample_Frames, self).__init__(*args, **kwargs)

    def video(self, filename: Optional[str] = None, *args, **kwargs) -> PURE_Video:
        video_kwargs = {
            "regex": "frame_*.png",
            "start": len("frame_")
        }
        return PURE_Video(path=self.path, relpath=filename, video_kwargs=video_kwargs)


from src.datamodules.datasources.models.pure import PURE_rPPGToolbox_Video, PURE_rPPGToolbox_BVP


class PURE_rPPGToolbox_DataSample(DatasetSample):
    """ rPPG-Toolbox PURE Dataset Sample

    Default structure:
        <root>/
            <subject_id>-<scenario_id>/
                video.avi
                bvp_rppgtoolbox.txt
                metadata.HDF5

    """
    def __init__(self, *args, **kwargs) -> None:
        super(PURE_rPPGToolbox_DataSample, self).__init__(*args, **kwargs)
        
    def data(self) -> None:
        raise NotImplementedError("rPPG-Toolbox PURE generic data accessor")

    def video(self, filename: Optional[str] = None) -> PURE_rPPGToolbox_Video:
        # Frame directory
        path = self.path.joinpath(filename) if filename is not None else self.path # 01-01/01-01 or 01-01/<filename>
        return PURE_rPPGToolbox_Video(path)

    def bvp(self, filename: Optional[str] = None) -> PURE_rPPGToolbox_BVP:
        path = self.path.joinpath(filename if filename is not None else "bvp_rppgtoolbox.HDF5")
        return PURE_rPPGToolbox_BVP(path)
    
    def metadata(self, filename: Optional[str] = None) -> PURE_Metadata:
        path = self.path.joinpath(filename if filename is not None else PURE_SourceFile.METADATA.value)
        return PURE_Metadata(path)


class PURE_rPPGToolbox_DataSample_Frames(PURE_rPPGToolbox_DataSample):
    """ rPPG-Toolbox PURE Dataset implementation which uses frames folders instead of video files.
    """
    def __init__(self, *args, **kwargs) -> None:
        super(PURE_rPPGToolbox_DataSample_Frames, self).__init__(*args, **kwargs)

    def video(self, filename: Optional[str] = None) -> PURE_Video:
        """ Override the `video` function to use the filename as the sub-directory in which images frames are located.
        """
        video_kwargs = {
            "regex": "frame_*.png",
            "start": len("frame_")
        }
        return PURE_rPPGToolbox_Video(path=self.path, relpath=filename, video_kwargs=video_kwargs)


class Default_PURE_rPPGToolbox_DataSample_Frames(PURE_rPPGToolbox_DataSample):
    """ rPPG-Toolbox PURE Dataset implementation which uses frames folders instead of video files.
    """
    def __init__(self, *args, **kwargs) -> None:
        super(Default_PURE_rPPGToolbox_DataSample_Frames, self).__init__(*args, **kwargs)

    def video(self, filename: Optional[str] = None) -> PURE_Video:
        """ Override the `video` function to use the filename as the sub-directory in which images frames are located.
        """
        video_kwargs = {
            "regex": "Image*.png",
            "start": len("Image")
        }
        return PURE_rPPGToolbox_Video(path=self.path, relpath=filename, video_kwargs=video_kwargs)


