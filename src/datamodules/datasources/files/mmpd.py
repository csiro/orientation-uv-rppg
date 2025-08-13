import cv2
import h5py
from enum import Enum
from pathlib import Path
from scipy.io import loadmat
from src.datamodules.datasources.files import DatasetFile
from src.datamodules.datasources.files.video import FFMPEGVideoFile

from typing import *
from numpy import ndarray


class MMPD_Keys(Enum):
    """ Keys for raw `MMPD` dataset files.

    Raw MMPD Format:
        __header__ (str)
        __version__ (float)
        __globals__ (List)
        video (ndarray[fp32]) [T,H,W,C]
        GT_ppg (ndarray[fp32]) [1,T] >> (ndarray[fp32]) [T] (squeeze=True)
        light (ndarray[U5]) (List[str])
        motion (ndarray[U5]) (List[str])
        exercise (ndarray[U5]) (List[str])
        skin_color (ndarray[U5]) (ndarray[int]) [1,1] > (int) (squeeze=True)
        gender (ndarray[U5]) (List[str])
        glasser (ndarray[U5]) (List[str]) : [True, False]
        hair_cover (ndarray[U5]) (List[str]) : [True, False]
        makeup (ndarray[U5]) (List[str]) : [True, False]

    Additional keys added:
        video_sps (float)
        video_format (str)
        GT_ppg_sps (float)
        subject (str)
        clip (str)
    
    """
    # MATLAB Data
    HEADER          = "__header__" 
    VERSION         = "__version__"
    GLOBALS         = "__globals__"

    # Video Data
    VIDEO           = "video"
    VIDEO_FPS       = "video_sps" # Added
    VIDEO_FORMAT    = "video_format" # Added
    VIDEO_LENGTH    = "video_length" # Added
    VIDEO_HEIGHT    = "video_height" # Added
    VIDEO_WIDTH     = "video_width" # Added

    # BVP PPG Data
    GT_BVP          = "GT_ppg"
    GT_BVP_SPS      = "GT_ppg_sps" # Added

    # Metadata
    SUBJECT         = "subject" # Added
    CLIP            = "clip" # Added
    LIGHT           = "light"
    MOTION          = "motion"
    EXERCISE        = "exercise"
    SKIN_TONE       = "skin_color"
    GENDER          = "gender"
    GLASSES         = "glasser"
    HAIR_COVER      = "hair_cover"
    MAKEUP          = "makeup"


MMPD_METADATA_KEYS = [
    MMPD_Keys.SUBJECT,
    MMPD_Keys.CLIP,
    MMPD_Keys.LIGHT,
    MMPD_Keys.MOTION,
    MMPD_Keys.EXERCISE,
    MMPD_Keys.SKIN_TONE,
    MMPD_Keys.GENDER,
    MMPD_Keys.GLASSES,
    MMPD_Keys.HAIR_COVER,
    MMPD_Keys.MAKEUP
]


class MMPD_MATLAB(DatasetFile):
    """ MMPD Dataset files are provided as monolithic MATLAB compatible .mat files.

    NOTE: Due to the monolithic nature of the MMPD MATLAB files we can only load in ALL
    of the data at once.
    
    """
    def __init__(self, path: str, metadata: Optional[Dict[str, Any]] = {}, *args, **kwargs) -> None:
        super(MMPD_MATLAB, self).__init__(path, *args, **kwargs)
        self.metadata = metadata

    def data(self, *args, **kwargs) -> Any:
        """ Parse additional descriptive information from the file and add it to the 
        file attributes (in this case a `Dict` with the data) for use within the file
        context.
        
        NOTE: Pass additional options to the reader to ensure items are squeezed to their
        minimum data dimensions (some `MMPD` items have variable size e.g. skin_color \in 
        [[val]]).

        NOTE: MATLAB utilizes multiple different formats across versions, for
        versions v4, v6, and v7 the `scipy.io.loadmat` is the only available Python
        interface. For version 7.3 and beyond a HDF5 interface can be used.

        """
        # Load .mat data
        data = loadmat(self.path, squeeze_me=True)

        # Extract information from the filepath
        parts: Tuple = self.path.stem[1:].split("_") # pX_Y (X=subject, Y=clip)
        subject: int = int(parts[0])
        clip: int = int(parts[1])

        # Create parsed metadata
        data[MMPD_Keys.SUBJECT.value] = subject
        data[MMPD_Keys.CLIP.value] = clip

        # Create fixed metadata
        data[MMPD_Keys.VIDEO_FPS.value] = 30.0
        data[MMPD_Keys.VIDEO_FORMAT.value] = "THWC"
        data[MMPD_Keys.GT_BVP_SPS.value] = 30.0

        # Create dynamic metadata
        for key, val in self.metadata.items():
            data[key] = val

        return data

from src.datamodules.datasources.files.video import FramesFolder
from src.datamodules.datamodels.frames import VideoFramesModel
from src.datamodules.datamodels.timeseries import TimeseriesModel
from src.datamodules.datapipes.format import convert_frames_opencv2torchvision


class MMPD_VideoFrames(FramesFolder):
    """ MMPD Video-frames stored as images
    """
    def __init__(self, *args, **kwargs) -> None:
        super(MMPD_VideoFrames, self).__init__(*args, **kwargs)

    def frame_fps(self, *args, **kwargs) -> float:
        return 30.0
 

class MMPD_Video(DatasetFile):
    """ MMPD Video data is formatted into frames

    NOTE: can also use video containers but was having reliability issues with high bit-rate streams
    dropping due to the ffmpeg consumer/producer queues. See stack overflow.
    """
    def __init__(self, *args, **kwargs) -> None:
        super(MMPD_Video, self).__init__(*args, **kwargs)
        self._video = MMPD_VideoFrames(path=self.path, regex="frame_*.png", start=len("frame_"))

    def data(self, start: Optional[int] = 0, stop: Optional[int] = None, *args, **kwargs) -> Dict[str, Any]:
        # Read frames from video into default format
        data : ndarray = self.video(start, stop)

        # Extract capture attributes
        attrs = {
            MMPD_Keys.VIDEO_FPS.value: self._video.frame_fps(), 
            MMPD_Keys.VIDEO_LENGTH.value: data.shape[0]
        }

        # Create `DataModel`
        model = VideoFramesModel(data=data, format="opencv", attrs=attrs)

        return model
    
    def video(self, start: Optional[int] = 0, stop: Optional[int] = None, *args, **kwargs) -> ndarray:
        """ Return data in the default `torchvision` format.

        [THWC] 

        NOTE: Could implement these as `DataPipe` processes, however since they are fixed per dataset
        specific format this is a more performant choice.
        """
        # Read frames from video
        return self._video.data(start, stop) # THWC

    def video_fps(self, *args, **kwargs) -> float:
        return self._video.frame_fps()
    
    def video_length(self, *args, **kwargs) -> int:
        return self._video.frame_count()
    
    def video_height(self, *args, **kwargs) -> int:
        return self._video.frame_height()
    
    def video_width(self, *args, **kwargs) -> int:
        return self._video.frame_width()
    

def filename_frame(frame_index: int) -> str:
    return f"frame_{str(frame_index).zfill(8)}.png"

from pathlib import Path
from PIL import Image
import numpy as np


# class MMPD_Frames:
#     """
#     """
#     def __init__(self, path: str) -> None:
#         self.path = path

#     def data(self, start: Optional[int] = 0, stop: Optional[int] = None, *args, **kwargs) -> Dict[str, Any]:
#         #
#         indexes = range(start, self.video_length) if stop is None else range(start, stop)
#         filenames = [filename_frame(index) for index in indexes]

#         #
#         root = Path(self.path)
#         video = [Image.open(root.joinpath(filename)) for filename in filenames]
#         video = [np.array(frame.getdata()) for frame in video]
#         video = np.array(video) # [0-255] [UINT8] [TCHW] [RGB]

#         #
#         attrs = {
#             MMPD_Keys.VIDEO_FPS.value: self.video_fps
#         }

#         return VideoFramesModel(data=data, format="TCHW", attrs=attrs)

#     def video(self, start: Optional[int] = 0, stop: Optional[int] = None, *args, **kwargs) -> ndarray:
#         #
#         indexes = range(start, self.video_length) if stop is None else range(start, stop)
#         filenames = [filename_frame(index) for index in indexes]

#         #
#         root = Path(self.path)
#         video = [Image.open(root.joinpath(filename)) for filename in filenames]
#         video = [np.array(frame.getdata()) for frame in video]
#         video = np.array(video)

#         return video

#     @property
#     def video_fps(self) -> float:
#         return 30.0
    
#     @property
#     def video_length(self) -> int:
#         return 1800
    
#     @property
#     def video_height(self) -> int:
#         return 320
    
#     @property
#     def video_width(self) -> int:
#         return 240

from src.datamodules.datamodels.timeseries import TimeseriesKeys


class MMPD_BVP:
    """ MMPD BVP data is formatted into a .HDF5 container with no compression or chunking.
    """
    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def data(self, start: Optional[int] = 0, stop: Optional[int] = None, *args, **kwargs) -> Dict[str, Any]:
        with h5py.File(self.path, "r") as fp:
            ds = fp[MMPD_Keys.GT_BVP.value]
            data = ds[start:stop] if stop is not None else ds[start:]
            attrs = {
                TimeseriesKeys.SPS.value: ds.attrs[MMPD_Keys.GT_BVP_SPS.value]
            }
        return TimeseriesModel(data=data, format="T", attrs=attrs)
    
    @property
    def gt_bvp_sps(self) -> float:
        with h5py.File(self.path, "r") as fp:
            bvp_sps = fp.attrs[MMPD_Keys.GT_BVP_SPS.value]
        return bvp_sps

    
from src.datamodules.datamodels import MetadataModel # 

class MMPD_Metadata:
    """ MMPD Metadata data is formatted into a .HDF5 container.

    ALL properties are accessed in a lazy manner.

    """
    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def data(self, *args, **kwargs) -> MetadataModel:
        with h5py.File(self.path, "r") as fp:
            attrs = {key: val for (key, val) in fp.attrs.items()} # return all
            # attrs = {key.value: fp.attrs[key.value] for key in MMPD_METADATA_KEYS}

        attrs["video_name"] = self.video_name

        model = MetadataModel(data=None, attrs=attrs)

        return model

    @property
    def video_name(self) -> str:
        return f"{self.subject}_{self.clip}"
    
    @property
    def subject(self) -> int:
        with h5py.File(self.path, "r") as fp:
            val = fp.attrs[MMPD_Keys.SUBJECT.value]
        return val
    
    @property
    def clip(self) -> int:
        with h5py.File(self.path, "r") as fp:
            val = fp.attrs[MMPD_Keys.CLIP.value]
        return val
    
    @property
    def light(self) -> int:
        with h5py.File(self.path, "r") as fp:
            val = fp.attrs[MMPD_Keys.LIGHT.value]
        return val
    
    @property
    def motion(self) -> int:
        with h5py.File(self.path, "r") as fp:
            val = fp.attrs[MMPD_Keys.MOTION.value]
        return val
    
    @property
    def exercise(self) -> int:
        with h5py.File(self.path, "r") as fp:
            val = fp.attrs[MMPD_Keys.EXERCISE.value]
        return val
    
    @property
    def skin_tone(self) -> int:
        with h5py.File(self.path, "r") as fp:
            val = fp.attrs[MMPD_Keys.SKIN_TONE.value]
        return val
    
    @property
    def gender(self) -> int:
        with h5py.File(self.path, "r") as fp:
            val = fp.attrs[MMPD_Keys.GENDER.value]
        return val
    
    @property
    def glasses(self) -> int:
        with h5py.File(self.path, "r") as fp:
            val = fp.attrs[MMPD_Keys.GLASSES.value]
        return val
    
    @property
    def hair_cover(self) -> int:
        with h5py.File(self.path, "r") as fp:
            val = fp.attrs[MMPD_Keys.HAIR_COVER.value]
        return val
    
    @property
    def makeup(self) -> int:
        with h5py.File(self.path, "r") as fp:
            val = fp.attrs[MMPD_Keys.MAKEUP.value]
        return val
