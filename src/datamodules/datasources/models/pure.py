""" Pulse Rate Detection Dataset - PURE

Dataset:
This data set consists of 10 persons performing different, controlled head motions in front 
of a camera. During these sentences the image sequences of the head as well as reference pulse 
measurements were recorded. The 10 persons (8 male, 2 female) that were recorded in 6 different 
setups resulting in a total number of 60 sequences of 1 minute each.

Recording Setup:
The videos were captured with a eco274CVGE camera by SVS-Vistek GmbH at a frame rate of 30 Hz with 
a cropped resolution of 640x480 pixels and a 4.8mm lens. Reference data have been captured in 
parallel using a finger clip pulse oximeter (pulox CMS50E) that delivers pulse rate wave and 
SpO2 readings with a sampling rate of 60 Hz.

The test subjects were placed in front of the camera with an average distance of 1.1 meters.  

Lighting condition was daylight trough a large window frontal to the face with clouds changing 
llumination conditions slightly over time.


Pulse Rate:
The pulse rate of the test persons varies slightly between and during sequences that do have 
a length of 1 Minute each. Minimum pulse rate measured using the oximeter is at 42 BPM and the 
maximum rate was 148 BPM. Although, we have had very high pulse rates from one subject, all 
recording were taken during rest.

Data:
The images of every single sequence is stored within a seperate folder. The first number of 
the folder name represents the person id, whereas the second number denotes the sequence 
type (ranging from 01 - Steady to 06 - Medium Rotation). Every folder is associated with a 
meta information file that is coded in JSON (JavaScript Object Notation) standard. The files 
do consist of the sensor readings recorded from the finger clip pulse oximeter and the timestamps 
of the images. Please note that the timestamp of the images is also contained in the image file 
names. All timestamps in the meta information file are given as unix timestamp in nanoseconds!

"""
import cv2
import json
import numpy as np
from PIL import Image
from enum import Enum
from src.datamodules.datasources.files import DatasetFile
from src.datamodules.datasources.files.video import FramesFolder
from src.datamodules.datasources.paths.pure import PURE_Default_Files, list_pure_default_files
from src.datamodules.datasources.paths.pure import PURE_Scenario

from typing import *
from numpy import ndarray


class PURE_Default_Frames(FramesFolder):
    """ 

    NOTE: 17/11/23 Added in a class to source frames from default PURE structure due to unreliability
    of FFMPEG/CV2 video streaming. Note, initialization of this will be slow due to globbing and sorting
    of the images within the folder, however should be reliable...

    We provide default PURE file structure for the image folder, however in the pre-processing pipeline
    you may re-name the images. In this case you should provide overrides regular expression matching and
    name separation (for ordering) through argument overrides.

    """
    def __init__(self, 
        regex: Optional[str] = "Image*.png", 
        start: Optional[int] = len("Image"), 
        *args, **kwargs
    ) -> None:
        super(PURE_Default_Frames, self).__init__(regex=regex, start=start, *args, **kwargs)


class PURE_Default_Data(DatasetFile):
    """

    NOTE: It's not recommended to use this interface directly with a `Dataset` as there are 
    significant in-efficiencies in the overhead associated with accessing properties e.g. 
    `video_height`. Instead use this for re-formatting.

    """
    def __init__(self, *args, **kwargs) -> None:
        super(PURE_Default_Data, self).__init__(*args, **kwargs)
        paths = list_pure_default_files(self.path)
        self.data = paths["data"]
        self.frames = paths["images"] # pre-sorted by timestamp

    def data(self) -> None:
        raise NotImplementedError

    @property
    def video_frames_iter(self) -> Generator:
        """ Iterate over sorted frames
        """
        class Frames:
            def __init__(self, frames: Iterable) -> None:
                self.frames = frames

            def __iter__(self) -> Generator:
                frames = self.frames
                for frame in frames:
                    img = Image.open(frame)
                    img = np.array(img)
                    yield img
        return Frames(self.frames)            

    @property
    def video_frames(self) -> ndarray:
        frames = np.array([frame for frame in self.video_frames_iter()])
        return frames

    @property
    def video_timestamps(self) -> ndarray:
        """ Load frame timestamps (nanoseconds).
        """
        with open(self.data, "r") as fp:
            data = json.load(fp)
            timestamps = np.array([val["Timestamp"] for val in data["/Image"]])
        return timestamps

    @property
    def video_length(self) -> int:
        return len(self.frames)

    @property
    def video_height(self) -> int:
        img = Image.open(next(iter(self.frames)))
        return img.height

    @property
    def video_width(self) -> int:
        img = Image.open(next(iter(self.frames)))
        return img.width

    @property
    def average_frame_rate(self) -> float:
        timestamps = self.video_timestamps / 1e9 # timestamps to s
        n_frames = self.video_length
        return n_frames / (timestamps[-1] - timestamps[0])

    @property
    def bvp_samples(self) -> None:
        """ Load BVP sensor waveform.
        """
        with open(self.data, "r") as fp:
            data = json.load(fp)
            samples = np.array([val["Value"]["waveform"] for val in data["/FullPackage"]])
        return samples

    @property
    def num_samples(self) -> int:
        with open(self.data, "r") as fp:
            data = json.load(fp)
        return len(data["/FullPackage"])

    @property
    def pulse_rate_samples(self) -> None:
        """ Load BVP sensor pulse rate reading.
        """
        with open(self.data, "r") as fp:
            data = json.load(fp)
            samples = np.array([val["Value"]["pulseRate"] for val in data["/FullPackage"]])
        return samples

    @property
    def spo2_samples(self) -> None:
        """ Load BVP sensor SpO2 reading.
        """
        with open(self.data, "r") as fp:
            data = json.load(fp)
            samples = np.array([val["Value"]["o2saturation"] for val in data["/FullPackage"]])
        return samples

    @property
    def sample_timestamps(self) -> None:
        """ Load BVP sensor timestamps (nanoseconds).
        """
        with open(self.data, "r") as fp:
            data = json.load(fp)
            timestamps = np.array([val["Timestamp"] for val in data["/FullPackage"]])
        return timestamps

    @property
    def average_sampling_rate(self) -> float:
        timestamps = self.sample_timestamps / 1e9 # ns to s
        return self.num_samples / (timestamps[-1] - timestamps[0])

    @property
    def subject(self) -> int:
        subject : str = self.path.name.split("-")[0]
        return int(subject)

    @property
    def scenario(self) -> int:
        scenario : str = self.path.name.split("-")[1]
        return PURE_Scenario(scenario).name


class PURE_Keys(Enum):
    # Metadata
    SUBJECT = "subject"
    SCENARIO = "scenario"



from src.datamodules.datamodels.timeseries import TimeseriesModel, TimeseriesKeys
from src.datamodules.datasources.paths.pure import PURE_SourceFile


class PURE_BVP(DatasetFile):
    """
    """
    def __init__(self, *args, **kwargs) -> None:
        super(PURE_BVP, self).__init__(*args, **kwargs)

    def data(self, start: Optional[int] = 0, stop: Optional[int] = None) -> ndarray:
        # load samples
        data = self.samples(start=start, stop=stop)

        # load data for attrs
        timestamps = self.timestamps(start=start, stop=stop)
        sps = data.shape[0] / (timestamps[-1] - timestamps[0])

        # create dict
        attrs = {
            TimeseriesKeys.SPS.value: sps,
            TimeseriesKeys.TIMESTAMPS.value: timestamps 
        }

        return TimeseriesModel(data=data, attrs=attrs)        

    def samples(self, start: Optional[int] = 0, stop: Optional[int] = None) -> ndarray:
        with h5py.File(self.path, "r") as fp:
            data = fp["data"][start:stop] if stop is not None else fp["data"][start:]
        return data

    def timestamps(self, start: Optional[int] = 0, stop: Optional[int] = None) -> ndarray:
        with h5py.File(self.path, "r") as fp:
            attr = fp["data"].attrs[TimeseriesKeys.TIMESTAMPS.value]
            data = attr[start:stop] if stop is not None else attr[start:]
        return data / 1e9

    def sps(self, start: Optional[int] = 0, stop: Optional[int] = None) -> float:
        with h5py.File(self.path, "r") as fp:
            # number of samples
            if stop is None:
                n_samples = fp["data"].shape[0] - start
            else:
                n_samples = stop - start

            # timestamps
            timestamps = self.timestamps(start=start, stop=stop)
        
        # sps
        return n_sampes / (timestamps[-1] - timestamps[0])


class PURE_rPPGToolbox_BVP(DatasetFile):
    """
    """
    def __init__(self, *args, **kwargs) -> None:
        super(PURE_rPPGToolbox_BVP, self).__init__(*args, **kwargs)

    def data(self, start: Optional[int] = 0, stop: Optional[int] = None) -> ndarray:
        # load samples
        data = self.samples(start=start, stop=stop)

        # load data for attrs
        sps = 30.0

        # create dict
        attrs = {
            TimeseriesKeys.SPS.value: sps
        }

        return TimeseriesModel(data=data, attrs=attrs)        

    def samples(self, start: Optional[int] = 0, stop: Optional[int] = None) -> ndarray:
        with h5py.File(self.path, "r") as fp:
            data = fp["data"][start:stop] if stop is not None else fp["data"][start:]
        return data

    def sps(self, *args, **kwargs) -> float:
        return 30.0


class PURE_Timestamps(DatasetFile):
    def __init__(self, *args, **kwargs) -> None:
        super(PURE_Timestamps, self).__init__(*args, **kwargs)

    def data(self, start: Optional[int] = 0, stop: Optional[int] = None) -> ndarray:
        if stop is not None:
            data = np.loadtxt(self.path, skiprows=start, max_rows=stop-start) / 1e9 # ns to s conversion
        else:
            data = np.loadtxt(self.path, skiprows=start) / 1e9 
        return data


from src.datamodules.datasources.files.video import FFMPEGVideoFile
from src.datamodules.datamodels.frames import VideoFramesModel, FramesKeys

class PURE_Video(DatasetFile):
    """ Our implementation of the video handling for the PURE Dataset.

    This implementation relies on the following dataset directory structure:
        <root>/
            <01-01>/ # Video-clip
                video.avi
                timestamps.txt # Provides frame-time information
            ...

    """
    def __init__(self, relpath: Optional[str] = None, video_kwargs: Optional[Dict[str,Any]] = {}, *args, **kwargs) -> None:
        super(PURE_Video, self).__init__(*args, **kwargs)
        
        # timestamps
        filepath = self.path.joinpath(PURE_SourceFile.VIDEO_TIMESTAMPS.value)
        self.timestamps = PURE_Timestamps(filepath)

        # video
        framedir = self.path.joinpath(relpath) if relpath is not None else self.path.joinpath(self.path.name)
        self._video = PURE_Default_Frames(path=framedir, **video_kwargs)

    def data(self, start: Optional[int] = 0, stop: Optional[int] = None, *args, **kwargs) -> Dict[str, Any]:
        # Read video
        data : ndarray = self.frames(start=start, stop=stop)

        # timestamps
        timestamps : ndarray = self.timestamps.data(start=start, stop=stop)

        # sps
        fps = data.shape[0] / (timestamps[-1] - timestamps[0]) # FPS depends on video slice used

        # Extract capture attributes
        attrs = {
            FramesKeys.VIDEO_FPS.value: fps,
            FramesKeys.VIDEO_LENGTH.value: data.shape[0],
            TimeseriesKeys.TIMESTAMPS.value: timestamps
        }

        # Create `DataModel`
        model = VideoFramesModel(data=data, format="opencv", attrs=attrs)

        return model

    def frames(self, start: Optional[int] = 0, stop: Optional[int] = None, *args, **kwargs) -> ndarray:
        return self._video.data(start=start, stop=stop) # THWC

    @property
    def video_length(self) -> int:
        return self._video.frame_count()
    
    @property
    def video_height(self) -> int:
        return self._video.frame_height()
    
    @property
    def video_width(self) -> int:
        return self._video.frame_width()

    def video_fps(self, start: Optional[int] = 0, stop: Optional[int] = None) -> float:
        # number of samples
        n_samples = self.video_length if stop is None else stop - start

        # timestamps
        timestamps = self.timestamps.data(start=start, stop=stop)
        
        # sps
        return n_samples / (timestamps[-1] - timestamps[0])


class PURE_rPPGToolbox_Video(DatasetFile):
    """ rPPG-Toolbox implementation of the video handling for the PURE Dataset.

    This implementation makes the following assumptions:
        - Video is sampled at a fixed and uniform 30 SPS
        - BVP is sampled at a fixed and uniform 30 SPS

    This implementation relies on the following dataset directory structure:
        <root>/
            <01-01>/ # Video-clip
                video.avi # Assume constant 30 FPS
            ...

    """
    def __init__(self,
        relpath: Optional[str] = None, 
        video_kwargs: Optional[Dict[str,Any]] = {}, 
        *args, **kwargs
    ) -> None:
        super(PURE_rPPGToolbox_Video, self).__init__(*args, **kwargs)

        # Video frames from directory
        frame_dir = self.path.joinpath(relpath) if relpath is not None else self.path.joinpath(self.path.name)
        self._video = PURE_Default_Frames(path=frame_dir, **video_kwargs)

    def data(self, start: Optional[int] = 0, stop: Optional[int] = None, *args, **kwargs) -> Dict[str, Any]:
        # Read video
        data : ndarray = self.frames(start=start, stop=stop)

        # sps
        fps = 30.0

        # Extract capture attributes
        attrs = {
            FramesKeys.VIDEO_FPS.value: fps,
            FramesKeys.VIDEO_LENGTH.value: data.shape[0]
        }

        # Create `DataModel`
        model = VideoFramesModel(data=data, format="opencv", attrs=attrs)

        return model

    def frames(self, start: Optional[int] = 0, stop: Optional[int] = None, *args, **kwargs) -> ndarray:
        return self._video.data(start=start, stop=stop) # THWC

    @property
    def video_length(self) -> int:
        return self._video.frame_count()
    
    @property
    def video_height(self) -> int:
        return self._video.frame_height()
    
    @property
    def video_width(self) -> int:
        return self._video.frame_width()

    def video_fps(self, *args, **kwargs) -> float:
        return 30.0



import h5py
from src.datamodules.datasources.paths.pure import PURE_Scenario
from src.datamodules.datamodels import MetadataModel

class PURE_Metadata(DatasetFile):
    """
    """
    def __init__(self, *args, **kwargs) -> None:
        super(PURE_Metadata, self).__init__(*args, **kwargs)

    def data(self, *args, **kwargs) -> MetadataModel:
        with h5py.File(self.path, "r") as fp:
            attrs = {key: val for (key, val) in fp.attrs.items()}
        attrs["video_name"] = self.video_name
        return MetadataModel(data=None, attrs=attrs)

    def metadata(self) -> Dict[str, Any]:
        return self

    @property
    def video_name(self) -> str:
        return f"{self.subject}_{self.scenario}"

    @property
    def subject(self) -> str:
        with h5py.File(self.path, "r") as fp:
            val = fp.attrs[PURE_Keys.SUBJECT.value]
        return val

    @property
    def scenario(self) -> str:
        with h5py.File(self.path, "r") as fp:
            val = fp.attrs[PURE_Keys.SCENARIO.value]
        return val