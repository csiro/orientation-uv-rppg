from pathlib import Path
from functools import cached_property
from src.datamodules.datasources.models import DatasetModel
from src.datamodules.datasources.files.mmpd import MMPD_Keys

from typing import *
from numpy import ndarray


class MMPD_DatasetModel_Raw(DatasetModel):
    """
    """
    def __init__(self, *args, **kwargs) -> None:
        super(MMPD_DatasetModel_Raw, self).__init__(*args, **kwargs)

    @property
    def video(self) -> ndarray:
        return self.data[MMPD_Keys.VIDEO.value]
    
    @property
    def video_fps(self) -> float:
        return self.data[MMPD_Keys.VIDEO_FPS.value]

    @property
    def gt_bvp(self) -> ndarray:
        return self.data[MMPD_Keys.GT_BVP.value]
    
    @property
    def gt_bvp_sps(self) -> ndarray:
        return self.data[MMPD_Keys.GT_BVP_SPS.value]
    
    @property
    def subject(self) -> int:
        return self.data[MMPD_Keys.SUBJECT.value]
    
    @property
    def clip(self) -> int:
        return self.data[MMPD_Keys.CLIP.value]
    
    @property
    def light(self) -> str:
        return self.data[MMPD_Keys.LIGHT.value]
    
    @property
    def motion(self) -> str:
        return self.data[MMPD_Keys.MOTION.value]
    
    @property
    def exercise(self) -> str:
        return self.data[MMPD_Keys.EXERCISE.value]
    
    @property
    def skin_tone(self) -> str:
        return self.data[MMPD_Keys.SKIN_TONE.value]
    
    @property
    def gender(self) -> str:
        return self.data[MMPD_Keys.GENDER.value]
    
    @property
    def glasses(self) -> str:
        return self.data[MMPD_Keys.GLASSES.value]
    
    @property
    def hair_cover(self) -> str:
        return self.data[MMPD_Keys.HAIR_COVER.value]
    
    @property
    def makeup(self) -> str:
        return self.data[MMPD_Keys.MAKEUP.value]



class MMPD_DatasetModel(MMPD_DatasetModel_Raw):
    """
    """
    @property
    def video_length(self) -> int:
        return self.data[MMPD_Keys.VIDEO_LENGTH.value]
    
    @property
    def video_height(self) -> int:
        return self.data[MMPD_Keys.VIDEO_HEIGHT.value]
    
    @property
    def video_width(self) -> int:
        return self.data[MMPD_Keys.VIDEO_WIDTH.value]

    










# class MMPDRaw_DatasetModel(DatasetModel):
#     """ DataModel for raw `MMPD` dataset files.

#     """
#     def __init__(self, data: Dict[str, Any], *args, **kwargs) -> None:
#         super(MMPDRaw_DatasetModel, self).__init__(data, *args, **kwargs)

#     @property
#     def frames(self) -> ndarray:
#         return self.data[self.frames_loc] # (T,H,W,C)
    
#     @property
#     def frames_sps(self) -> float:
#         return float(self.data[self.frames_sps_loc])
    
#     @property
#     def frames_format(self) -> str:
#         return str(self.data[self.frames_format_loc])
    
#     @property
#     def labels(self) -> ndarray:
#         return np.squeeze(self.data[self.labels_loc], axis=0) # (1,T) -> (T)
    
#     @property
#     def labels_sps(self) -> float:
#         return float(self.data[self.labels_sps_loc])
    
#     @property
#     def light(self) -> str:
#         return str(self.data[self.light_loc][0])
    
#     @property
#     def motion(self) -> str:
#         return str(self.data[self.motion_loc][0])
    
#     @property
#     def exercise(self) -> bool:
#         return str2bool(self.data[self.exercise_loc][0])
    
#     @property
#     def skin_tone(self) -> int:
#         return int(self.data[self.skin_tone_loc][0,0])
    
#     @property
#     def gender(self) -> str:
#         return str(self.data[self.gender_loc][0])
    
#     @property
#     def glasses(self) -> bool:
#         return str2bool(self.data[self.glasses_loc][0])
    
#     @property
#     def hair_cover(self) -> bool:
#         return str2bool(self.data[self.hair_cover_loc][0])
    
#     @property
#     def makeup(self) -> bool:
#         return str2bool(self.data[self.makeup_loc][0])
    
#     @property
#     def subject(self) -> int:
#         return int(self.data[self.subject_loc])
    
#     @property
#     def clip(self) -> int:
#         return int(self.data[self.clip_loc])

    
# from h5py import File as H5PYFile
# from h5py import Dataset as H5PYDataset

# from src.datamodules.datamodels.frames import VideoFramesModel
# from src.datamodules.datamodels.timeseries import TimeseriesModel

# class MMPDProcessedModel(DatasetModel):
#     """ `DatasetModel` for processed `MMPD` dataset files.

#     Processed MMPD Format:
#         <attrs>
#             light (str)
#             motion (str)
#             exercise (bool)
#             skin_color (int) # Fitzpatrick Skin-tone
#             gender (str)
#             glasses (bool)
#             hair_cover (bool)
#             makeup (bool)
#         frames (ndarray): 
#             <attrs>
#                 sps (float)
#                 format (str)
#                 ...
#         labels (ndarray):
#             <attrs>
#                 sps (float)

#     NOTE: We can explicitly seperated the formatting of the raw to processed `MMPD` file
#     into a different area, this is NOT ideal as it decouples the interface logic from the 
#     construction logic (which we want to be the same). However, until we determine a better
#     methods of accessing references without causing IO this works well during train-time.

#     Args:
#         DatasetModel (_type_): _description_
#     """
#     # Locations
#     frames_loc          : str = "/frames"
#     frames_sps_loc      : str = "/frames/sps"
#     frames_format_loc   : str = "/frames/format"
#     labels_loc          : str = "/labels"
#     labels_sps_loc      : str = "/labels/sps"
#     light_loc           : str = "/light"
#     motion_loc          : str = "/motion"
#     exercise_loc        : str = "/exercise"
#     skin_tone_loc       : str = "/skin_tone"
#     gender_loc          : str = "/gender"
#     glasses_loc         : str = "/glasses"
#     hair_cover_loc      : str = "/hair_cover"
#     makeup_loc          : str = "/makeup"
#     subject_loc         : str = "/subject"
#     clip_loc            : str = "/clip"

#     # Attributes
#     attribute_names = [
#         "subject", "clip", "motion", "light", "skin_tone", "exercise", "gender", "glasses", "hair_cover", "makeup"
#     ]

#     def __init__(self, data: H5PYFile, *args, **kwargs) -> None:
#         """_summary_

#         Args:
#             data (H5PYFile): _description_
#             detection (Optional[str], optional): _description_. Defaults to None.
#             start (Optional[int], optional): _description_. Defaults to 0.
#             stop (Optional[int], optional): _description_. Defaults to None.
#         """
#         super(MMPDProcessedModel, self).__init__(data, *args, **kwargs)
#         # assert type(self.data) == H5PYFile, f"Require {H5PYFile} to initialize `DatasetModel`."

#     @cached_property
#     def frames(self) -> H5PYDataset:
#         """_summary_

#         NOTE: `cached_property` approach will write the property is an attribute using the `settr` method
#         if it doesn't exist, hence allowing us to maintain a reference to the instantiated item, reducing
#         execution time for following accesses. Should NOT be used if the method call reqires variables as
#         the cache can grow without bound.

#         NOTE: We handle caching of the underlying data slices within the specific `DataModel` instances and
#         only cache references to the handling class here.

#         Returns:
#             ndarray: _description_
#         """
#         return VideoFramesModel(data=self.data[self.frames_loc], start=self.start, stop=self.stop)
    
#     @property
#     def frames_sps(self) -> float:
#         parent, name = self.location(self.frames_sps_loc)
#         return self.data[parent].attrs[name]
    
#     @property
#     def frames_format(self) -> str:
#         parent, name = self.location(self.frames_format_loc)
#         return self.data[parent].attrs[name]
    
#     @cached_property
#     def labels(self) -> H5PYDataset:
#         return TimeseriesModel(self.data[self.labels_loc], start=self.start, stop=self.stop)
    
#     @property
#     def labels_sps(self) -> float:
#         parent, name = self.location(self.labels_sps_loc)
#         return self.data[parent].attrs[name]

#     @property
#     def light(self) -> str:
#         return self.data.attrs[Path(self.light_loc).name]
    
#     @property
#     def motion(self) -> str:
#         return self.data.attrs[Path(self.motion_loc).name]
    
#     @property
#     def exercise(self) -> bool:
#         return self.data.attrs[Path(self.exercise_loc).name]
    
#     @property
#     def skin_tone(self) -> int:
#         return self.data.attrs[Path(self.skin_tone_loc).name]
    
#     @property
#     def gender(self) -> str:
#         return self.data.attrs[Path(self.gender_loc).name]
    
#     @property
#     def glasses(self) -> bool:
#         return self.data.attrs[Path(self.glasses_loc).name]
    
#     @property
#     def hair_cover(self) -> bool:
#         return self.data.attrs[Path(self.hair_cover_loc).name]
    
#     @property
#     def makeup(self) -> bool:
#         return self.data.attrs[Path(self.makeup_loc).name]
    
#     @property
#     def subject(self) -> int:
#         return self.data.attrs[Path(self.subject_loc).name]
    
#     @property
#     def clip(self) -> int:
#         return self.data.attrs[Path(self.clip_loc).name]
    
#     @property
#     def attributes(self) -> Dict[str, Union[str, int, bool]]:
#         return {key: getattr(self, key) for key in self.attribute_names}