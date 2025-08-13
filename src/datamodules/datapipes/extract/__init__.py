import cv2
import h5py
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from abc import ABC, abstractmethod
from src.datamodules.datapipes import DataOperation
from src.datamodules.datasources.files.video import FFMPEGVideoFile 
from src.datamodules.datasources.files.detections import DetectionKeys

from typing import *
from numpy import ndarray

log = logging.getLogger(__name__)


class ExtractorModel(ABC):
    """ Base-class for Video Extraction

    # TODO: Generalize this to an iterable e.g. frames, images, or timeseries

    Returns:
        _type_: _description_
    """
    def __init__(self, model: Callable, dattrs: Optional[Dict[str, Any]] = {}) -> None:
        super(ExtractorModel, self).__init__()
        self.dattrs = dattrs 
        self.model = model

    def __call__(self, frame: ndarray, index: int, *args, **kwargs) -> List[ndarray]:
        frame = self.preprocess(frame)
        results = self.update(frame, *args, **kwargs)
        outputs = self.postprocess(results, frame, index)
        return outputs
    
    @property
    @staticmethod
    @abstractmethod
    def algorithm() -> str:
        """ Detection algorithm used by the `DetectorModel`.
        """
        pass

    @property
    @staticmethod
    @abstractmethod
    def detection_type() -> str:
        """ Detection type output by the `algorithm`.
        """
        pass

    @property
    @staticmethod
    @abstractmethod
    def detection_format() -> str:
        """ Detection type format output by the `algorithm`.
        """
        pass

    @property
    @staticmethod
    @abstractmethod
    def filename() -> str:
        """ Detection type format output by the `algorithm`.
        """
        pass
    
    @property
    def attrs(self) -> Dict[str, Any]:
        """ Attributes to be used to described the specific implementations.
        """
        static_attributes = {
            "algorithm": self.algorithm,
            "detection_type": self.detection_type,
            "detection_format": self.detection_format
        }
        dynamic_attributes = self.dattrs

        return {**static_attributes, **dynamic_attributes}

    @abstractmethod
    def preprocess(self, frame: ndarray) -> ndarray:
        """ Perform model-specific image pre-processing required for the detector.
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        """
        """
        pass

    @abstractmethod
    def postprocess(self, results: Any, frame: ndarray, index: int) -> ndarray:
        """ Perform model-specific result post-processing to parse into unified format.
        """
        pass



class ExtractVideoFeatures(DataOperation):
    """ `DataOperation` which performs feature extraction and formatting of the input based on a provided
    algorithm.

    Common use-cases include performing object detection (`Detector` and `Tracker`), landmark detection,
    semantic segmentation, etc.

    # TODO: Convert this to a model standard inference process using LightningModules for distributed prediction.
    """
    def __init__(self, model: ExtractorModel, skip_existing: Optional[bool] = True, *args, **kwargs) -> None:
        super(ExtractVideoFeatures, self).__init__(*args, **kwargs)
        self.model = model
        self.skip_existing = skip_existing

    def __call__(self, path: str) -> str:
        """ Extract video frames by applying the `model` to the video data.
        """
        with tqdm() as pbar:
            # Video Streamer
            video = FFMPEGVideoFile(path)

            # Destination file
            filepath = video.path.parent.joinpath(self.model.filename)
            rname = f"{video.path.parent.name}/{self.model.filename}"

            if filepath.exists() and self.skip_existing:
                pbar.set_description(f"[{rname}] Already exists, skipped.")

            else:
                # Create destination file
                with h5py.File(filepath, "w") as fp:
                    # Stream frames from video container
                    results = []
                    pbar.set_description(f"[{rname}] Extracting video features: {self.model.algorithm}...")
                    for idx, frame in tqdm(enumerate(video), total=video.frame_count(), desc=f"[{video.path.name}]"):
                        # Detect features in frame
                        result = self.model(frame, idx)

                        # Accumulate results
                        for item in result:
                            results.append(item)

                    #
                    pbar.set_description(f"[{rname}] Exporting extracted features...")

                    # Conditionally 
                    if "data" in fp:
                        del fp["data"]

                    # Save existing dataset
                    results = np.stack(results)
                    # results = results.astype(np.float32) # FP64 to FP32 (should put this elsewhere)
                    dataset = fp.create_dataset("data", data=results)

                    # Assign attributes
                    for key, val in self.model.attrs.items():
                        dataset.attrs.create(key, val)

                    # Assign time attributes
                    dataset.attrs[DetectionKeys.CREATED.value] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

                    pbar.set_description(f"[{rname}] Completed extraction: {self.model.algorithm}.")

            return filepath


class ExtractFeaturesFromVideos(DataOperation):
    def __init__(self, extractor: Callable, *args, **kwargs) -> None:
        super(ExtractFeaturesFromVideos, self).__init__(*args, **kwargs)
        self.extractor = extractor

    def __call__(self, paths: Iterable[str], *args, **kwargs) -> Any:
        """ Perform formatting of the provided paths using the formatter.
        """
        return [self.extractor(path) for path in paths] # Return new directory with formatted results



from pathlib import Path
from src.datamodules.datasources.paths.detections import DetectionsSourceFile
from src.datamodules.datasources.files.detections import BoundingBoxesFile, LandmarksFile


class InterpolateDetections(DataOperation):
    """ Interpolate detections based on the frame indexes through linear interpolations.
    """
    def __init__(self, key: Optional[str] = None, max_interpolate: Optional[int] = 1, skip_existing: Optional[bool] = True, *args, **kwargs) -> None:
        super(InterpolateDetections, self).__init__(*args, **kwargs)

        # Detections to interpolate (defaults to all)
        self.key = key

        self.max_interpolate = max_interpolate
        self.skip_existing = skip_existing

    @property
    @abstractmethod
    def detection_model(self) -> Callable:
        pass

    @abstractmethod
    def update_frame_index(self, data: ndarray, index: int) -> ndarray:
        """ Update the associated frame index for a single Detection result.
        """
        pass

    def __call__(self, path: str) -> str:
        """
        """
        # Logging
        rname = f"{'/'.join(Path(path).parts[-2:])}:{self.key}]"

        with tqdm() as pbar:
            # Open detection file
            with h5py.File(path, "a") as fp:
                keys = [self.key] if self.key is not None else list(fp.keys())

                for key in keys:
                    # Flags
                    already_interpolated = "interpolated" in fp[key].attrs.keys()

                    # Conditionally interpolate
                    if already_interpolated and self.skip_existing:
                        pbar.set_description(f"[{rname}] Already interpolated, skipping...")

                    else:
                        # Read detections
                        detections = fp[key][:]

                        # Extract current frame indexes : will incur read
                        frame_idxs = self.detection_model.frame_indexes(detections)

                        # Compute differences in frame indexes (prepend to ensure aligned diff is 1)
                        diff_idxs = np.diff(frame_idxs, prepend=frame_idxs[0]-1)
                        mask = (diff_idxs > 1) & (diff_idxs <= self.max_interpolate + 1) # max_interp = max. no. missing frames
                        miss_idxs = np.where(mask)[0]

                        # Perform interpolation between detections
                        if len(miss_idxs) > 0:
                            # Prepare
                            val = detections # Read all detections
                            interp_idxs = []

                            pbar.set_description(f"[{rname}] Performing interpolation...")

                            for miss_idx in tqdm(miss_idxs):
                                # Blend factors between initial/final points
                                alphas = np.linspace(0, 1, diff_idxs[miss_idx]+1, endpoint=True)[1:-1] # blend factors

                                # Perform linear interpolation between initial/final points
                                for idx, alpha in enumerate(alphas):
                                    # Compute blend of landmark values:
                                    '''
                                    bbox: [4]
                                    landmark: [478,4]
                                    mask: [H,W]
                                    '''
                                    res = (1 - alpha) * val[miss_idx] + alpha * val[miss_idx-1]

                                    # Assign frame index
                                    res = self.update_frame_index(res, frame_idxs[miss_idx-1] + idx + 1)

                                    # Update
                                    interp_idxs.append(miss_idx + idx + 1)
                                    val = np.append(val, [res], axis=0)

                            #
                            pbar.set_description(f"[{rname}] Exporting interpolated detections...")

                            # Sort by frame indexes (now ready to save)
                            sorted_idxs = np.argsort(self.detection_model.frame_indexes(val))
                            new_data = val[sorted_idxs]

                            # Extract attributes to re-assign to new dataset
                            attrs = {key:val for (key,val) in fp.file[key].attrs.items()}
                            attrs["interpolated"] = interp_idxs

                            # Overwrite dataset with interpolated frames
                            if key in fp.file:
                                del fp.file[key]
                            dataset = fp.file.create_dataset(key, data=new_data)
                            for key, val in attrs.items():
                                dataset.attrs.create(key, val)

                            pbar.set_description(f"[{rname}] Completed interpolation")
                        
                        else:
                            num_missing_frames = len(np.where(diff_idxs > 1)[0])
                            if num_missing_frames > 0:
                                pbar.set_description(f"[{rname}] Cannot be interpolated (missing={num_missing_frames})")
                            else:
                                pbar.set_description(f"[{rname}] Does not require interpolation")

        return path


class InterpolateDetectionsFromPaths(DataOperation):
    def __init__(self, interpolator: Callable, *args, **kwargs) -> None:
        super(InterpolateDetectionsFromPaths, self).__init__(*args, **kwargs)
        self.interpolator = interpolator

    def __call__(self, paths: Iterable[str], *args, **kwargs) -> Any:
        """ Perform formatting of the provided paths using the formatter.
        """
        return [self.interpolator(path) for path in paths] # Return new directory with formatted results
