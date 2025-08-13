import cv2
import h5py
import numpy as np
from src.datamodules.datasources.paths.detections import DetectionsSourceFile
from src.datamodules.datapipes import DataOperation, DatasetOperation
from src.datamodules.datamodels.frames import VideoFramesModel
from src.datamodules.datamodels.landmarks import LandmarksModel

from typing import *
from torch import Tensor
from numpy import ndarray


class FormatDatasetFiles(DataOperation):
    def __init__(self, formatter: Callable, *args, **kwargs) -> None:
        super(FormatDatasetFiles, self).__init__(*args, **kwargs)
        self.formatter = formatter

    def __call__(self, paths: Iterable[str], *args, **kwargs) -> Any:
        """ Perform formatting of the provided paths using the formatter.
        """
        return [self.formatter(path) for path in paths] # Return new directory with formatted results


class ExportVideo(DatasetOperation):
    """
    """
    def __init__(self, key: str, filename: str, *args, **kwargs) -> None:
        super(ExportVideo, self).__init__(*args, **kwargs)
        self.key = key
        self.filename = filename

    def apply(self, data: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        # Reference video
        video : VideoFramesModel = data[self.key]

        # Extract root
        root = video.attrs["root"]
        path = root.joinpath(self.filename)        

        # Create `VideoWriter`
        writer = cv2.VideoWriter(
            filename = str(path),
            fourcc = cv2.VideoWriter_fourcc(*'MJPG'),
            fps = video.fps,
            frameSize = [video.width, video.height],
            isColor = video.channels == 3 # 3-channels
        )

        # Extract frames
        video.to_opencv() # convert to opencv
        frames = video.data

        # Process frames
        for frame in frames:
            # Write frames to container
            writer.write(frame)

        # Release stream
        writer.release()


from PIL import Image

class ExportSlicedVideoFrames(DatasetOperation):
    """
    """
    def __init__(self, size: int, key: str, dkey: str, dirname: str, *args, **kwargs) -> None:
        super(ExportSlicedVideoFrames, self).__init__(*args, **kwargs)
        self.key = key
        self.dkey = dkey
        self.dirname = dirname
        self.size = size

    def apply(self, data: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        # Reference video
        video : VideoFramesModel = data[self.key]
        frames = video.data

        # Detection-based frame indexes
        frame_idxs : Set[int] = set(data[self.dkey].frame_indexes(data[self.dkey].data).tolist()) # frame indexes (from detections)
        all_idxs : Set[int] = set(range(0, max(frame_idxs)))
        zero_idxs : Set[int] = all_idxs - (all_idxs & frame_idxs)

        # Extract root
        frame_dir = video.attrs["root"].joinpath(self.dirname)
        frame_dir.mkdir(exist_ok=True, parents=True)

        # Export frames
        # NOTE: Each frame has an associated frame index defined by the detections.
        for idx, frame in zip(frame_idxs, frames): 
            # Filename
            filepath = frame_dir.joinpath(f"frame_{str(idx).zfill(8)}.png")

            # Handle grey-scale
            if frame.shape[-1] == 1:
                frame = np.concatenate([frame]*3, axis=-1)

            # Convert frame
            frame = Image.fromarray(frame)

            # Save frame
            frame.save(filepath)

        # Export empty frames (ensure consistent indexing)
        for idx in zero_idxs:
            # Filename
            filepath = frame_dir.joinpath(f"frame_{str(idx).zfill(8)}.png")

            # Blank frame
            frame = np.zeros((self.size,self.size,3))
            frame = frame.astype(np.uint8)
            frame = Image.fromarray(frame)

            # Save frame
            frame.save(filepath)


def convert_frames_torchvision2opencv(frames: Tensor) -> ndarray:
    """ Convert `frames` from `torchvision` format to `cv2` format.
    """
    frames = frames.permute(0,2,3,1) # [TCHW] to [THWC]
    frames = frames.numpy() # Tensor to ndarray
    frames = (2**8 - 1) * frames # [0-1] to [0-255]
    frames = frames.astype(np.uint8) # [FP32] to [UINT8]
    # frames = cv2.cvtColor(frames, cv2.COLOR_RGB2BGR) # [RGB] to [BGR]
    frames = frames[:,:,:,::-1] # [RGB] to [BRG]
    return frames


def convert_frame_torchvision2opencv(frame: Tensor) -> ndarray:
    """ Convert `frame` from `torchvision` format to `cv2` format.
    """
    frame = frame.permute(1,2,0) # [CHW] to [HWC]
    frame = frame.numpy() # Tensor to ndarray
    frame = (2**8 - 1) * frame # [0-1] to [0-255]
    frame = frame.astype(np.uint8) # [FP32] to [UINT8]
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # [RGB] to [BGR]
    frames = frames[:,:,::-1] # [RGB] to [BRG]
    return frame


def convert_frames_opencv2torchvision(frames: ndarray) -> Tensor:
    """
    """
    frames = frames.astype(np.float32) # [UINT8] to [FP32]
    frames = frames / (2**8 - 1) # [0-255] to [0-1]
    frames = frames[:,:,:,::-1] # [BGR] to [RGB]
    frames = torch.from_numpy(frames.copy()).contiguous() # Non-negative stride contiguous block (done in totensor)
    frames = frames.permute(0,3,1,2) # [THWC] to [TCHW] (done in model.prepare)
    return frames


def convert_frame_opencv2torchvision(frame: ndarray) -> Tensor:
    """
    """
    frame = frame.astype(np.float32) # [UINT8] to [FP32]
    frame = frame / (2**8 - 1) # [0-255] to [0-1]
    frame = frame[:,:,::-1] # [BGR] to [RGB]
    frame = torch.from_numpy(frame.copy()).contiguous() # Non-negative stride contiguous block
    frame = frame.permute(2,0,1) # [HWC] to [CHW]
    return frame


class ExportLandmarks(DatasetOperation):
    """
    """
    def __init__(self, key: str, filename: str, *args, **kwargs) -> None:
        super(ExportLandmarks, self).__init__(*args, **kwargs)
        self.key = key
        self.filename = filename

    def apply(self, data: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        # Reference landmarks
        landmarks : Landmarks = data[self.key]

        # Extract root
        root = landmarks.attrs["root"]
        path = root.joinpath(self.filename)
        # path = root.joinpath(self.filename if self.filename is not None else DetectionsSourceFile.LANDMARKS.value) 

        # Export landmarks
        with h5py.File(path, "w") as fp:
            # Clear existing dataset
            if "data" in fp:
                del fp["data"]

            # Write to dataset
            ds = fp.create_dataset("data", data=landmarks.data)

            # Write ALL availbale attributes
            for key, val in landmarks.attrs.items():
                if key not in ["root"]:
                    ds.attrs.create(key, val)