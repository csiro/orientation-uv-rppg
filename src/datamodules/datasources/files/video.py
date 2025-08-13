import cv2
import warnings
import numpy as np
from enum import Enum
from src.datamodules.datasources.files import DatasetFile

from typing import *
from numpy import ndarray


class FrameProperties(Enum):
    FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
    FRAME_FPS = cv2.CAP_PROP_FPS



class VideoFile(DatasetFile):
    """ Video file with the backend set to image mode (standard).
    """
    def __init__(self, *args, **kwargs) -> None:
        super(VideoFile, self).__init__(*args, **kwargs)

    def data(self, start: Optional[int] = 0, stop: Optional[int] = None) -> ndarray:
        # prepare
        idx = 0
        frames = []
        
        # create cap
        cap = cv2.VideoCapture(str(self.path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start) # start reading from this point
        
        while True:
            flag, frame = cap.read() # load flag
            if flag:
                idx += 1
                frames.append(frame)
                if stop is not None:
                    if idx >= stop - start:
                        break
            else:
                break # reached end of stream OR failed (unhandled)
        
        # release cap
        cap.release()

        return np.array(frames)

    # def __iter__(self) -> Generator:
    #     idx = 0

    #     cap = cv2.VideoCapture(str(self.path))
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

    #     while True:
    #         flag, frame = cap.read() # CAP_PROP_POS_FRAMES will automatically increment
    #         if flag:
    #             idx += 1
    #             yield frame
    #         else:
    #             break

    #     cap.release()

    def read_property(self, prop, cap: Optional[Any] = None) -> Any:
        if cap is None:
            cap = cv2.VideoCapture(str(self.path))
            prop_val = cap.get(prop)
            cap.release()
        else:
            prop_val = cap.get(prop)
        return prop_val

    def frame_count(self, cap: Optional[Any] = None) -> int:
        return int(self.read_property(cv2.CAP_PROP_FRAME_COUNT, cap))
    
    def frame_height(self, cap: Optional[Any] = None) -> int:
        return int(self.read_property(cv2.CAP_PROP_FRAME_HEIGHT, cap))
    
    def frame_width(self, cap: Optional[Any] = None) -> int:
        return int(self.read_property(cv2.CAP_PROP_FRAME_WIDTH, cap))
    
    def frame_size(self, cap: Optional[Any] = None) -> Tuple[int, int]:
        return (self.frame_height(cap), self.frame_width(cap))
    
    def frame_fps(self, cap: Optional[Any] = None) -> float:
        return float(self.read_property(cv2.CAP_PROP_FPS, cap))
    

def write_video(path: str, frames: ndarray, **kwargs) -> None:
    writer = cv2.VideoWriter(path, **kwargs)
    for frame in frames:
        writer.write(frame)
    writer.release()


class FFMPEGVideoFile(VideoFile):
    """ Video file with the cv2 backend set to `FFMPEG` for video decoding.
    """
    def data(self, start: Optional[int] = 0, stop: Optional[int] = None) -> ndarray:
        # prepare
        idx = 0
        frames = []
        
        # create cap
        cap = cv2.VideoCapture(str(self.path), cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start) # start reading from this point
        
        while True:
            flag, frame = cap.read() # load flag
            if flag:
                idx += 1
                frames.append(frame)
                if stop is not None:
                    if idx >= stop - start:
                        break
            else:
                break # reached end of stream OR failed (unhandled)
        
        # release cap
        cap.release()

        return np.array(frames)

    def __iter__(self) -> Generator:
        # Stream from initial frame
        idx = 0

        # Open video and set frame position
        cap = cv2.VideoCapture(str(self.path), cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        # Stream frames
        while True:
            flag, frame = cap.read() # CAP_PROP_POS_FRAMES will automatically increment
            if flag:
                idx += 1
                yield frame
            else:
                break

        cap.release()



# TODO: Determine alternative for loading blank frames
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class FramesFolder(DatasetFile):
    """ 
    
    Directory containing the video frame images must have the following format:
        <directory>/
            <regex><monotonically increasing int number per frame>.png

    """
    def __init__(self, regex: str, start: int, *args, **kwargs) -> None:
        super(FramesFolder, self).__init__(*args, **kwargs)
        self.paths = sorted(self.path.glob(regex), key = lambda p: int(p.stem[start:]))

    def data(self, start: Optional[int] = 0, stop: Optional[int] = None, *args, **kwargs) -> ndarray:
        """
        TODO: Determine if there is a lazy way to slice into paths upon request instead of constructing list by default
        """
        frames = []
        paths = self.paths[start:stop] if stop is not None else self.paths[start:]
        for path in paths:
            # Resolve path (symbolic links)
            path = path.resolve().absolute()

            # Load frame
            frame = Image.open(path)
            frame = np.array(frame)
            frames.append(frame)

        frames = np.array(frames)
        return frames

    def frame_count(self, *args, **kwargs) -> int:
        return len(self.paths)

    def video_height(self, *args, **kwargs) -> int:
        frame = cv2.imread(str(self.paths[0])) # H,W,C
        return frame.shape[0]
    
    def video_width(self, *args, **kwargs) -> int:
        frame = cv2.imread(str(self.paths[0])) # H,W,C
        return frame.shape[1]