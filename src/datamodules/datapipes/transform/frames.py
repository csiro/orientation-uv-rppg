from typing import Optional
import cv2
import copy
import torch
import numpy as np
import torchvision
from tqdm import tqdm
import torchvision.transforms.functional as tv_f
from src.datamodules.datapipes import DatasetOperation
from src.datamodules.datapipes.transform import Difference, Normalize, Standardize, NormalizedDifference
from src.datamodules.datamodels.boxes import BoundingBoxesModel


from typing import *
from numpy import ndarray
from torch import Tensor


class ToFloat(DatasetOperation):
    """ Perform datatype conversion to FP32 and scale results to 0-1.

    NOTE: We typically define a specific transformation for images from the loaded
    UINT8 0-255 format to FP32 0-1 before returning the data sample, as image
    processing libraries are typically optimized for UINT8.
    """
    def __init__(self, fkey: str, max_value: float, *args, **kwargs) -> None:
        super(ToFloat, self).__init__(*args, **kwargs)
        self.fkey = fkey
        self.max_value = max_value
    
    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        data[self.fkey].data = data[self.fkey].data.astype(np.float32) # cast to float
        data[self.fkey].data = data[self.fkey].data / self.max_value # scale


class ToInt(DatasetOperation):
    """ Perform datatype conversion to UINT8/16/32 and scale results to 0-max_val.
    """
    def __init__(self, fkey: str, max_value: float, dtype: str, *args, **kwargs) -> None:
        super(ToInt, self).__init__(*args, **kwargs)
        self.fkey = fkey
        self.max_value = max_value
        self.dtype = dtype
        assert self.dtype in ["uint8", "uint16", "uint32"]
    
    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        data[self.fkey].data = self.max_value * data[self.fkey].data # scale
        data[self.fkey].data = data[self.fkey].data.astype(getattr(np, self.dtype)) # cast to int
        


class FrameDifference(Difference):
    """ Perform an forward differencing operation along the temporal axis.
    
    Compute the n-th order forward difference between frames along the time-axis.
    frame[i] = frame[i+1] - frame[i] : Applied n times recursively
    """
    def __init__(self, frames: str, *args, **kwargs) -> None:
        super(FrameDifference, self).__init__(key=frames, *args, **kwargs)


class NormalizeFrames(Normalize):
    """ Normalize the frames by minimum/maximum
    frame[i] = (frame[i] - min) / (max - min)
    """
    def __init__(self, frames: str, *args, **kwargs) -> None:
        super(NormalizeFrames, self).__init__(key=frames, *args, **kwargs)


class StandardizeFrames(Standardize):
    """ Standardize the frames such that mean=0, var=1.
    frame[i] = (frame[i] - mean) / std
    """
    def __init__(self, frames: str, *args, **kwargs) -> None:
        super(StandardizeFrames, self).__init__(key=frames, *args, **kwargs)


class NormalizedFrameDifference(NormalizedDifference):
    """ Compute the normalized frame difference between frames.
    frame[i] = (frame[i+1] - frame[i]) / (frame[i+1] + frame[i])
    """
    def __init__(self, frames: str, *args, **kwargs) -> None:
        super(NormalizedFrameDifference, self).__init__(key=frames, *args, **kwargs)


class VideoChannelNormalization(DatasetOperation):
    """ Perform per-channel normalization of a single video based on the video statistics 
    or from a pre-computed mean and std if provided.
    """
    def __init__(self, 
        key: str, 
        mean: Optional[List[float]] = None, 
        std: Optional[List[float]] = None, 
        *args, **kwargs
    ) -> None:
        super(VideoChannelNormalization, self).__init__(*args, **kwargs)

        self.key = key

        self.mean = torch.Tensor(mean).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) if mean is not None else None
        self.std = torch.Tensor(std).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) if std is not None else None

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        # Extract frames
        frames = data[self.key].data

        # Apply normalization
        if self.mean is None:
            mean = torch.mean(frames, dim=(0,2,3)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            std = torch.std(frames, dim=(0,2,3)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            frames = (frames - mean) / std
        
        else: # apply from pre-computed mean/std
            frames = (frames - self.mean) / self.std

        # Assign
        data[self.key].data = frames


class VideoNormalization(DatasetOperation):
    """ Perform normalization of a single video based on the video statistics 
    or from a pre-computed mean and std if provided.
    """
    def __init__(self, 
        key: str, 
        mean: Optional[List[float]] = None, 
        std: Optional[List[float]] = None, 
        *args, **kwargs
    ) -> None:
        super(VideoNormalization, self).__init__(*args, **kwargs)

        self.key = key

        self.mean = torch.Tensor(mean) if mean is not None else None
        self.std = torch.Tensor(std) if std is not None else None

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        # Extract frames
        frames = data[self.key].data

        # Normalization
        frames = frames - torch.mean(frames) if self.mean is None else frames - self.mean
        frames = frames / torch.std(frames) if self.std is None else frames / self.std

        # Assign
        data[self.key].data = frames





class ScaleOffsetFrames(DatasetOperation):
    def __init__(self, frames: str, scale: float, offset: float, *args, **kwargs) -> None:
        super(ScaleOffsetFrames, self).__init__(*args, **kwargs)
        self.fkey = frames
        self.scale = scale
        self.offset = offset

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        # Apple scale and offset
        data[self.fkey].data = self.scale * data[self.fkey].data + self.offset


class ClipFramesNSigma(DatasetOperation):
    def __init__(self, frames: str, n: float, *args, **kwargs) -> None:
        super(ClipFramesNSigma, self).__init__(*args, **kwargs)
        self.fkey = frames
        self.n = n

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        # Get frames
        frames = data[self.fkey].data

        # Compute mean and std
        mean = torch.mean(data[self.fkey].data) # compute over video and all channels
        std = torch.std(data[self.fkey].data) 

        # Minimum/maximum bounds
        min = mean - self.n * std
        max = mean + self.n * std

        # Clip
        frames[frames < min] = min
        frames[frames > max] = max

        data[self.fkey].data = frames 



class PadFrames(DatasetOperation):
    """ Pad frames to the specified height and width
    """
    def __init__(self, fkey: str, dkey: str, size: List[int] = [None, None], value: Optional[int] = 0, *args, **kwargs) -> None:
        super(PadFrames, self).__init__(*args, **kwargs)
        self.fkey = fkey
        self.dkey = dkey
        self.size = size
        assert len(size) == 2, f"Must provide [height, width], can leave a value as None if you don't wish to pad that dim"
        self.value = value

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> Any:
        height, width = data[self.fkey].data.size(2), data[self.fkey].data.size(3)
        dh = (self.size[0] - height) // 2 if self.size[0] is not None else 0
        dw = (self.size[1] - width) // 2 if self.size[1] is not None else 0
        
        # Pad frames
        data[self.fkey].data = tv_f.pad(data[self.fkey].data, padding=[dw, dh], fill=self.value, padding_mode="constant")

        # Offset detections
        data[self.dkey].data = data[self.dkey].offset(data[self.dkey].data, [dw, dh, 0]) # never pad in z-axis


class SquarePadFrames(DatasetOperation):
    """ Pad frames along the minimum axis such that the resultant height == width

    """
    def __init__(self, fkey: str, dkey: str, value: Optional[int] = 0, *args, **kwargs) -> None:
        super(SquarePadFrames, self).__init__(*args, **kwargs)
        self.fkey = fkey
        self.dkey = dkey
        self.value = value

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> Any:
        height, width = data[self.fkey].data.size(2), data[self.fkey].data.size(3)
        max_side = max(height, width)
        dh = (max_side - height) // 2
        dw = (max_side - width) // 2
        
        # Pad frames
        data[self.fkey].data = tv_f.pad(img=data[self.fkey].data, padding=[dw, dh], fill=self.value, padding_mode="constant")

        # Offset detections
        data[self.dkey].data = data[self.dkey].offset(data[self.dkey].data, [dw, dh, .0]) # never pad in z-axis


# class ResizeFrames(DatasetOperation):
#     """ Resize frames to the specified height and width

    # THIS ASSUMES ALL FRAMES ARE THE SAME SIZE WHICH MAY NOT BE THE CASE IWHT DYNAMIC DETECTION METHODS

#     Args:
#         DataOperation (_type_): _description_
#     """
#     def __init__(self, fkey: str, dkey: str, size: List[int], mode: Optional[str] = "BILINEAR", *args, **kwargs) -> None:
#         super(ResizeFrames, self).__init__(*args, **kwargs)
#         self.fkey = fkey
#         self.dkey = dkey
#         self.size = [size, size] if isinstance(size, int) else size
#         assert len(self.size) == 2, f"Must provide [height, width]"
#         self.mode = getattr(torchvision.transforms.InterpolationMode, mode)

#     def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
#         # Calculate scale
#         hscale = self.size[0] / data[self.fkey].data.size(2)
#         wscale = self.size[1] / data[self.fkey].data.size(3)

#         # Resize image
#         data[self.fkey].data = tv_f.resize(img=data[self.fkey].data, size=(self.size[0], self.size[1]), interpolation=self.mode, antialias=None)
        
#         # Scale detections
#         data[self.dkey].data = data[self.dkey].scale(data[self.dkey].data, [wscale, hscale, wscale]) # always maintain same z-axis scale


class SquareResizeFrames(DatasetOperation):
    """ Resize frames to be the same size in each axis

    Args:
        ResizeFrames (_type_): _description_
    """
    def __init__(self, fkey: str, dkey: str, size: List[int], mode: Optional[str] = "BILINEAR", *args, **kwargs) -> None:
        super(SquareResizeFrames, self).__init__(*args, **kwargs)
        self.fkey = fkey
        self.dkey = dkey
        self.size = size
        self.mode = getattr(torchvision.transforms.InterpolationMode, mode)

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        # Compute resize factor
        height, width = data[self.fkey].data.size(2), data[self.fkey].data.size(3)
        hscale = self.size - height
        wscale = self.size - width

        # Resize image
        data[self.fkey].data = tv_f.resize(img=data[self.fkey].data, size=(self.size, self.size), interpolation=self.mode, antialias=None)
        
        # Scale detections
        data[self.dkey].data = data[self.dkey].scale(data[self.dkey].data, [wscale, hscale, wscale]) # always maintain same z-axis scale


class MaskOnBox(DatasetOperation):
    """ Mask the image onto the provided detection

    Args:
        DataOperation (_type_): _description_
    """
    def __init__(self, fkey: str, dkey: str, *args, **kwargs) -> None:
        super(MaskOnBox, self).__init__(*args, **kwargs)
        self.fkey = fkey
        self.dkey = dkey

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        minimums = data[self.dkey].min(data[self.dkey].data).numpy()
        maximums = data[self.dkey].max(data[self.dkey].data).numpy()

        for idx in range(data[self.fkey].data.size(0)): # N, C, H, W for each frame
            x_min, y_min, z_min = tuple([round(v) for v in minimums[idx].tolist()])
            x_max, y_max, z_max = tuple([round(v) for v in maximums[idx].tolist()])
            data[self.fkey].data[idx,:,:y_min,:] = .0 # < y_min
            data[self.fkey].data[idx,:,y_max:,:] = .0 # > y_min
            data[self.fkey].data[idx,:,:,:x_min] = .0 # < x_min
            data[self.fkey].data[idx,:,:,x_max:] = .0 # > x_max


class ComputeAndApplyMasks(DatasetOperation):
    """ Mask the image using the specified mask (detections)

    Args:
        DatasetOperation (_type_): _description_
    """
    def __init__(self, frames: str, detections: str, *args, **kwargs) -> None:
        super(ComputeAndApplyMasks, self).__init__(*args, **kwargs)
        self.fkey = frames
        self.dkey = detections

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        # retrieve data
        frames = data[self.fkey].data
        landmarks = data[self.dkey].data

        # compute masks from landmarks
        masks = data[self.dkey].maskConvexHull(frames, landmarks)

        # apply mask across channels
        for cdx in range(frames.shape[1]):
            frames[:,cdx,:,:][masks] = .0

        data[self.fkey].data = frames


from src.datamodules.datasources.files.detections import BoundingBoxFormats

class ConvertToBoundingBoxes(DatasetOperation):
    def __init__(self, dkey: str, *args, **kwargs) -> None:
        super(ConvertToBoundingBoxes, self).__init__(*args, **kwargs)
        self.dkey = dkey

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        # retrieve detections
        detections = data[self.dkey]

        # boxes
        boxes = detections.convert_to_bounding_boxes(detections.data)

        # re-assign
        data[self.dkey] = BoundingBoxesModel(data=boxes, format=BoundingBoxFormats.XYXY, attrs=detections.attrs)


class ScaleBoundingBoxes(DatasetOperation):
    def __init__(self, fkey: str, dkey: str, scale: float, clip: Optional[bool] = True, *args, **kwargs) -> None:
        super(ScaleBoundingBoxes, self).__init__(*args, **kwargs)
        self.fkey = fkey
        self.dkey = dkey
        self.scale = scale
        self.clip = clip

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        # retrieve data
        frames = data[self.fkey].data # T, C, H, W
        boxes = data[self.dkey].data

        # apply scaling to detections
        boxes = data[self.dkey].scale_sides(boxes, self.scale)

        # clip boxes to frames
        if self.clip:
            height, width = frames.shape[-2], frames.shape[-1]
            boxes = data[self.dkey].clip(boxes, [0, 0, width, height]) # x1 y1 x2 y2

        # re-assign
        data[self.dkey].data = boxes


class SquareBoundingBoxes(DatasetOperation):
    def __init__(self, dkey: str, clip: Optional[bool] = True, *args, **kwargs) -> None:
        super(SquareBoundingBoxes, self).__init__(*args, **kwargs)
        self.dkey = dkey
        self.clip = clip

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        boxes = data[self.dkey].data

        w = data[self.dkey].width(boxes)
        h = data[self.dkey].height(boxes)

        if w - h > 0: # w is longest
            dh = w - h
            boxes[:,3] -= dh # y1
            boxes[:,5] += dh # y2

        elif w - h < 0: # h is longest
            dw = h - w
            boxes[:,2] -= dw # x1
            boxes[:,4] += dw # x2

        else:
            ...

        data[self.dkey].data = boxes



class CropOnBoundingBoxes(DatasetOperation):
    def __init__(self, fkey: str, dkey: str, frame_index: Optional[int] = None, *args, **kwargs) -> None:
        super(CropOnBoundingBoxes, self).__init__(*args, **kwargs)
        self.fkey = fkey
        self.dkey = dkey
        self.frame = frame_index

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        # retrieve data
        frames = data[self.fkey].data
        boxes = data[self.dkey].data 

        # iterate over pairs of frames/boxes
        new_frames = []
        for idx in range(frames.shape[0]):
            # dynamic/static cropping
            didx = idx if self.frame == None else self.frame # dynamic cropping

            # coordinates for cropping based on bounding box
            x_min, y_min, x_max, y_max = tuple([round(v) for v in boxes[didx,2:6].numpy()])
            top, left = y_min, x_min
            height, width = y_max - y_min, x_max - x_min

            # cropping with input padding
            frame = tv_f.crop(data[self.fkey].data[idx], top, left, height, width)
            new_frames.append(frame)

            # offset detections
            data[self.dkey].data = data[self.dkey].offset(data[self.dkey].data, [-left, -top, 0], idx=idx)

        # concat new frames
        data[self.fkey].data = torch.stack(new_frames, dim=0)


class ResizeFrames(DatasetOperation):
    def __init__(self, fkey: str, dkey: str, size: List[int], *args, **kwargs) -> None:
        super(ResizeFrames, self).__init__(*args, **kwargs)
        self.fkey = fkey
        self.dkey = dkey
        self.size = [size, size] if isinstance(size, int) else size

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        # retrieve data
        frames = data[self.fkey].data # T, C, H, W

        #
        new_frames = []
        for idx in range(frames.shape[0]):
            # calculate scale factor
            h_scale = self.size[0] / frames.shape[2]
            w_scale = self.size[1] / frames.shape[3]

            # resize image
            frame = tv_f.resize(img=frames[idx], size=self.size, antialias=None)
            new_frames.append(frame)

            # scale detections
            data[self.dkey].data = data[self.dkey].scale(data[self.dkey].data, [w_scale, h_scale, w_scale], idx=idx)

        # concat new frames
        data[self.fkey].data = torch.stack(new_frames, dim=0)


class CropOnBoxResize(DatasetOperation):
    """

    Process : iterate over frames...
        1. Crop image and detection onto bounding box defined by the detection.
        2. Resize image and detection to fixed size.

    Args:
        DatasetOperation (_type_): _description_
    """
    def __init__(self, fkey: str, dkey: str, size: int, frame_index: Optional[int] = None, pad_square: Optional[bool] = False, *args, **kwargs) -> None:
        super(CropOnBoxResize, self).__init__(*args, **kwargs)
        self.fkey = fkey
        self.dkey = dkey
        self.size = size
        self.frame_index = frame_index
        self.pad_square = pad_square

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        # retrieve data
        frames = data[self.fkey].data
        detections = data[self.dkey].data
        boxes = data[self.dkey].convert_to_bounding_boxes(detections)

        # static cropping
        if self.frame_index != None:
            x_min, y_min, x_max, y_max = tuple([round(v) for v in boxes[self.frame_index,2:6].numpy()])
            top, left = y_min, x_min
            height, width = y_max - y_min, x_max - x_min

        new_frames = []
        for idx in range(frames.shape[0]): # N, C, H, W for each frame
            # --- CROP ON BOX ---
            # dynamic cropping
            if self.frame_index == None:
                # coordinates for cropping based on bbox
                x_min, y_min, x_max, y_max = tuple([round(v) for v in boxes[idx,2:6].numpy()])
                top, left = y_min, x_min
                height, width = y_max - y_min, x_max - x_min
            
            # --- PAD BOX TO SQUARE TO RETAIN ASPECT RATIO ---
            if self.pad_square:
                cx = left + width / 2
                cy = top + height / 2 
                max_side_len = max(height, width)

                # new centered square box
                height = int(round(max_side_len, 0))
                width = int(round(max_side_len, 0))
                left = int(round(cx - max_side_len / 2))
                top = int(round(cy - max_side_len / 2))

            # crop current frame onto bbox
            frame = tv_f.crop(data[self.fkey].data[idx], top, left, height, width)

            # offset current detection
            data[self.dkey].data = data[self.dkey].offset(data[self.dkey].data, [-left, -top, 0], idx)

            # --- RESIZE IMAGE TO SIZE ---
            # calculate scale
            h_scale = self.size / frame.size(1)
            w_scale = self.size / frame.size(2)

            # resize image
            frame = tv_f.resize(img=frame, size=(self.size, self.size), antialias=None)
            new_frames.append(frame)
            
            # scale current detection
            boxes = data[self.dkey].scale(data[self.dkey].data, [w_scale, h_scale], idx)
            data[self.dkey].data = torch.round(boxes).to(int)

        data[self.fkey].data = torch.stack(new_frames, dim=0)


class CropOnBoxPad(DatasetOperation):
    def __init__(self, frames: str, detections: str, size: int, *args, **kwargs) -> None:
        super(CropOnBoxPad, self).__init__(*args, **kwargs)
        self.fkey = frames
        self.dkey = detections
        self.size = size

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        minimums = data[self.dkey].min(data[self.dkey].data).numpy()
        maximums = data[self.dkey].max(data[self.dkey].data).numpy()

        frames = []
        for idx in range(data[self.fkey].data.size(0)): # N, C, H, W for each frame
            # --- CROP ON BOX ---
            # Crop on box
            x_min, y_min, z_min = tuple([round(v) for v in minimums[idx].tolist()])
            x_max, y_max, z_max = tuple([round(v) for v in maximums[idx].tolist()])
            top, left = y_min, x_min
            height, width = y_max - y_min, x_max - x_min
            
            frame = tv_f.crop(data[self.fkey].data[idx], top, left, height, width)

            # Offset detections
            data[self.dkey].data = data[self.dkey].offset(data[self.dkey].data, [-left, -top, 0], idx=idx)

            # --- PAD BOX TO SQUARE ---
            height, width = frame.size(1), frame.size(2)
            max_side = max(height, width)
            dh = (max_side - height) // 2
            dw = (max_side - width) // 2
            
            # Pad frames
            frame = tv_f.pad(img=frame, padding=[dw, dh], fill=0, padding_mode="constant")

            # Offset detections
            data[self.dkey].data = data[self.dkey].offset(data[self.dkey].data, [dw, dh, 0], idx=idx)

            # --- RESIZE IMAGE TO SIZE ---
            # Calculate scale
            hscale = self.size / frame.size(1)
            wscale = self.size / frame.size(2)

            # Resize image
            frame = tv_f.resize(img=frame, size=(self.size, self.size), antialias=None)
            
            # Scale detections
            data[self.dkey].data = data[self.dkey].scale(data[self.dkey].data, [wscale, hscale, wscale], idx=idx)

            frames.append(frame)

        data[self.fkey].data = torch.stack(frames, dim=0)



from src.datamodules.datamodels.timeseries import TimeseriesModel

class RGBAverage(DatasetOperation):
    """ Compute the average of each `RGB` value per frame resulting in RGB traces.

    Args:
        DatasetOperation (_type_): _description_
    """
    def __init__(self, frames: str, traces: str, retain: Optional[str] = None, exclude_zeros: Optional[bool] = False, *args, **kwargs) -> None:
        super(RGBAverage, self).__init__(*args, **kwargs)
        self.key = frames
        self.new_key = traces
        self.retain = retain
        self.exclude_zeros = exclude_zeros

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        """ Compute
        
        Conditionally shunt key.data into new key

        frames [N, C, H, W] -> mean(H,W) -> traces [N, C]
        """
        # retrieve frames
        frames = data[self.key].data

        # Compute the rgb trace per batch
        rgb_trace = []
        for frame in frames:
            if self.exclude_zeros:
                mask = torch.sum(frame, dim=0) > .0 + 1e-5
                rgb_avg = torch.mean(frame[:,mask], dim=1)
            else:
                rgb_avg = torch.mean(frame, dim=(1,2))
            rgb_trace.append(rgb_avg)

        # Combine compute RGB trace values
        rgb_trace = torch.stack(rgb_trace, dim=0)

        # Create new `DataModel` for the new type of data
        data[self.new_key] = TimeseriesModel(
            data = rgb_trace,
            attrs = {
                "sps": data[self.key].sps
            }
        )

        # Conditionally delete the old entry
        if self.retain is not None:
            data[self.retain] = copy.deepcopy(data[self.key])
        del data[self.key]


class ConcatenateFrames(DatasetOperation):
    def __init__(self, frames: List[str], output: str, dim: int, *args, **kwargs) -> None:
        super(ConcatenateFrames, self).__init__(*args, **kwargs)
        self.frames = frames
        self.output = output
        self.dim = dim

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        """_summary_

        Args:
            data (Dict[str, Any]): _description_
        """
        frames = [data[key].data for key in self.frames]
        frames = torch.cat(frames, dim=self.dim) # concat along channel dim

        data[self.output].data = frames


from scipy.interpolate import griddata
from skimage.transform import PiecewiseAffineTransform, warp
from src.datamodules.datapipes.extract.landmarks import MEDIAPIPE_FACES, MEDIAPIPE_UV_COORDINATES


def cosine_angle(vec1: ndarray, vec2: ndarray):
    """ Compute the angle in radians between two vectors using the cosine rule.

    Args:
        vec1 (_type_): Vector in R^n
        vec2 (_type_): Vector in R^n

    Returns:
        _type_: Angle between vectors in radians.
    """
    return np.arccos(np.dot(vec1,vec2) / (np.sqrt(np.dot(vec1,vec1)) * np.sqrt(np.dot(vec2,vec2)))) / np.pi


def normalized_dot_product(vec1: ndarray, vec2: ndarray) -> float:
    """ Returns the normalized dot product

    Args:
        vec1 (ndarray): _description_
        vec2 (ndarray): _description_

    Returns:
        float: [-1, 1]
    """
    return np.dot(vec1, vec2) / (np.sqrt(np.dot(vec1,vec1)) * np.sqrt(np.dot(vec2,vec2)))


from src.datamodules.datamodels.frames import VideoFramesModel

class ComputeAngleMap(DatasetOperation):
    """ 
    """
    def __init__(self, fkey: str, dkey: str, okey: str, fill: Optional[float] = -1.0, *args, **kwargs) -> None:
        super(ComputeAngleMap, self).__init__(*args, **kwargs)
        # Frame & detection keys
        self.fkey = fkey
        self.dkey = dkey
        self.okey = okey

        # Camera-plane normal vector
        '''
        Camera-plane outwards direction is [0,0,-1], however we want to know orientation 
        w.r.t pointing into the camera hence [0,0,1].
        '''
        self.camera_normal = np.array([0, 0, -1]).T
    
        # Spatial interpolation
        self.fill = fill

    def apply(self, data: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        # Extract frames & detections
        frames = data[self.fkey]
        landmarks = data[self.dkey]

        # Outputs
        angles = []

        # Iterate over frames
        for frame, frame_landmarks in zip(frames.data, landmarks.data):
            # Construct mesh and compute vertex normals
            mesh = landmarks.constructMesh(frame_landmarks, MEDIAPIPE_FACES)
            normals = mesh.vertex_normals

            # Compute relative orientation: surface wrt camera normal    
            vertex_dp = np.array([normalized_dot_product(normal, self.camera_normal) for normal in normals])

            # Compute mask for landmarks/normals based on normal orientation
            vertex_mask = vertex_dp > .0

            # Define grid for spatial interpolation in XY space
            height, width = frame.size(1), frame.size(2)
            grid_x, grid_y = np.meshgrid(np.array(range(width)), np.array(range(height)))

            # Perform spatial interpolation in XY space
            '''
            Into-camera cos(0)=1 & Out-of-camera cos(180) = -1 hence we want, fill -1.0
            '''
            # print(frame_landmarks.shape, vertex_dp.shape, frame.shape, grid_x.shape, grid_y.shape)
            frame_angles = griddata(
                points = (frame_landmarks[:468,1][vertex_mask], frame_landmarks[:468,2][vertex_mask]), 
                values = vertex_dp[vertex_mask], 
                xi = (grid_x, grid_y), 
                method = "linear", 
                fill_value = self.fill
            )
            angles.append(frame_angles)

        # Covnert to tensors and store
        angles = torch.from_numpy(np.array(angles)).unsqueeze(-1).permute(0,3,1,2).to(dtype=torch.float32)
        
        # Create new `DataModel` entry.
        data[self.okey] = VideoFramesModel(
            data = angles,
            format = frames.format,
            attrs = frames.attrs
        )


from src.datamodules.datamodels.landmarks import LandmarksModel

class ConvertFramesXY2UV(DatasetOperation):
    def __init__(self, fkey: str, dkey: str, okey: str, size: int, warp_kwargs: Optional[Dict[str,Any]] = {}, *args, **kwargs) -> None:
        super(ConvertFramesXY2UV, self).__init__(*args, **kwargs)
        # Frame & detection keys
        self.fkey = fkey
        self.dkey = dkey
        self.okey = okey

        # Relative surface angle function
        self.camera_normal = np.array([0, 0, -1]).T

        # UV texture mapping
        self.size = int(size)
        self.keypoints_uv = self.size * MEDIAPIPE_UV_COORDINATES

        # Affine transformation
        self.transform = PiecewiseAffineTransform() 
        self.warp_kwargs = warp_kwargs

    def apply(self, data: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        # Extract frames & detections
        frames = data[self.fkey]
        landmarks = data[self.dkey]

        # Output results
        frames_uv = []

        # Iterate over frames
        for frame, frame_landmarks in zip(frames.data, landmarks.data):
            # Estimate affine transformation from XY to UV coordinate-space
            self.transform.estimate(self.keypoints_uv, frame_landmarks[:468,1:3])

            # Transform image from XY to UV space
            uv_frame = warp(
                image = frame.permute(1,2,0).numpy(),
                inverse_map = self.transform,
                output_shape = (self.size, self.size),
                **self.warp_kwargs
            )
            frames_uv.append(uv_frame)

        # Covnert to tensors and store
        frames_uv = torch.from_numpy(np.array(frames_uv)).permute(0,3,1,2).to(dtype=torch.float32)
        data[self.okey] = VideoFramesModel(data=frames_uv, format=frames.format, attrs=frames.attrs)

        # Store updated landmarks
        # keypoints_uv = np.zeros((landmarks.data.shape[0], 468, 4))
        keypoints_uv = landmarks.data # [N, 478, 4] (retain frame idxs)
        keypoints_uv = keypoints_uv[:,:468,:] # remove un-used landmars
        keypoints_uv[:,:,1:3] = torch.from_numpy(self.keypoints_uv) # override with UV landmarks
        # keypoints_uv = torch.from_numpy(keypoints_uv)
        data[f"{self.dkey}_uv"] = LandmarksModel(data=keypoints_uv, format="FUV", attrs=landmarks.attrs)


class ComputeFramesAndAnglesUV(DatasetOperation):
    """
    """
    def __init__(self, 
        fkey: str, 
        dkey: str, 
        frames_okey: str, 
        angles_okey: str, 
        size: int, 
        fill: Optional[float] = -1.0, 
        warp_kwargs: Optional[Dict[str, Any]] = {}, 
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        # Frame & detection keys
        self.fkey = fkey
        self.dkey = dkey
        self.frames_okey = frames_okey
        self.angles_okey = angles_okey

        # Relative surface angle function
        self.camera_normal = np.array([0, 0, -1]).T
        self.fill = fill

        # UV texture mapping    
        self.size = int(size)
        self.grid = np.array(range(self.size))
        self.keypoints_uv = self.size * MEDIAPIPE_UV_COORDINATES

        # Define grid
        self.grid_u, self.grid_v = np.meshgrid(self.grid, self.grid)

        # Affine transformation
        self.transform = PiecewiseAffineTransform()
        self.warp_kwargs = warp_kwargs
        

    def apply(self, data: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        # Extract frames & detections
        frames : VideoFramesModel = data[self.fkey]
        landmarks : LandmarksModel = data[self.dkey]
        assert frames.format == "opencv", f"Require frame format to be in openCV [THWC] [BGR]"

        # Output results
        frames_uv, angles_uv = [], []

        # Iterate over frames
        for frame, frame_landmarks in tqdm(zip(frames.data, landmarks.data), total=frames.data.shape[0]):
            # Construct mesh and vertex normals
            mesh = landmarks.constructMesh(frame_landmarks, MEDIAPIPE_FACES)
            normals = mesh.vertex_normals

            # Compute surface orientation wrt camera       
            vertex_dp = np.array([normalized_dot_product(normal, self.camera_normal) for normal in normals])
            vertex_dp = (vertex_dp + 1) / 2 # scale from [-1,1] to [0,1]

            # Define affine transformation from XY to UV coordinate-space
            self.transform.estimate(self.keypoints_uv, frame_landmarks[:468,1:3])
            landmarks_uv = self.transform.inverse(frame_landmarks[:468,1:3])

            # Transform image from XY to UV space
            frame_uv = warp(
                image = frame, # torchvision [CHW] to opencv [HWC]
                inverse_map = self.transform,
                output_shape = (self.size, self.size),
                **self.warp_kwargs
            )
            frame_uv = (255 * frame_uv).astype(np.uint8) # consider loss
            frames_uv.append(frame_uv)

            # Perform spatial interpolation in UV space
            angle_uv = griddata(
                points = (landmarks_uv[:,0], landmarks_uv[:,1]), 
                values = vertex_dp, 
                xi = (self.grid_u, self.grid_v), 
                method = "linear", 
                fill_value = self.fill
            )
            angle_uv = np.expand_dims(angle_uv, axis=-1) # expand channels
            angle_uv = (255 * angle_uv).astype(np.uint8) # consider loss
            angles_uv.append(angle_uv)

        # Covnert to tensors and store results
        frames_uv = np.array(frames_uv)
        angles_uv = np.array(angles_uv)
        
        # Store new `DataModel`
        data[self.frames_okey] = VideoFramesModel(data=frames_uv, format=frames.format, attrs=frames.attrs)
        data[self.angles_okey] = VideoFramesModel(data=angles_uv, format=frames.format, attrs=frames.attrs)

        # Convert landmarks to UV coordinates: UV is fixed across all frames
        keypoints_uv = landmarks.data.astype(np.float32) # [N, 478, 4] (retain frame idxs)
        keypoints_uv = keypoints_uv[:,:468,:] # remove un-used landmars
        keypoints_uv[:,:,1:3] = torch.from_numpy(self.keypoints_uv) # override with UV landmarks : but not frame indexes
        data[f"{self.dkey}_uv"] = LandmarksModel(data=keypoints_uv, format="FUV", attrs=landmarks.attrs)


class ThresholdFramesFromAngles(DatasetOperation):
    def __init__(self, mkey: str, fkey: str, angle: float, fill: Optional[float] = .0, *args, **kwargs) -> None:
        super(ThresholdFramesFromAngles, self).__init__(*args, **kwargs)
        self.mkey = mkey
        self.fkey = fkey
        self.threshold = np.cos(angle)
        self.fill = fill

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        # Generate mask : [T,C,H,W]
        mask = data[self.mkey].data[:,0] < self.threshold
        stacked_mask = torch.stack([mask] * data[self.fkey].data.shape[1], dim=1)

        # Apply mask : 
        data[self.fkey].data[stacked_mask] = self.fill


class ThresholdFramesLTE(DatasetOperation):
    def __init__(self, mkey: str, fkey: str, threshold: float, fill: Optional[float] = .0, *args, **kwargs) -> None:
        super(ThresholdFramesLTE, self).__init__(*args, **kwargs)
        self.mkey = mkey
        self.fkey = fkey
        self.threshold = threshold # angle in radians
        self.fill = fill

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        """_summary_

        Args:
            data (Dict[str, Any]): _description_
        """
        # Generate mask
        mask = data[self.mkey].data < self.threshold

        # Apply mask
        data[self.fkey].data[:,mask] = self.fill


class ThresholdFramesGTE(DatasetOperation):
    def __init__(self, frames: str, threshold: float, fill: Optional[float] = .0, *args, **kwargs) -> None:
        super(ThresholdFramesGTE, self).__init__(*args, **kwargs)
        self.fkey = frames
        self.threshold = threshold
        self.fill = fill

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        data[self.fkey].data[data[self.fkey].data >= self.threshold] = self.fill


class ThresholdFramesByKeyLTE(DatasetOperation):
    def __init__(self, fkey: str, mkey: str, threshold: Optional[float] = 0.0, fill: Optional[float] = 0.0, *args, **kwargs) -> None:
        super(ThresholdFramesByKeyLTE, self).__init__(*args, **kwargs)
        self.fkey = fkey
        self.mkey = mkey
        self.threshold = threshold
        self.fill = fill

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        frames = data[self.fkey].data
        masks = data[self.mkey].data <= self.threshold
        for idx in range(frames.size(0)):
            frames[idx][:,masks[idx,0].squeeze(0)] = self.fill # [3-channel]
        data[self.fkey].data = frames

