from typing import Optional
import torch
import random
import numpy as np
import albumentations
from src.datamodules.datapipes import DatasetOperation
from abc import abstractmethod

from typing import *
from torch import Tensor
from numpy import ndarray


class SampleAugmentation(DatasetOperation):
    """ Perform video augmentations

    Each video-clip will have an augmentation applied with probability p.

    Each frame in a video-clip will have the augmentation applied with probability alb.p
    which may be (alb. p=1.0) applied across all frames or (alb. p<1.0) applied across
    a subset of the frames.

    Each frame may recieve a fixed augmentation if the parameters of the initialized 
    augmentation do NOT vary.

    # TODO: Overhaul datapipe transforms to handle video and detections in a more coherent manner
    
    """
    def __init__(self, 
        augmentations: Dict[str, Callable],
        fkey: str, 
        dkey: Optional[str] = None, 
        p_video: Optional[float] = 0.50, 
        always_apply_video: Optional[bool] = False, 
        p_frame: Optional[float] = 0.50, 
        always_apply_frame: Optional[bool] = False, 
        per_video: Optional[bool] = True, 
        *args, **kwargs) -> None:
        """
        """
        super(SampleAugmentation, self).__init__(*args, **kwargs)

        # Loc. frames & detections
        self.fkey = fkey
        self.dkey = dkey

        # Video
        self.p_video = p_video
        self.always_apply_video = always_apply_video

        # Frames
        self.p_frame = p_frame
        self.always_apply_frame = always_apply_frame
        
        # Whether to apply with fixed params per video
        self.per_video = per_video

        # Augmentations
        self.augmentation = albumentations.Compose(
            list(augmentations.values()), 
            bbox_params = albumentations.BboxParams(format="pascal_voc")
        )

    def apply(self, data: Dict[str, Any], *args, **kwargs) -> None:
        if (self.always_apply_video) or (random.random() < self.p_video): # Augment video?

            # Fixed parameters across frames
            if self.per_video:
                aug_params = [t.get_params() for t in self.augmentation.transforms]

            new_frames = []
            new_detections = []

            # Iterate over frames and apply albumentations augmentations
            for idx in range(data[self.fkey].data.shape[0]):
                
                # Extract current index
                frame = data[self.fkey].data[idx]
                detection = data[self.dkey].data[idx]

                if (self.always_apply_frame) or (random.random() < self.p_frame): # Augment frame?

                    # Format into albumentations
                    frame = frame.permute(1,2,0).numpy() # H, W, C
                    bboxes = detection[2:].unsqueeze(0).numpy() # 1, 4
                    aug_data = {"image": frame, "bboxes": bboxes}

                    # Dynamic parameters per frame
                    if not self.per_video:
                        aug_params = [t.get_params() for t in self.augmentation.transforms]

                    # Pre-process
                    for p in self.augmentation.processors.values():
                        p.preprocess(aug_data)

                    for idx, augmentation in enumerate(self.augmentation.transforms):
                        # data = t(**data)

                        # # Target dependent parameters
                        # if augmentation.targets_as_params:
                        #     assert all(key in kwargs for key in augmentation.targets_as_params), "{} requires {}".format(
                        #         self.__class__.__name__, augmentation.targets_as_params
                        #     )
                        #     targets_as_params = {k: kwargs[k] for k in augmentation.targets_as_params}
                        #     params_dependent_on_targets = augmentation.get_params_dependent_on_targets(targets_as_params)
                        #     aug_params[idx].update(params_dependent_on_targets)

                        # Augment
                        aug_data : Dict[str, Any] = augmentation.apply_with_params(aug_params[idx], **aug_data)

                    # Post-process
                    for p in self.augmentation.processors.values():
                        p.postprocess(aug_data)

                    # Extract from response
                    frame : ndarray = aug_data["image"]
                    bboxes : ndarray = aug_data["bboxes"]

                    # Format into default format
                    frame = torch.from_numpy(frame).permute(2,0,1) # C, H, W
                    bboxes = torch.from_numpy(np.array(bboxes))
                    detection[2:] = bboxes

                # Accumulate
                new_frames.append(frame)
                new_detections.append(detection)
                
            # Store results
            data[self.fkey].data = torch.stack(new_frames, dim=0)
            if self.dkey is not None:
                data[self.dkey].data = torch.stack(new_detections, dim=0)


class BlurVideo(SampleAugmentation):
    def __init__(self, aug_args: Optional[Dict[str, Any]] = {}, *args, **kwargs) -> None:
        augmentations = {"blur": albumentations.Blur(**aug_args)}
        super(BlurVideo, self).__init__(augmentations, *args, **kwargs)
        

class HorizontalFlipVideo(SampleAugmentation):
    def __init__(self, aug_args: Optional[Dict[str, Any]] = {}, *args, **kwargs) -> None:
        augmentations = {"hflip": albumentations.HorizontalFlip(**aug_args)}
        super(HorizontalFlipVideo, self).__init__(augmentations, *args, **kwargs)
     

class VerticalFlipVideo(SampleAugmentation):
    def __init__(self, aug_args: Optional[Dict[str, Any]] = {}, *args, **kwargs) -> None:
        augmentations = {"vflip": albumentations.VerticalFlip(**aug_args)}
        super(VerticalFlipVideo, self).__init__(augmentations, *args, **kwargs)
     

class RotateVideo(SampleAugmentation):
    def __init__(self, aug_args: Optional[Dict[str, Any]] = {}, *args, **kwargs) -> None:
        augmentations = {"rotate": albumentations.Rotate(**aug_args)}
        super(RotateVideo, self).__init__(augmentations, *args, **kwargs)
