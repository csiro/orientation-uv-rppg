
import cv2
import torch
import logging
import numpy as np
from scipy.spatial import ConvexHull
from src.datamodules.datamodels import DetectionModel
from src.datamodules.datamodels.mesh import Mesh

from typing import *
from numpy import ndarray
from torch import Tensor

log = logging.getLogger(__name__)


class LandmarksModel(DetectionModel):
    """ Class for interfacing with `Landmarks`

    NOTE: All `staticmethod`s expect the data to be in the default format described below, as is returned from
    the `retrieve` property.
    
    landmarks (dataset)
        <val> : [N, 478, 4] in format
            [
                N (num detections), 
                478 (landmarks), 
                [frame_idx, x, y, z]
            ]
        .<attrs>
            detector: Landmark detection algorithm pipeline
            format: "xyz"
            MEDIA_PIPE ::
            contours: frozenset([(landmark_idx, landmark_jdx), ...]) landmark_idx connects to landmark_jdx
            tesselation: frozenset([(landmark_idx, landmark_jdx), ...]) landmark_idx connects to landmark_jdx (tesselation triangles)
    
    """
    # Keys
    format_f : str = "f" # frame index
    format_x : str = "x"
    format_y : str = "y"
    format_z : str = "z"

    def __init__(self, *args, **kwargs) -> None:
        super(LandmarksModel, self).__init__(*args, **kwargs)

    @property
    def algorithm(self) -> str:
        return self.attrs["algorithm"] if "algorithm" in self.attrs else None

    @property
    def f_idx(self) -> None:
        self.format.lower().find(self.format_f.lower())

    @property
    def x_idx(self) -> None:
        return self.format.lower().find(self.format_x.lower())
    
    @property
    def y_idx(self) -> None:
        return self.format.lower().find(self.format_y.lower())
    
    @property
    def z_idx(self) -> None:
        return self.format.lower().find(self.format_z.lower())
    
    @property
    def max_length(self) -> None:
        return self.data.shape[0]
    
    @property
    def length(self) -> None:
        start = self.start if self.start is not None else 0
        stop = self.stop if self.stop is not None else self.max_length
        return stop - start
    
    # def prepare(self, data: ndarray) -> ndarray:
    #     """_summary_

    #     Args:
    #         data (ndarray): _description_

    #     Returns:
    #         ndarray: _description_
    #     """
    #     # Convert to `Tensor` in contiguous format
    #     data = torch.from_numpy(data).contiguous()

    #     return data

    @staticmethod
    def frame_indexes(data: ndarray) -> ndarray:
        return data[:,0,0].astype(int)

    @staticmethod
    def min(data: Tensor) -> Tensor:
        """ Calculate the minimum (x,y,z) landmark value for each frame.

        Args:
            data (Tensor): _description_

        Returns:
            Tensor: [[x_min, y_min, z_min], ...] for each frame of shape [N,3]
        """
        return torch.min(data[:,:,1:], dim=1).values

    @staticmethod
    def max(data: Tensor) -> Tensor:
        """ Calculate the maximum (x,y,z) landmark values for each frame.

        Args:
            data (Tensor): _description_

        Returns:
            Tensor: [[x_max, y_max, z_max], ...] for each frame of shape [N,3]
        """
        return torch.max(data[:,:,1:], dim=1).values

    @staticmethod
    def offset(data: Tensor, offset: List[float], idx: Optional[int] = None) -> Tensor:
        """ Offset all of the landmarks by the specific `offset` along the [x, y, z] axis
        respectively.

        Args:
            data (Tensor): _description_
            offset (List[float]): _description_
            idx (Optional[int], optional): _description_. Defaults to None.

        Returns:
            Tensor: _description_
        """
        if idx is None: # Offset all frames.
            data[:,:,1] += offset[0]
            data[:,:,2] += offset[1]
            data[:,:,3] += offset[2]
        else: # Compute for a given frame.
            data[idx,:,1] += offset[0]
            data[idx,:,2] += offset[1]
            data[idx,:,3] += offset[2]
        return data
    
    @staticmethod
    def scale(data: Tensor, scale: List[float], idx: Optional[int] = None) -> Tensor:
        """_summary_

        Args:
            data (Tensor): _description_
            scale (List[float]): _description_
            idx (Optional[int], optional): _description_. Defaults to None.

        Returns:
            Tensor: _description_
        """
        if idx is None: # Scale all frames.
            data[:,:,1] = scale[0] * data[:,:,1]
            data[:,:,2] = scale[1] * data[:,:,2]
            data[:,:,3] = scale[2] * data[:,:,3]
        else: # Scale a given frame
            data[idx,:,1] = scale[0] * data[idx,:,1]
            data[idx,:,2] = scale[1] * data[idx,:,2]
            data[idx,:,3] = scale[2] * data[idx,:,3]
        return data
    
    @staticmethod
    def clip(data: Tensor, size: List[float], idx: Optional[int] = None) -> Tensor:
        """ Clip to the size provided 

        Args:
            data (Tensor): _description_
            size (List[float]): [x_min, x_max, y_min, y_max, z_min, z_max]
            idx (Optional[int], optional): _description_. Defaults to None.

        Returns:
            Tensor: _description_
        """
        if idx is None:
            data[:,:,data[:,:,1] < size[0]] = size[0] # < x_min
            data[:,:,data[:,:,1] > size[1]] = size[1] # > x_max
            data[:,:,data[:,:,2] < size[2]] = size[2] # < y_min
            data[:,:,data[:,:,2] > size[3]] = size[3] # > y_max
            data[:,:,data[:,:,3] < size[4]] = size[4] # < z_min
            data[:,:,data[:,:,3] > size[5]] = size[5] # > z_max
        else:
            data[idx,:,data[idx,:,1] < size[0]] = size[0] # < x_min
            data[idx,:,data[idx,:,1] > size[1]] = size[1] # > x_max
            data[idx,:,data[idx,:,2] < size[2]] = size[2] # < y_min
            data[idx,:,data[idx,:,2] > size[3]] = size[3] # > y_max
            data[idx,:,data[idx,:,3] < size[4]] = size[4] # < z_min
            data[idx,:,data[idx,:,3] > size[5]] = size[5] # > z_max
        return data
    

    @staticmethod
    def overlay(frames: ndarray, landmarks: Union[Tensor, ndarray], *args, **kwargs) -> ndarray:
        """_summary_

        Args:
            frames (ndarray): Format [T,H,W,C]
            landmarks (Union[Tensor, ndarray]): _description_

        Returns:
            ndarray: _description_
        """
        for idx, _ in enumerate(frames):
            for landmark in landmarks[idx,:,1:]:
                if type(landmark) == Tensor: landmark = landmark.numpy()
                x, y, z = (round(v) for v in landmark.tolist())
                cv2.circle(frames[idx], center=(x,y), radius=0, thickness=-1, *args, **kwargs)                    
        return frames


    @staticmethod
    def to_absolute(self, height: int, width: int) -> None:
        # NOTE: z-dim has approx same scale as width of image (typically)
        self.scale(width, height, width)

    @staticmethod
    def convert_to_bounding_boxes(data: Tensor) -> Tensor:
        """ Convert Landmarks in [N,478,[f,x,y,z]] to Bounding Boxes [N,[f,0,x1,y1,x2,y2]] format.

        Args:
            data (Tensor): _description_

        Returns:
            Tensor: _description_
        """
        boxes = []
        for idx in range(data.shape[0]):
            # extract values
            f = data[idx,0,0]
            x = data[idx,:,1]
            y = data[idx,:,2]

            # convert to bbox values
            x_min = torch.min(x, dim=0).values
            y_min = torch.min(y, dim=0).values
            x_max = torch.max(x, dim=0).values
            y_max = torch.max(y, dim=0).values

            # format bbox
            box = torch.stack([f, torch.zeros(1)[0], x_min, y_min, x_max, y_max])
            boxes.append(box)

        boxes = torch.stack(boxes)

        return boxes





    '''
    convexHull
    overlayHull
    maskHull 
    '''
    
    @staticmethod
    def maskConvexHull(frames: Tensor, landmarks: Tensor, outside: Optional[bool] = True) -> ndarray:
        """_summary_

        Args:
            frames (ndarray): Format [T,C,H,W]
            landmarks (Union[Tensor, ndarray]): Format [T,478,4]

        Returns:
            ndarray: Masks in [T,H,W] format
        """
        # sz
        sz = frames.size()
        T, H, W = sz[0], sz[-2], sz[-1]

        # define x, y
        y, x = np.mgrid[:H,:W]
        masks = torch.zeros((T,H,W)).to(dtype=torch.bool) if outside else torch.ones((T,H,W)).to(dtype=torch.bool)

        # for each frame
        for idx in range(frames.shape[0]):
            # compute convex hull
            hull = ConvexHull(landmarks[idx,:,1:-1]) # [x,y]

            # for each equation
            for eqn in hull.equations:
                A, B, b = tuple(eqn) # defined by [A = hyperplane coeff vals, b = offset] this only works for 2D

                # compute mask from hyperplane
                '''
                convex hull satisfies Ax <= -b ; we mask y <= so flip conditionally on gradient

                NOTE: http://www.qhull.org/html/qh-opto.htm#n
                '''
                if outside:
                    if B != 0: # if 0
                        masks[idx] |= y <= - (b + A * x) / B if B < 0 else y > - (b + A * x) / B # hyper
                    else:
                        log.warning(f"Hyperplane y-gradient B=0 for {idx}")
                        masks[idx] |= x <= -b / A if A < 0 else x > -b / A
                else:
                    if B != 0: # if 0
                        masks[idx] &= y > - (b + A * x) / B if B < 0 else y <= - (b + A * x) / B # hyper
                    else:
                        log.warning(f"Hyperplane y-gradient B=0 for {idx}")
                        masks[idx] &= x >= -b / A if A < 0 else x < -b / A

        return masks

    @staticmethod
    def constructMesh(landmarks: Tensor, simplexes: ndarray) -> Mesh:
        """_summary_

        Args:
            landmarks (Tensor): _description_
            simplexes (ndarray): _description_

        Returns:
            Mesh: _description_
        """
        # Error checking
        assert len(landmarks.shape) == 2, f"Require landmarks to be of shape [N,4], only pass 1-frame at a time."

        # Create mesh object 
        mesh = Mesh()
        mesh.update(landmarks[:468,1:], simplexes) # MediaPipe only provides tessellation for non-refined

        return mesh
