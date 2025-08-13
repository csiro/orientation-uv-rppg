from orientation_uv_rppg.utils.landmark_detector import MediaPipeLandmarkDetector
from orientation_uv_rppg.utils.frame_warp import GPUPiecewiseAffineTransform
from orientation_uv_rppg.utils.frame_mask import UVAngleMasker

from torch import Tensor


class OrientationMaskedTextureSpaceVideoProcessor:
    def __init__(self,
        min_detection_confidence: float = 0.45,
        min_tracking_confidence: float = 0.45,
        device: str = "cuda",
        output_size: int = 64,
        degree_threshold: float = 60.0,
    ) -> None:
        self.detector = MediaPipeLandmarkDetector(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.warper = GPUPiecewiseAffineTransform(
            keypoints_uv=self.detector.landmarks_uv,
            device=device
        )
        self.masker = UVAngleMasker(
            faces=self.detector.mesh_faces,
            camera_normal=self.detector.camera_normal
        )
        self.output_size = output_size
        self.degree_threshold = degree_threshold
        

    def __call__(self, frames_xy: Tensor) -> Tensor:
        """_summary_

        Args:
            frames (Tensor): Video frames of shape [N,H,W,C]

        Returns:
            Tensor: Frames of shape [N,output_size,output_size,C]
        """
        landmarks_xyz = self.detector(frames_xy) # [N,V,3]
        frames_uv = self.warper(frames_xy, landmarks_xyz, output_size=self.output_size) # [N,output_size,output_size,3]
        frames_uv_masked = self.masker(frames_uv, self.detector.landmarks_uv, landmarks_xyz, degree_threshold=self.degree_threshold)
        return frames_uv_masked