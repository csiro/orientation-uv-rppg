import numpy as np
import torch
from torch import Tensor
from scipy.interpolate import griddata
from typing import Optional, Tuple


class UVAngleMasker:
    """
    Mask a UV image by surface orientation relative to a camera normal.

    Overview
    --------
    For each frame:
      1) Compute per-vertex normals from (V,3) vertex positions and (F,3) faces.
      2) Take dot product with the camera normal -> cos(theta) in [-1, 1] per vertex.
      3) Interpolate these cosines from irregular UV points (V,2 in [0,1]) to an HxW grid.
      4) Zero out UV pixels whose orientation is more oblique than a degree threshold
         (i.e., cos(theta) < cos(threshold)).

    Inputs
    ------
    Static:
      faces: (F,3) np.ndarray[int] triangle indices.

    Per call:
      frames_uv:     (T,H,W,C) torch.Tensor. The UV-space frames to be masked.
      landmarks_uv:  (V,2) torch.Tensor or np.ndarray. UV for each vertex in [0,1].
      landmarks_xyz: (T,V,3) or (V,3) torch.Tensor/np.ndarray. Vertex positions per frame
                     (broadcasts if shape is (V,3)).
      degree_threshold: float. Obliquity cutoff in degrees. Pixels with cos(theta) below
                        cos(degree_threshold) are masked.

    Notes
    -----
    - Pixels outside the convex hull of UV samples receive `fill_value` before thresholding.
      If `fill_value` < cos(degree_threshold) those pixels will be masked.
    - Set `flip_v=True` if your UV V-axis is opposite to the raster grid’s row direction.
    """

    def __init__(
        self,
        faces: np.ndarray,
        camera_normal: Tuple[float, float, float] = (0.0, 0.0, -1.0),
        method: str = "linear",          # 'linear' or 'nearest' (scipy.interpolate.griddata)
        fill_value: float = -1.0,
        flip_v: bool = False,
        eps: float = 1e-12,
    ) -> None:
        self.faces = np.asarray(faces, dtype=np.int32)
        cam = np.asarray(camera_normal, dtype=np.float64)
        self.cam = cam / max(np.linalg.norm(cam), eps)
        self.method = method
        self.fill_value = float(fill_value)
        self.flip_v = bool(flip_v)
        self.eps = float(eps)

    # ------------------------- public API -------------------------

    def __call__(
        self,
        frames_uv: Tensor,            # (T,H,W,C)
        landmarks_uv,                 # (V,2) torch or np
        landmarks_xyz,                # (T,V,3) or (V,3) torch or np
        degree_threshold: Optional[float] = 90.0,
    ) -> Tensor:
        """
        Apply the orientation mask to frames_uv and return a masked copy.
        """
        frames_uv_masked = frames_uv.clone()
        UV = self._to_numpy_2d(landmarks_uv)                         # (V,2) np
        XYZ = self._to_numpy_xyz(landmarks_xyz)                      # (T,V,3) np
        T, H, W, C = frames_uv.shape
        cos_thr = float(np.cos(np.deg2rad(degree_threshold)))

        for t in range(T):
            cos_grid = self._cosine_map(H, W, UV, XYZ[t])            # (H,W) np
            mask_np = np.isfinite(cos_grid) & (cos_grid < cos_thr)   # True => zero-out
            mask_t = torch.from_numpy(mask_np).to(device=frames_uv.device, dtype=torch.bool)

            # Expand mask to (H,W,C) then zero selected pixels
            mask3 = mask_t.unsqueeze(-1).expand(H, W, C)
            frames_uv_masked[t] = torch.where(mask3, torch.zeros_like(frames_uv_masked[t]), frames_uv_masked[t])

        return frames_uv_masked

    # Backward-compatible alias: returns cos(theta) grid like before.
    def angle_map(self, frame: Tensor, vertices_xyz, uv) -> np.ndarray:
        """
        Deprecated name. Use `_cosine_map` for clarity.
        Returns an (H,W) grid of cos(theta). `frame` is used only for H,W inference.
        """
        H, W = int(frame.shape[0]), int(frame.shape[1])
        UV = self._to_numpy_2d(uv)
        XYZ = self._to_numpy_xyz(vertices_xyz)
        if XYZ.ndim == 3:   # take first frame if a batch was passed
            XYZ = XYZ[0]
        return self._cosine_map(H, W, UV, XYZ)

    def mask_lt_deg(self, angle_map: np.ndarray, deg: Optional[float] = None) -> np.ndarray:
        """
        Return boolean mask where cos(theta) < cos(deg). Non-finite values are False.
        """
        deg = self.default_degree_threshold if deg is None else float(deg)
        cos_thr = np.cos(np.deg2rad(deg))
        mask = np.isfinite(angle_map) & (angle_map < cos_thr)
        return mask

    # ------------------------- internals -------------------------

    @staticmethod
    def _compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray, eps: float) -> np.ndarray:
        V = np.asarray(vertices, dtype=np.float64)
        F = np.asarray(faces, dtype=np.int32)
        e0 = V[F[:, 1]] - V[F[:, 0]]
        e1 = V[F[:, 2]] - V[F[:, 0]]
        fn = np.cross(e0, e1)                         # area-weighted face normals
        N = np.zeros_like(V, dtype=np.float64)
        np.add.at(N, F[:, 0], fn)
        np.add.at(N, F[:, 1], fn)
        np.add.at(N, F[:, 2], fn)
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), eps)
        return N

    def _interp_uv_grid(
        self,
        uv: np.ndarray,               # (V,2)
        values: np.ndarray,           # (V,)
        H: int,
        W: int,
    ) -> np.ndarray:
        UV = np.clip(np.asarray(uv, dtype=np.float64), 0.0, 1.0)
        Z = np.asarray(values, dtype=np.float64)
        Uq, Vq = np.meshgrid(np.linspace(0.0, 1.0, W), np.linspace(0.0, 1.0, H))
        if self.flip_v:
            Vq = 1.0 - Vq
        return griddata((UV[:, 0], UV[:, 1]), Z, (Uq, Vq), method=self.method, fill_value=self.fill_value)

    def _cosine_map(self, H: int, W: int, uv: np.ndarray, vertices_xyz: np.ndarray) -> np.ndarray:
        """
        Return (H,W) of cos(theta) between per-vertex normals and camera normal.
        """
        normals = self._compute_vertex_normals(vertices_xyz, self.faces, self.eps)   # (V,3)
        cos_vals = normals @ self.cam                                               # (V,)
        return self._interp_uv_grid(uv, cos_vals, H, W)

    # ------------------------- helpers -------------------------

    @staticmethod
    def _to_numpy_2d(arr) -> np.ndarray:
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Expected (V,2) UV array. Got {arr.shape}.")
        return arr

    @staticmethod
    def _to_numpy_xyz(arr) -> np.ndarray:
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[None, ...]  # (1,V,3) broadcastable across T
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Expected (T,V,3) or (V,3) vertices. Got {arr.shape}.")
        return arr


# masker = UVAngleMasker(detector.mesh_faces)
# frames_uv_masked = masker(frames_uv, detector.landmarks_uv, landmarks, 60)
# frames_uv_masked.shape # >>> [T,64,64,C]