import torch
import numpy as np
import torch.nn.functional as F
from scipy.spatial import Delaunay, ConvexHull

from torch import Tensor


class GPUPiecewiseAffineTransform:
    """
    GPU piecewise-affine warp from UV -> image with an exterior mask.

    Summary
    -------
    - Inputs: reference UV keypoints [N,2] and per-frame image-space landmarks [N,2].
    - Build: Delaunay triangulation on UV points. Convex hull on the same UVs.
    - Run: Create UV grid, fast-reject pixels outside the convex hull, then
           map only inside pixels by barycentric interpolation of matched image triangles.
    - Output: warped image in UV frame with zero background; optional mask.

    Assumptions
    -----------
    - UV keypoints and image landmarks share 1:1 ordering.
    - UV keypoints are in [0,1]^2 or pixels; pixels are auto-normalized.
    - Frame is [C,H,W] or [H,W,C], uint8 or float in [0,1].

    Notes
    -----
    - Complexity per chunk: O(m*T) for m points vs T triangles; the convex hull
      prefilter reduces m. Choose chunk_size to manage memory.
    - Set soft_edge_gamma>0 for a soft falloff at triangle borders using min barycentric.
    """

    def __init__(self, keypoints_uv, device: str = "cuda"):
        """
        Parameters
        ----------
        keypoints_uv : array-like, shape [N,2]
            Reference UV coordinates. Pixels allowed; will be normalized to [0,1].
        device : str
            Torch device string.
        """
        self.device = torch.device(device)

        # Normalize UVs to [0,1] if they look like pixels
        if torch.is_tensor(keypoints_uv):
            uv_np = keypoints_uv.detach().cpu().numpy()
        else:
            uv_np = np.asarray(keypoints_uv, dtype=np.float32)
        if uv_np.max() > 1.5:
            uv_np = uv_np / uv_np.max()

        self.keypoints_uv = torch.tensor(uv_np, dtype=torch.float32, device=self.device)  # [N,2]
        self.n_landmarks = int(self.keypoints_uv.shape[0])

        # Delaunay triangulation over real UV points only
        tri = Delaunay(self.keypoints_uv.detach().cpu().numpy())
        self.face_triangles = torch.tensor(tri.simplices, dtype=torch.long, device=self.device)     # [T,3]
        self.uv_face_tris = self.keypoints_uv[self.face_triangles]                                   # [T,3,2]

        # Convex hull in CCW order; cache edges for vectorized half-space tests
        hull = ConvexHull(self.keypoints_uv.detach().cpu().numpy())
        self.hull_idx = torch.tensor(hull.vertices, dtype=torch.long, device=self.device)            # [K]
        self.hull_uv = self.keypoints_uv[self.hull_idx]                                              # [K,2]
        self._hull_edges = torch.roll(self.hull_uv, -1, 0) - self.hull_uv                            # [K,2]

    # ------------------------------- public API -------------------------------

    def __call__(self,
        frames: Tensor,
        landmarks: Tensor,
        output_size: int,
    ) -> Tensor:
        frames_uv = []
        for frame, frame_landmarks in zip(frames, landmarks):
            frame_uv = self.estimate_and_transform(
                frame=frame,
                landmarks_xy=frame_landmarks[:,:2],
                output_size=output_size,
                return_mask=False,
            )
            frames_uv.append(frame_uv)
        frames_uv = torch.stack(frames_uv, dim=0)
        return frames_uv

    # ------------------------------ core routines -----------------------------

    def estimate_and_transform(
        self,
        frame,
        landmarks_xy,
        output_size: int,
        return_mask: bool = False,
        chunk_size: int = 65536,
        soft_edge_gamma: float = 0.0,
    ):
        """
        Warp an input frame into UV space.

        Parameters
        ----------
        frame : torch.Tensor | np.ndarray
            [C,H,W] or [H,W,C]. dtype uint8 or float. Range [0,255] or [0,1].
        landmarks_xy : torch.Tensor | np.ndarray
            [N,2] or [N,3] image-space pixels aligned with keypoints_uv order.
        output_size : int
            Output H = W = output_size in UV space.
        return_mask : bool
            If True, also return mask [1,H,W] of interior pixels.
        chunk_size : int
            Number of UV pixels processed per barycentric batch.
        soft_edge_gamma : float
            0 for hard mask. >0 applies smooth falloff using min barycentric^gamma.

        Returns
        -------
        warped : torch.Tensor
            [C,output_size,output_size], float32 in [0,1].
        mask : torch.Tensor (optional)
            [1,output_size,output_size], float32 in [0,1].
        """
        img = self._to_image_tensor(frame)             # [1,C,H,W]
        lm  = self._to_landmarks_tensor(landmarks_xy)  # [N,2]
        _, C, H, W = img.shape

        # UV grid [HW,2] in [0,1]
        h = w = int(output_size)
        ys, xs = torch.meshgrid(
            torch.linspace(0, 1, h, device=self.device),
            torch.linspace(0, 1, w, device=self.device),
            indexing="ij",
        )
        grid_uv = torch.stack([xs, ys], dim=-1).reshape(-1, 2)  # [HW,2]

        # Fast convex-hull test in UV
        inside_hull = self._point_in_convex_hull_uv(grid_uv)     # [HW] bool

        # Map only hull-interior pixels
        src_xy = torch.zeros((h * w, 2), device=self.device, dtype=torch.float32)
        inside_tri = torch.zeros(h * w, device=self.device, dtype=torch.bool)
        min_bary = torch.zeros(h * w, device=self.device, dtype=torch.float32) if soft_edge_gamma > 0 else None

        if inside_hull.any():
            coords, in_any, mb = self._uv_to_image_coords(
                grid_uv[inside_hull], lm,
                chunk_size=chunk_size,
                return_min_bary=(soft_edge_gamma > 0)
            )
            src_xy[inside_hull] = coords
            inside_tri[inside_hull] = in_any
            if soft_edge_gamma > 0:
                min_bary[inside_hull] = torch.where(in_any, mb, torch.zeros_like(mb))

        # Build normalized sampling grid
        src_xy = src_xy.reshape(h, w, 2)
        sampling_grid = torch.stack(
            [
                2.0 * src_xy[..., 0] / (W - 1) - 1.0,
                2.0 * src_xy[..., 1] / (H - 1) - 1.0,
            ],
            dim=-1,
        )  # [h,w,2]
        inside = inside_tri.reshape(h, w)
        sampling_grid[~inside] = 2.0  # invalidate outside -> zeros with padding_mode='zeros'

        warped = F.grid_sample(
            img, sampling_grid.unsqueeze(0),
            mode="bilinear", padding_mode="zeros", align_corners=True
        )  # [1,C,h,w]

        # Construct mask and kill bilinear bleed
        if soft_edge_gamma > 0:
            mask = (min_bary.reshape(1, 1, h, w).clamp(0, 1) ** soft_edge_gamma)
        else:
            mask = inside.reshape(1, 1, h, w).float()

        out = (warped * mask).squeeze(0)  # [C,h,w]
        out = out.permute(1,2,0)
        return (out, mask.squeeze(0)) if return_mask else out

    @torch.no_grad()
    def _uv_to_image_coords(self, uv_points, landmarks_xy, chunk_size=65536, return_min_bary=False):
        """
        Map UV points -> image coordinates by barycentric interpolation.

        Parameters
        ----------
        uv_points : torch.Tensor [M,2]
        landmarks_xy : torch.Tensor [N,2]
        chunk_size : int
        return_min_bary : bool

        Returns
        -------
        coords : torch.Tensor [M,2]
        inside_any : torch.Tensor [M] bool
        min_bary : torch.Tensor [M] float (min barycentric) if requested else None
        """
        M = uv_points.shape[0]
        coords = torch.zeros((M, 2), device=self.device, dtype=torch.float32)
        inside_any_all = torch.zeros(M, device=self.device, dtype=torch.bool)
        min_bary_all = torch.zeros(M, device=self.device, dtype=torch.float32) if return_min_bary else None

        for s in range(0, M, chunk_size):
            e = min(M, s + chunk_size)
            pts = uv_points[s:e]  # [m,2]

            tri_idx, bary, inside_any = self._find_triangles_batch(pts, self.uv_face_tris)  # [m], [m,3], [m]
            img_tris = landmarks_xy[self.face_triangles[tri_idx]]  # [m,3,2]
            xy = torch.sum(bary.unsqueeze(-1) * img_tris, dim=1)   # [m,2]

            coords[s:e] = xy
            inside_any_all[s:e] = inside_any

            if return_min_bary:
                mb = bary.min(dim=-1).values
                min_bary_all[s:e] = torch.where(inside_any, mb, torch.zeros_like(mb))

        return coords, inside_any_all, min_bary_all

    @staticmethod
    def _find_triangles_batch(points, tri_verts):
        """
        Vectorized point-in-triangle over a set of triangles.

        Parameters
        ----------
        points   : torch.Tensor [m,2]
        tri_verts: torch.Tensor [T,3,2] triangle vertices in UV

        Returns
        -------
        tri_idx   : torch.LongTensor [m] index of chosen triangle
        bary_sel  : torch.FloatTensor [m,3] barycentric coords (valid where inside_any=True)
        inside_any: torch.BoolTensor [m]
        """
        m = points.shape[0]

        v0 = tri_verts[:, 0, :].unsqueeze(0)  # [1,T,2]
        v1 = tri_verts[:, 1, :].unsqueeze(0)
        v2 = tri_verts[:, 2, :].unsqueeze(0)
        p  = points.unsqueeze(1)              # [m,1,2]

        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p  = p  - v0

        dot00 = (v0v2 * v0v2).sum(-1)  # [1,T]
        dot01 = (v0v2 * v0v1).sum(-1)  # [1,T]
        dot02 = (v0v2 * v0p ).sum(-1)  # [m,T]
        dot11 = (v0v1 * v0v1).sum(-1)  # [1,T]
        dot12 = (v0v1 * v0p ).sum(-1)  # [m,T]

        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-10)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        w = 1.0 - u - v
        bary = torch.stack([w, u, v], dim=-1)     # [m,T,3]

        inside = (bary >= -1e-6).all(dim=-1)      # [m,T]
        inside_any = inside.any(dim=-1)           # [m]

        # pick among valid triangles using distance to barycenter as tie-breaker
        bary_dist = (bary - 1.0 / 3.0).abs().sum(dim=-1)  # [m,T]
        bary_dist[~inside] = float("inf")
        tri_idx = torch.argmin(bary_dist, dim=-1)         # [m]

        ar = torch.arange(m, device=points.device)
        bary_sel = bary[ar, tri_idx]                      # [m,3]
        return tri_idx, bary_sel, inside_any

    # ------------------------------ small utils --------------------------------

    def _point_in_convex_hull_uv(self, pts):
        """
        Vectorized point-in-convex-polygon for the precomputed UV hull.

        Parameters
        ----------
        pts : torch.Tensor [M,2] in [0,1]

        Returns
        -------
        inside : torch.BoolTensor [M]
        """
        # cross(e_i, pts - v_i) >= 0 for all edges (CCW hull) -> inside
        rel = pts.unsqueeze(1) - self.hull_uv.unsqueeze(0)          # [M,K,2]
        e = self._hull_edges.unsqueeze(0)                            # [1,K,2]
        cross = e[..., 0] * rel[..., 1] - e[..., 1] * rel[..., 0]    # [M,K]
        return (cross >= -1e-6).all(dim=1)

    def _to_image_tensor(self, frame):
        """Convert input to [1,C,H,W] float32 in [0,1] on device."""
        if torch.is_tensor(frame):
            img = frame.to(self.device)
        else:
            img = torch.tensor(frame, dtype=torch.float32, device=self.device)
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)
        if img.dtype != torch.float32:
            img = img.float()
        if img.max() > 1.1:
            img = img / 255.0
        if img.ndim == 3:
            img = img.unsqueeze(0)
        return img

    def _to_landmarks_tensor(self, landmarks_xy):
        """Convert landmarks to [N,2] float32 on device, matching stored UV count."""
        if torch.is_tensor(landmarks_xy):
            lm = landmarks_xy.to(self.device).float()
        else:
            lm = torch.tensor(landmarks_xy, dtype=torch.float32, device=self.device)
        if lm.shape[1] > 2:
            lm = lm[:, :2]
        if lm.shape[0] > self.n_landmarks:
            lm = lm[: self.n_landmarks]
        return lm


# device = "cuda" if torch.cuda.is_available() else "cpu"
# warper = GPUPiecewiseAffineTransform(detector.landmarks_uv, device=device)
# frames_uv = warper(frames_xy, landmarks_xyz, output_size=64)
# frames_uv.shape # >>> [T,64,64,C]