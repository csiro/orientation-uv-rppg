import numpy as np
from open3d.geometry import TriangleMesh
from open3d.utility import Vector3dVector, Vector3iVector
from src.datamodules.datamodels import DataModel

from numpy import ndarray
from typing import *


class Mesh(DataModel):
    def __init__(self, data: Optional[TriangleMesh] = TriangleMesh(), *args, **kwargs) -> None:
        super(Mesh, self).__init__(data, *args, **kwargs)

    def update(self, vertices: ndarray, triangles: ndarray) -> None:
        self.data.vertices = Vector3dVector(vertices)
        self.data.triangles = Vector3iVector(triangles)
    
    @property
    def vertex_normals(self) -> ndarray:
        self.data.compute_vertex_normals()
        return np.asarray(self.data.vertex_normals)
