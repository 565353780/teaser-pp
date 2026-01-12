import torch
import numpy as np
import open3d as o3d
from typing import Union

def toNumpy(
    data: Union[torch.Tensor, np.ndarray, list],
    dtype=np.float64,
) -> np.ndarray:
    if isinstance(data, list):
        data = np.asarray(data)
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    data = data.astype(dtype)
    return data

def toPcd(
    pts: Union[torch.Tensor, np.ndarray, list],
) -> o3d.geometry.PointCloud:
    pts = toNumpy(pts, np.float64).reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd
