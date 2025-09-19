import logging
from typing import Tuple, Dict

import numpy as np
import open3d as o3d

from pypointsetreg import fuzzy_cluster_reg


def normalize(x: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Normalize a point cloud in array format and store the normalizing constants.

    :param x: Input point cloud array, Nx3
    :type x: np.ndarray
    :return: Normalized point cloud array, dict with normalization constants (xd and xscale)
    :rtype: Tuple[np.ndarray, Dict[str, float]]
    """
    N = x.shape[0]
    pre_normal = {}
    pre_normal["xd"] = np.mean(x, axis=0)
    x = x - pre_normal["xd"]
    pre_normal["xscale"] = np.sqrt(np.sum(x**2) / N)
    X = x / pre_normal["xscale"]
    return X, pre_normal


def denormalize(pre_normal: Dict[str, float], data_deformed: np.ndarray) -> np.ndarray:
    """
    Revert normalization of a point cloud array.

    :param pre_normal: Dict with normalization constants (xd and xscale)
    :type pre_normal: Dict[str, float]
    :param data_deformed: Deformed point cloud array, Nx3
    :type data_deformed: np.ndarray
    :return: Unnormalized point cloud array
    :rtype: np.ndarray
    """
    data_denormal = data_deformed * pre_normal["xscale"] + pre_normal["xd"]
    return data_denormal


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    pcd_source = o3d.io.read_point_cloud("data/tr_reg_059.ply")
    pcd_target = o3d.io.read_point_cloud("data/tr_reg_057.ply")

    pcd_source.voxel_down_sample(voxel_size=0.1)
    pcd_target.voxel_down_sample(voxel_size=0.1)

    # convert to numpy arrays and normalize them
    arr_source = np.asarray(pcd_source.points)
    arr_target = np.asarray(pcd_target.points)
    arr_source, pre_source = normalize(arr_source)
    arr_target, pre_target = normalize(arr_target)

    o3d.visualization.draw_geometries(
        [pcd_source, pcd_target],
        window_name="Source (colored) and target (black) point clouds",
    )

    alpha, T = fuzzy_cluster_reg(arr_source, arr_target)
    arr_target_deformed = denormalize(pre_target, T)

    pcd_target_deformed = o3d.geometry.PointCloud()
    pcd_target_deformed.points = o3d.utility.Vector3dVector(arr_target_deformed)

    o3d.visualization.draw_geometries(
        [pcd_target, pcd_target_deformed],
        window_name="Deformed source (colored) and target (black) point clouds",
    )
