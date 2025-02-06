# 3) Read all depth maps and poses
# 4) Use Knn search to show mutliple point clouds which are color coded

import cv2
from dataclasses import dataclass
import open3d as o3d
from pyquaternion import Quaternion
import numpy as np
import viser
from pathlib import Path
import matplotlib.pyplot as plt


@dataclass
class PosedRGBDWithCalibration:
    # Calibration
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    # Pose: tf_w2c
    t_w2c: np.ndarray
    q_w2c: np.ndarray

    # RGBD image
    depth: np.ndarray
    rgb: np.ndarray

    def __init__(
        self,
        w: int,
        h: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        t: np.ndarray,
        q: np.ndarray,
        d: np.ndarray,
        rgb: np.ndarray,
    ):
        self.width = w
        self.height = h
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.t_w2c = t
        self.q_w2c = q
        self.depth = d
        self.rgb = rgb

        assert self.depth.dtype == np.uint16
        assert self.rgb.dtype == np.uint8

    def to_point_cloud(self):
        depth_o3d = o3d.geometry.Image(self.depth)
        rgb_o3d = o3d.geometry.Image(self.rgb)
        rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=False
        )

        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            self.width, self.height, self.fx, self.fy, self.cx, self.cy
        )

        tf_w2c = np.eye(4)
        tf_w2c[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(self.q_w2c)
        tf_w2c[:3, 3] = self.t_w2c

        # Extrinsic in Open3D is tf_w2c (Open3D internally takes inverse of extrinsic in order to provide a point cloud w.r.t world frame)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_o3d, intrinsic_o3d, tf_w2c
        )

        return pcd


def read_depth_map(filename):
    # Return a depth map in meter
    with open(filename, "r") as f:
        # Read dimensions
        data = f.readlines()
        rows, cols = map(int, data[0].split())

        # Read depth values (all elements after the first two)
        depth_values = list(map(float, data[1].split()))

        # Ensure we have the correct number of depth values
        if len(depth_values) != rows * cols:
            raise ValueError(
                f"Expected {rows * cols} depth values, but got {len(depth_values)}"
            )

        # Reshape to 2D array
        depth_map = np.array(depth_values).reshape((rows, cols))

    return depth_map


def get_scaled_intrinsic(
    w: int,
    h: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    w_scaled: int,
    h_scaled: int,
):
    fx_scaled = fx * w_scaled / w
    fy_scaled = fy * h_scaled / h
    cx_scaled = (cx - w / 2) / w * w_scaled + w_scaled / 2
    cy_scaled = (cy - h / 2) / h * h_scaled + h_scaled / 2

    return fx_scaled, fy_scaled, cx_scaled, cy_scaled


def main():
    server = viser.ViserServer()

    # Hardcode path here...
    depth_path = Path(
        "/home/fuyu/workspace/data/2025-01-26T04_09_44.533Z/3DGS/colmap/depth/0002_depth.txt"
    )
    color_path = Path(
        "/home/fuyu/workspace/data/2025-01-26T04_09_44.533Z/3DGS/colmap/post/images/0002.png"
    )
    arkit_poses = Path(
        "/home/fuyu/workspace/data/2025-01-26T04_09_44.533Z/3DGS/colmap/sparse/0/images.txt"
    )
    colmap_poses = Path(
        "/home/fuyu/workspace/data/2025-01-26T04_09_44.533Z/3DGS/colmap/post/sparse/offline/colmap/final/images.txt"
    )
    colmap_icp_poses = Path(
        "/home/fuyu/workspace/data/2025-01-26T04_09_44.533Z/3DGS/colmap/post/sparse/offline/colmap_ICP/final/images.txt"
    )
    k_path = Path(
        "/home/fuyu/workspace/data/2025-01-26T04_09_44.533Z/3DGS/colmap/sparse/0/distort_cameras.txt"
    )

    assert depth_path.is_file()
    assert color_path.is_file()
    assert arkit_poses.is_file()
    assert colmap_poses.is_file()
    assert colmap_icp_poses.is_file()
    assert k_path.is_file()

    # Read depth map
    depth = (read_depth_map(depth_path) * 1000).astype(np.uint16)

    # TODO: read intrinsics from k_path
    depth_fx, depth_fy, depth_cx, depth_cy = get_scaled_intrinsic(
        1280, 720, 949.321, 949.321, 645.6093, 361.05295, 256, 144
    )

    # Read color image
    rgb = cv2.imread(color_path, cv2.IMREAD_UNCHANGED)[:, :, ::-1]
    rgb = cv2.resize(rgb, (256, 144))

    # TODO: read pose from arkit_poses
    q_w2c = np.array(
        [0.018867744, -0.6829312, 0.7199132, 0.122367874]
    )  # [qw, qx, qy, qz]
    t_w2c = np.array([-0.21990505, 0.29071486, -0.7248777])  # [tx, ty, tz]

    posed_rgbd = PosedRGBDWithCalibration(
        256, 144, depth_fx, depth_fy, depth_cx, depth_cy, t_w2c, q_w2c, depth, rgb
    )
    pcd = posed_rgbd.to_point_cloud()

    while True:
        server.scene.add_frame("test")

        server.scene.add_point_cloud(
            "point cloud", np.asarray(pcd.points), np.asarray(pcd.colors), 0.05
        )


if __name__ == "__main__":
    main()
