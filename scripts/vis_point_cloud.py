# 4) Use Knn search to show multiple point clouds which are color coded

import cv2
from dataclasses import dataclass
import open3d as o3d
from pyquaternion import Quaternion
import numpy as np
import viser
from pathlib import Path
import matplotlib.pyplot as plt
from nerfstudio.data.utils.colmap_parsing_utils import (
    read_cameras_text,
    read_images_text,
)


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
        # TODO: is truncation required for too small depth value or too large depth value?
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_o3d, intrinsic_o3d, tf_w2c
        )

        return pcd


class DatasetReader:
    def __init__(
        self, depth_folder: Path, color_folder: Path, calib_file: Path, pose_file
    ):
        assert "cameras.txt" in calib_file.name  # colmap format
        assert pose_file.name == "images.txt"  # colmap format

        self.id_to_depth_path = {
            int(depth_path.stem[:-6]): depth_path
            for depth_path in depth_folder.glob("*_depth.txt")
        }
        self.id_to_color_path = {
            int(color_path.stem): color_path
            for color_path in color_folder.glob("*.png")
        }

        self.id_to_k = read_cameras_text(calib_file)
        self.id_to_pose = read_images_text(pose_file)

        print(f"Number of depth maps: {len(self.id_to_depth_path)}")
        print(f"Number of color images: {len(self.id_to_color_path)}")
        print(f"Number of calibrations: {len(self.id_to_k)}")
        print(f"Number of poses: {len(self.id_to_pose)}")

    def peek(self, index: int):
        return (
            index in self.id_to_depth_path
            and index in self.id_to_color_path
            and index in self.id_to_k
            and index in self.id_to_pose
        )

    def get_posed_rgbd(self, index: int) -> PosedRGBDWithCalibration:
        # Convert stored depth value from meter to mm
        depth = (self.__read_depth_map(self.id_to_depth_path[index]) * 1000).astype(
            np.uint16
        )

        rgb = cv2.imread(self.id_to_color_path[index], cv2.IMREAD_UNCHANGED)[:, :, ::-1]

        # Read intrinsic for color image
        fx, fy, cx, cy = self.id_to_k[index].params

        # compute intrinsic for depth map and resize color image if required.
        if depth.shape != rgb.shape[:2]:
            fx, fy, cx, cy = self.__compute_scaled_intrinsic(
                rgb.shape[1],
                rgb.shape[0],
                fx,
                fy,
                cx,
                cy,
                depth.shape[1],
                depth.shape[0],
            )
            rgb = cv2.resize(rgb, (depth.shape[::-1]))

        return PosedRGBDWithCalibration(
            depth.shape[1],
            depth.shape[0],
            fx,
            fy,
            cx,
            cy,
            self.id_to_pose[index].tvec,
            self.id_to_pose[index].qvec,
            depth,
            rgb,
        )

    # TODO: move to util
    def __compute_scaled_intrinsic(
        self,
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

    # TODO: move to util
    def __read_depth_map(self, filename):
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


def main():
    server = viser.ViserServer()

    # TODO: use argparse
    # Hardcode path here...
    depth_path = Path(
        "/home/fuyu/workspace/data/2025-01-26T04_09_44.533Z/3DGS/colmap/depth/"
    )
    color_path = Path(
        "/home/fuyu/workspace/data/2025-01-26T04_09_44.533Z/3DGS/colmap/post/images/"
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

    assert depth_path.is_dir()
    assert color_path.is_dir()
    assert arkit_poses.is_file()
    assert colmap_poses.is_file()
    assert colmap_icp_poses.is_file()
    assert k_path.is_file()

    reader = DatasetReader(depth_path, color_path, k_path, arkit_poses)

    # TODO: not hardcode index range and sliding window size here
    while True:
        for index in range(2, 10):
            posed_rgbd = reader.get_posed_rgbd(index)
            pcd = posed_rgbd.to_point_cloud()

            server.scene.add_frame(
                f"frame{index}", wxyz=posed_rgbd.q_w2c, position=posed_rgbd.t_w2c
            )

            server.scene.add_point_cloud(
                f"pcd{index}", np.asarray(pcd.points), np.asarray(pcd.colors)
            )


if __name__ == "__main__":
    main()
