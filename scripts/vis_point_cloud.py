# 4) Use Knn search to show multiple point clouds which are color coded
# Example: python scripts/vis_point_cloud.py -d /home/fuyu/workspace/data/2025-01-26T04_09_44.533Z/3DGS/colmap/depth/ -c /home/fuyu/workspace/data/2025-01-26T04_09_44.533Z/3DGS/colmap/post/images/ -p /home/fuyu/workspace/data/2025-01-26T04_09_44.533Z/3DGS/colmap/post/sparse/offline/colmap_ICP/final/images.txt -k /home/fuyu/workspace/data/2025-01-26T04_09_44.533Z/3DGS/colmap/sparse/0/distort_cameras.txt

import time
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import viser
from pyquaternion import Quaternion

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
        self,
        depth_folder: Path,
        color_folder: Path,
        calib_file: Path,
        pose_file: Path,
        is_pose_colmap: bool,
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

    # TODO: move to util or reuse code in arkit_pose_to_colmap.py
    def __arkit_pose_to_colmap_pose(arkit_pose):
        # arkit pose: tf_c2w
        #      ^ (y)
        #      |
        #      |---> (x)
        #     /
        #    /
        #   v (z)
        # colmap pose: tf_w2c
        #         ^ (z)
        #        /
        #       /
        #      |---> (x)
        #      |
        #      |
        #      v (y)

        arkit_pose = np.linalg.inv(arkit_pose)
        tf_arkit2colmap = np.eye(4)
        tf_arkit2colmap[:2, :2] = np.array([[1, 0, 0], [0, -1, 0], [0, -1, 0]])
        colmap_pose = tf_arkit2colmap @ arkit_pose

        return colmap_pose

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


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--depth",
        "-d",
        required=True,
        type=Path,
        help="path to a folder which stores depth images in *_depth.txt",
    )
    parser.add_argument(
        "--color",
        "-c",
        required=True,
        type=Path,
        help="path to a folder which stores color image in *.png",
    )
    parser.add_argument(
        "--pose",
        "-p",
        required=True,
        type=Path,
        help="path to images.txt which stores camera poses in colmap definition",
    )
    parser.add_argument(
        "--calib",
        "-k",
        required=True,
        type=Path,
        help="path to cameras.txt which stores camera intrinsic parameters",
    )
    parser.add_argument(
        "--index",
        "-i",
        default=-1,
        type=int,
        help="select which point cloud and its neighboring point cloud(s) to be shown in viser (default: save all point clouds in a ply file)",
    )
    return parser.parse_args()


def main():
    args = parse_arg()

    if args.index >= 0:
        server = viser.ViserServer()

    depth_path = args.depth
    color_path = args.color
    pose_path = args.pose
    k_path = args.calib

    assert depth_path.is_dir()
    assert color_path.is_dir()
    assert pose_path.is_file()
    assert k_path.is_file()

    # FIX: using arkit_poses won't work
    reader = DatasetReader(depth_path, color_path, k_path, pose_path, True)

    # TODO: show point cloud in a sliding window instead of all
    pcd_all = None
    for index in tqdm(range(2, 2000)):
        posed_rgbd = reader.get_posed_rgbd(index)
        pcd = posed_rgbd.to_point_cloud()

        if args.index >= 0:
            # server.scene.add_frame(
            #     f"frame{index}", wxyz=posed_rgbd.q_w2c, position=posed_rgbd.t_w2c
            # )

            server.scene.add_point_cloud(
                f"pcd{index}", np.asarray(pcd.points), np.asarray(pcd.colors)
            )
        else:
            if pcd_all is None:
                pcd_all = pcd
            else:
                pcd_all += pcd

    if args.index >= 0:
        while True:
            time.sleep(0.033)
    else:
        assert o3d.io.write_point_cloud("pcd_all.ply", pcd_all, write_ascii=True)


if __name__ == "__main__":
    main()
