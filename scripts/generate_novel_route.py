""" It reads cameras' extrinsic and generates a novel pose for rendering.

python scripts/generate_novel_route.py \
    --roomplan-json data/20241202_145754/20241202_145545_RoomPlan.json \
    --extrinsic-path data/20241202_145754/20241202_145754_nerfstudio/colmap/colmap/0/images.txt \
    --route-height 1500 \
    --training-view-sampling-rate 200 \
    --camera-path-json camera_trajectory-1500.json \
&& ns-render camera-path \
    --load-config outputs/20241202_145754_nerfstudio/splatfacto/2024-12-24_010137/config.yml \
    --camera-path-filename camera_trajectory-1500.json \
    --output-path camera_trajectory-1500.mp4

"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import splines
import splines.quaternion
import viser.transforms as tf
from scipy import interpolate
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation
from trimesh import Trimesh

MM2METER = 1e-3


@dataclasses.dataclass
class Keyframe:
    """Adapted from Keyframe in nerfstudio/viewer/render_panel.py"""

    position: np.ndarray
    wxyz: np.ndarray
    fov_deg: float
    aspect: float

    @staticmethod
    def from_camera(pos: np.ndarray, wxyz: np.ndarray, fov: float, aspect: float) -> Keyframe:
        return Keyframe(
            pos,
            wxyz,
            fov_deg=fov,
            aspect=aspect,
        )


class CameraPath:
    """Adapted from CameraPath in nerfstudio/viewer/render_panel.py"""

    def __init__(self, framerate: int = 30, tension: float = 0.5, fov: float = 75, transition_sec: float = 2):
        self.framerate: int = framerate
        self.tension: float = tension
        self.fov: float = fov
        self.transition_sec: float = transition_sec

        self._keyframes: List[Keyframe] = []

    def add_keyframe(self, keyframe: Keyframe) -> None:
        self._keyframes.append(keyframe)

    def compute_duration(self) -> float:
        """Compute the total duration of the camera path."""
        assert len(self._keyframes) >= 2
        durations = []
        for i in range(len(self._keyframes) - 1):
            durations.append(self.transition_sec)
        return sum(durations)

    def compute_transition_times_cumsum(self) -> np.ndarray:
        """Compute the total duration of the trajectory."""
        total = 0.0
        out = [0.0]
        for i, keyframe in enumerate(self._keyframes):
            if i == 0:
                continue
            total += self.transition_sec
            out.append(total)

        return np.array(out)

    def spline_t_from_t_sec(self, time: np.ndarray) -> np.ndarray:
        """From a time value in seconds, compute a t value for our geometric
        spline interpolation. An increment of 1 for the latter will move the
        camera forward by one keyframe.

        We use a PCHIP spline here to guarantee monotonicity.
        """
        transition_times_cumsum = self.compute_transition_times_cumsum()
        spline_indices = np.arange(transition_times_cumsum.shape[0])

        interpolator = interpolate.PchipInterpolator(x=transition_times_cumsum, y=spline_indices)

        # Clip to account for floating point error.
        return np.clip(interpolator(time), 0, spline_indices[-1])

    def update_spline(self) -> None:
        self._orientation_spline = splines.quaternion.KochanekBartels(
            [
                splines.quaternion.UnitQuaternion.from_unit_xyzw(np.roll(keyframe.wxyz, shift=-1))
                for keyframe in self._keyframes
            ],
            tcb=(self.tension, 0.0, 0.0),
            endconditions="natural",
        )
        self._position_spline = splines.KochanekBartels(
            [keyframe.position for keyframe in self._keyframes],
            tcb=(self.tension, 0.0, 0.0),
            endconditions="natural",
        )

    def interpolate_pose(self, normalized_t: float):
        """Interpolate camera pose at a given normalized time."""
        assert len(self._keyframes) >= 2

        max_t = self.compute_duration()
        t = max_t * normalized_t
        spline_t = float(self.spline_t_from_t_sec(np.array(t)))

        quat = self._orientation_spline.evaluate(spline_t)
        assert isinstance(quat, splines.quaternion.UnitQuaternion)
        pos = self._position_spline.evaluate(spline_t)

        return tf.SE3.from_rotation_and_translation(
            tf.SO3(np.array([quat.scalar, *quat.vector])),
            pos,
        )


def gen_bbox_from_dimension(dimension: np.ndarray) -> Trimesh:
    """Generate 2D or 3D bounding box from a given `dimension`. `dimension` is
    composed of `width`, `height` and `depth`. The output could be either a 2D
    or 3D bounding box. If the z componenet in `dimension` is 0, then the 3D
    bounding box collapses to 2D bounding box.
    """
    x_len = dimension[0]
    y_len = dimension[1]
    z_len = dimension[2]

    if z_len == 0:  # 2D bounding box
        assert x_len != 0 and y_len != 0, f"x_len and y_len mustn't be zero. Got {x_len} and {y_len}"
        points = np.array(  # topleft, botleft, botright, topright
            [
                [-0.5 * x_len, 0.5 * y_len],
                [-0.5 * x_len, -0.5 * y_len],
                [0.5 * x_len, -0.5 * y_len],
                [0.5 * x_len, 0.5 * y_len],
            ]
        )  # (4, 2)
        triangles = Delaunay(points).simplices
        points = np.hstack(
            (
                points,
                np.zeros(points.shape[0]).reshape(-1, 1),
            )
        )  # Append zeros for z-axis to form (4, 3)

    else:  # 3D bounding box
        assert (
            x_len != 0 and y_len != 0 and z_len != 0
        ), f"x_len, y_len and z_len mustn't be zero. Got {x_len}, {y_len}, {z_len}"

        points = np.array(
            [
                [-0.5 * x_len, 0.5 * y_len, 0.5 * z_len],  # 0
                [-0.5 * x_len, -0.5 * y_len, 0.5 * z_len],  # 1
                [0.5 * x_len, -0.5 * y_len, 0.5 * z_len],  # 2
                [0.5 * x_len, 0.5 * y_len, 0.5 * z_len],  # 3
                [-0.5 * x_len, 0.5 * y_len, -0.5 * z_len],  # 4
                [-0.5 * x_len, -0.5 * y_len, -0.5 * z_len],  # 5
                [0.5 * x_len, -0.5 * y_len, -0.5 * z_len],  # 6
                [0.5 * x_len, 0.5 * y_len, -0.5 * z_len],  # 7
            ]
        )  # (8, 3)

        # Define 12 triangles, each represented by 3 points from the `points` array
        triangles = np.array(
            [
                # Front face (split into 2 triangles)
                [0, 1, 2],  # Triangle 1
                [0, 2, 3],  # Triangle 2
                # Back face (split into 2 triangles)
                [4, 5, 6],  # Triangle 3
                [4, 6, 7],  # Triangle 4
                # Left face (split into 2 triangles)
                [0, 1, 5],  # Triangle 5
                [0, 5, 4],  # Triangle 6
                # Right face (split into 2 triangles)
                [3, 2, 6],  # Triangle 7
                [3, 6, 7],  # Triangle 8
                # Top face (split into 2 triangles)
                [0, 3, 7],  # Triangle 9
                [0, 7, 4],  # Triangle 10
                # Bottom face (split into 2 triangles)
                [1, 2, 6],  # Triangle 11
                [1, 6, 5],  # Triangle 12
            ]
        )

    assert triangles.shape[1] == 3 and points.shape[1] == 3
    # triangles.shape: (num_triangles, 3)
    # vertices.shape: (num_points, 3)
    return Trimesh(vertices=points, faces=triangles)


def read_roomplan(roomplan_json: Path, targets: Dict[str, List]):
    # target includes objects and surfaces

    with open(roomplan_json, "r") as file:
        roomplan_root = json.load(file)
    roomplan_targets = {}  # A dict of surfaces and objects in RoomPlan
    for item_name, item_indices in targets.items():
        for idx, roomplan_thing in enumerate(roomplan_root[item_name]):
            if idx not in item_indices:
                continue

            transform = np.array(roomplan_thing["transform"]).reshape((4, 4))
            dimension = np.array(roomplan_thing["dimensions"])
            bbox = gen_bbox_from_dimension(dimension)  # bbox is either 2D or 3D

            transformed_bbox = bbox.apply_transform(transform.T)

            if item_name == "walls":
                name = "wall"
            elif item_name == "floors":
                name = "floor"
            elif item_name == "doors":
                name = "door"
            elif item_name == "objects":
                name = "object"
            else:
                raise ValueError(f"Wrong object name: {item_name}")

            roomplan_targets[f"{name}_{idx}"] = {
                "trimesh": transformed_bbox,
                "width_meter": dimension[0],
                "height_meter": dimension[1],
                "depth_meter": dimension[2],
            }

    return roomplan_targets


def read_camera_extrinsic(extrinsic_path: Path) -> List[Dict[str, Any]]:
    # Read extrinsic.
    cameras_extrinsic = []
    with open(extrinsic_path, "r") as file:
        for line in file:
            components = line.split()
            if len(components) == 0:  # Blank line
                continue
            if components[0] == "#":  # line starts with #
                continue
            if len(components) > 10:  # colmap stuff
                continue

            # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, FILE_NAME
            id_num = int(components[0])
            quaternion = list(map(float, components[1:5]))
            translation = list(map(float, components[5:8]))
            camera_id = int(components[8])
            image_filename = components[9]

            cameras_extrinsic.append(
                {
                    "quaternion": quaternion,
                    "translation": translation,
                    "camera_id": camera_id,
                    "image_filename": image_filename,
                }
            )
    return cameras_extrinsic


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def gen_lookforward_rot(curr_position: np.ndarray, next_position: np.ndarray) -> np.ndarray:
    look_to_dir = normalize(next_position - curr_position)

    up_dir = np.array([0, 1, 0], dtype=float)

    right_dir = normalize(np.cross(up_dir, look_to_dir))

    # Recompute the corrected up direction to ensure orthogonality
    corrected_up_dir = np.cross(look_to_dir, right_dir)

    # Construct the rotation matrix (3x3)
    rotation_matrix = np.stack([right_dir, corrected_up_dir, look_to_dir], axis=1)

    return rotation_matrix


def get_camera_center(quaternion: np.ndarray, translation: np.ndarray) -> np.ndarray:
    quaternion = np.roll(quaternion, -1)  # (w, x, y, z) -> (x, y, z, w)
    rotation = Rotation.from_quat(quaternion).as_matrix()  # (x, y, z, w)
    assert rotation.shape == (3, 3)
    return -1 * np.dot(rotation.T, translation)


def write_camera_path_json(
    keyframes: List[Keyframe],
    interpolated_poses: List[tf.SE3],
    fov: float,
    transition_sec: int,
    render_height: int,
    render_width: int,
    fps: int,
    duration: float,
    aspect_ratio: float,
    camera_path_json: Path,
):
    # write camera parameters to json
    json_data = {}
    json_data["default_fov"] = fov
    json_data["default_transition_sec"] = transition_sec

    json_data["keyframes"] = []
    # writing keyframe is not important, it won't change result
    for keyframe in keyframes:
        pose = tf.SE3.from_rotation_and_translation(
            tf.SO3(keyframe.wxyz),
            keyframe.position,
        )
        keyframe_dict = {
            "matrix": pose.as_matrix().flatten().tolist(),
            "fov": keyframe.fov_deg,
            "aspect": keyframe.aspect,
            "override_transition_enabled": False,
            "override_transition_sec": None,
        }
        json_data["keyframes"].append(keyframe_dict)

    json_data["camera_type"] = "perspective"
    json_data["render_height"] = render_height
    json_data["render_width"] = render_width
    json_data["fps"] = fps
    json_data["seconds"] = duration
    json_data["is_cycle"] = False
    json_data["smoothness_value"] = 0

    json_data["camera_path"] = []
    for pose in interpolated_poses:
        pose = tf.SE3.from_rotation_and_translation(
            pose.rotation(),
            pose.translation(),
        )
        camera_path_dict = {
            "camera_to_world": pose.as_matrix().flatten().tolist(),
            "fov": fov,
            "aspect": aspect_ratio,
        }
        json_data["camera_path"].append(camera_path_dict)

    # Dump to a JSON file
    with open(camera_path_json, "w") as json_file:
        json.dump(json_data, json_file, indent=4)  # Save to file with indentation


def main(
    roomplan_json: Path,
    extrinsic_path: Path,
    camera_path_json: Path,
    relative_route_height: int,
    training_view_sampling_rate: int,
    video_fps: int,
    transition_sec: int,
    fov: float,
    aspect_ratio: float,
    render_height: int,
    render_width: int,
):
    cameras_extrinsic = read_camera_extrinsic(extrinsic_path)
    print("number of training view: ", len(cameras_extrinsic))
    roomplan_targets = read_roomplan(roomplan_json, {"floors": [0]})
    assert np.all(
        np.isclose(
            roomplan_targets["floor_0"]["trimesh"].vertices[:, 1], roomplan_targets["floor_0"]["trimesh"].vertices[0, 1]
        )
    )  # all vertices y coordinates must be the same
    route_height = relative_route_height * MM2METER + roomplan_targets["floor_0"]["trimesh"].vertices[0, 1]
    print("route_height: ", route_height)

    cameras_center = []
    for extrinsic in cameras_extrinsic:
        camera_center = get_camera_center(extrinsic["quaternion"], extrinsic["translation"])
        cameras_center.append(camera_center)

    keyframes_center = []
    for i in range(0, len(cameras_center), training_view_sampling_rate):
        keyframes_center.append(cameras_center[i])

    keyframes_center = np.array(keyframes_center)
    cameras_center = np.array(cameras_center)
    assert cameras_center.shape[1] == 3

    print("camera_centers: ", cameras_center.shape)
    print("    x, y, z (min):", cameras_center.min(axis=0))
    print("    x, y, z (max):", cameras_center.max(axis=0))
    plt.figure(figsize=(6, 6))
    plt.scatter(cameras_center[:, 2], cameras_center[:, 0], c="blue", s=10, alpha=0.7)
    plt.plot(keyframes_center[:, 2], keyframes_center[:, 0], marker="o", linestyle="-", color="red")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig("camera_centers.png", dpi=150, bbox_inches="tight")

    keyframes = []
    for i in range(0, len(keyframes_center)):
        curr_pos = keyframes_center[i]
        curr_pos = np.array([curr_pos[0], route_height, curr_pos[2]])

        if i == len(keyframes_center) - 1:
            # The last keyframe points to the initial one.
            next_pos = keyframes_center[0]
        else:
            next_pos = keyframes_center[i + 1]
        next_pos = np.array([next_pos[0], route_height, next_pos[2]])

        lookforward_rot = gen_lookforward_rot(curr_pos, next_pos)
        c2w_lookforward_rot = lookforward_rot.T
        quat = Rotation.from_matrix(c2w_lookforward_rot).as_quat()  # (x, y, z, w)
        wxyz = np.roll(quat, 1)  # (x, y, z, w) -> (w, x, y, z)
        keyframes.append(Keyframe.from_camera(pos=curr_pos, wxyz=wxyz, fov=fov, aspect=aspect_ratio))

    camerapath = CameraPath(framerate=video_fps, tension=0.5, fov=fov, transition_sec=transition_sec)
    for keyframe in keyframes:
        camerapath.add_keyframe(keyframe=keyframe)

    total_duration = camerapath.compute_duration()
    num_frames = int(total_duration * camerapath.framerate)
    print("num_frames", num_frames)
    print("total duration", total_duration)

    camerapath.update_spline()

    interpolated_poses = []
    for i in range(num_frames):
        pose = camerapath.interpolate_pose(normalized_t=i / num_frames)
        interpolated_poses.append(pose)

    write_camera_path_json(
        keyframes,
        interpolated_poses,
        fov,
        transition_sec,
        render_height,
        render_width,
        camerapath.framerate,
        total_duration,
        aspect_ratio,
        camera_path_json,
    )


def get_args():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "--roomplan-json",
        type=Path,
        required=True,
        help="Path to the RoomPlan.json",
    )
    parser.add_argument("--extrinsic-path", type=Path, required=True)
    parser.add_argument("--camera-path-json", type=Path, required=True)
    parser.add_argument("--route-height", type=int, required=True, help="Route height in millimeters.")
    parser.add_argument("--training-view-sampling-rate", type=int, required=True)
    parser.add_argument("--fov", type=float, default=75)
    parser.add_argument("--aspect-ratio", type=float, default=1.7777)
    parser.add_argument("--transition-sec", type=int, default=5)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--render-height", type=int, default=1080)
    parser.add_argument("--render-width", type=int, default=1920)

    args = parser.parse_args()

    if args.route_height <= 0:
        parser.error("Route height must be positive.")

    return args


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "t", "yes", "1"}:
        return True
    elif value.lower() in {"false", "f", "no", "0"}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'.")


if __name__ == "__main__":
    args = get_args()
    main(
        roomplan_json=args.roomplan_json,
        extrinsic_path=args.extrinsic_path,
        camera_path_json=args.camera_path_json,
        relative_route_height=args.route_height,
        training_view_sampling_rate=args.training_view_sampling_rate,
        video_fps=args.fps,
        transition_sec=args.transition_sec,
        fov=args.fov,
        aspect_ratio=args.aspect_ratio,
        render_height=args.render_height,
        render_width=args.render_width,
    )
