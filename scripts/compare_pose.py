import argparse
import numpy as np
import os
from evo.core import lie_algebra as lie
from evo.tools import file_interface
from evo.tools import plot
from evo.core import sync
from evo.core import metrics
from evo.core import trajectory
import matplotlib.pyplot as plt

def read_colmap_images(file_path):
    poses = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i in range(4, len(lines), 2):
            line = lines[i].split()
            qw, qx, qy, qz = map(float, line[1:5])
            tx, ty, tz = map(float, line[5:8])
            timestamp = float(line[0])  # Using image ID as timestamp
            poses.append((timestamp, tx, ty, tz, qx, qy, qz, qw))
    return poses

def convert_to_tum(poses):
    tum_poses = []
    for pose in poses:
        timestamp, tx, ty, tz, qx, qy, qz, qw = pose
        tum_poses.append(f"{timestamp} {tx} {ty} {tz} {qx} {qy} {qz} {qw}")
    return tum_poses

def plot_trajectories(traj1, traj2, title, output_path, plot_mode):
    fig, ax = plt.subplots()
    plot.trajectories(fig, [traj1, traj2], plot_mode=plot_mode)
    ax.set_title(title)
    ax.legend(["Trajectory 1", "Trajectory 2"])
    ax.set_xlabel("X")
    ax.set_ylabel("Y" if plot_mode == plot.PlotMode.xy else "Z")
    plt.savefig(output_path)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Read two COLMAP images.txt files, align poses and visualize trajectories")
    parser.add_argument("--input_file1", help="Path to the first COLMAP images.txt file")
    parser.add_argument("--input_file2", help="Path to the second COLMAP images.txt file")
    parser.add_argument("--output_dir", help="Path to the output directory")
    args = parser.parse_args()

    # Read two COLMAP images.txt files
    poses1 = read_colmap_images(args.input_file1)
    poses2 = read_colmap_images(args.input_file2)

    # Convert to TUM format
    tum_poses1 = convert_to_tum(poses1)
    tum_poses2 = convert_to_tum(poses2)

    # Create temporary TUM files
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file1:
        temp_file1.write("\n".join(tum_poses1))
        temp_file_path1 = temp_file1.name
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file2:
        temp_file2.write("\n".join(tum_poses2))
        temp_file_path2 = temp_file2.name

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Read trajectories using evo
    traj1 = file_interface.read_tum_trajectory_file(temp_file_path1)
    traj2 = file_interface.read_tum_trajectory_file(temp_file_path2)

    # Visualization before alignment
    plot_trajectories(traj1, traj2, "Trajectories Before Alignment (XY)", os.path.join(args.output_dir, "trajectories_before_alignment_xy.png"), plot.PlotMode.xy)
    plot_trajectories(traj1, traj2, "Trajectories Before Alignment (XZ)", os.path.join(args.output_dir, "trajectories_before_alignment_xz.png"), plot.PlotMode.xz)

    # Align trajectories
    traj2.align(traj1)

    # Visualization after alignment
    plot_trajectories(traj1, traj2, "Trajectories After Alignment (XY)", os.path.join(args.output_dir, "trajectories_after_alignment_xy.png"), plot.PlotMode.xy)
    plot_trajectories(traj1, traj2, "Trajectories After Alignment (XZ)", os.path.join(args.output_dir, "trajectories_after_alignment_xz.png"), plot.PlotMode.xz)

    # Time series plot
    fig_time, axs_time = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    plot.trajectories(fig_time, [traj1, traj2], plot_mode=plot.PlotMode.xyz)
    axs_time[0].set_title("X over Time")
    axs_time[1].set_title("Y over Time")
    axs_time[2].set_title("Z over Time")
    axs_time[2].set_xlabel("Time")
    fig_time.legend(["Trajectory 1", "Trajectory 2 (Aligned)"])
    plt.savefig(os.path.join(args.output_dir, "trajectories_time_series.png"))
    plt.close(fig_time)

    print(f"Evaluation visualization results have been saved to directory: {args.output_dir}")

    # Delete temporary files
    os.unlink(temp_file_path1)
    os.unlink(temp_file_path2)

if __name__ == "__main__":
    main()