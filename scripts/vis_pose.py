import argparse
import numpy as np
from evo.core import lie_algebra as lie
from evo.tools import file_interface
from evo.tools import plot
import matplotlib.pyplot as plt

def read_colmap_images(file_path):
    poses = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i in range(4, len(lines), 2):
            line = lines[i].split()
            qw, qx, qy, qz = map(float, line[1:5])
            tx, ty, tz = map(float, line[5:8])
            timestamp = float(line[0])  # 使用图像ID作为时间戳
            poses.append((timestamp, tx, ty, tz, qx, qy, qz, qw))
    return poses

def convert_to_tum(poses):
    tum_poses = []
    for pose in poses:
        timestamp, tx, ty, tz, qx, qy, qz, qw = pose
        tum_poses.append(f"{timestamp} {tx} {ty} {tz} {qx} {qy} {qz} {qw}")
    return tum_poses

def main():
    parser = argparse.ArgumentParser(description="读取COLMAP images.txt文件并可视化轨迹")
    parser.add_argument("--input_file", help="COLMAP images.txt文件的路径")
    parser.add_argument("--output_file", help="输出PNG文件的路径")
    args = parser.parse_args()

    # 读取COLMAP images.txt文件
    poses = read_colmap_images(args.input_file)

    # 转换为TUM格式
    tum_poses = convert_to_tum(poses)

    # 创建临时TUM文件
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write("\n".join(tum_poses))
        temp_file_path = temp_file.name

    # 使用evo读取轨迹
    traj = file_interface.read_tum_trajectory_file(temp_file_path)
    
    fig, ax = plt.subplots()

    # 绘制轨迹
    plot.trajectories(fig, [traj], plot_mode=plot.PlotMode.xy)

    # 设置图形标题和轴标签
    ax.set_title("trajectory (XY)")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")

    # 保存图形为PNG文件
    plt.savefig(args.output_file)
    print(f"轨迹图已保存为: {args.output_file}")

    # 删除临时文件
    import os
    os.unlink(temp_file_path)

if __name__ == "__main__":
    main()