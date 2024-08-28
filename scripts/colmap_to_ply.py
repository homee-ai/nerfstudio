import re

def convert_to_ply(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # 写入PLY文件头
    with open(output_file, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex 16346\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # 处理每一行数据
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            
            parts = re.split(r'\s+', line.strip())
            if len(parts) >= 7:
                x, y, z = map(float, parts[1:4])
                r, g, b = map(int, parts[4:7])
                f.write(f"{x} {y} {z} {r} {g} {b}\n")

# 使用函数
input_txt = 'data/Saiens_island_nerfstudio/colmap/sparse/0/points3D.txt'
output_ply = 'data/Saiens_island_nerfstudio/colmap/sparse/0/points3D.ply'
convert_to_ply(input_txt, output_ply)