import numpy as np
import open3d as o3d
import copy
import os
import pycolmap
import hloc.utils.read_write_model as hloc_io
import argparse
import shutil

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_points3d(file_path, base_dir):
    """Read points from points3D.txt file."""
    points = []
    colors = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
                
            elements = line.strip().split()
            if len(elements) < 7:
                continue
                
            x, y, z = map(float, elements[1:4])
            r, g, b = map(int, elements[4:7])
            
            points.append([x, y, z])
            colors.append([r/255.0, g/255.0, b/255.0])
    
    # Convert to numpy arrays
    points = np.array(points)
    colors = np.array(colors)
    
    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(os.path.join(base_dir, "points3D.ply"), pcd)
    
    return pcd

def read_obj_vertices(file_path):
    """Read vertices from OBJ file and create point cloud."""
    vertices = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                # Split line into elements and extract x, y, z coordinates
                elements = line.strip().split()
                x, y, z = map(float, elements[1:4])
                vertices.append([x, y, z])
    
    # Convert to numpy array
    vertices = np.array(vertices)
    
    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    
    return pcd

def draw_registration_result(source, target, transformation):
    """Visualize registration result."""
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])  # yellow
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # blue
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def register_point_clouds(source, target):
    """Perform ICP registration."""
    # Estimate normals if they don't exist
    source.estimate_normals()
    target.estimate_normals()
    
    # Initial transformation
    threshold = 0.05  # Distance threshold
    trans_init = np.eye(4)  # Initial transformation as identity matrix
    
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init)
    print(evaluation)   
    # Point-to-plane ICP
    print("Applying point-to-plane ICP...")
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    
    print("ICP fitness score:", reg_p2l.fitness)
    print("ICP RMSE:", reg_p2l.inlier_rmse)
    
    return reg_p2l.transformation

def transform_images(images_path, transformation, output_path):
    """
    Transform camera poses from images.txt using ICP transformation matrix.
    Accounts for world-to-camera transformation order.
    """
    images = hloc_io.read_images_text(images_path)
    
    with open(output_path, 'w') as f:
        # Write header...
        
        for image_id, image in images.items():
            # Get original world-to-camera transformation
            R_orig = qvec2rotmat(image.qvec)
            t_orig = image.tvec
            
            # Create 4x4 world-to-camera matrix
            Tw2c = np.eye(4)
            Tw2c[:3, :3] = R_orig
            Tw2c[:3, 3] = t_orig
            
            # New transformation: T_icp * Tw2c
            # This first transforms from world to camera, then applies ICP alignment
            pose_new = Tw2c @ np.linalg.inv(transformation)  # Note the inverse!
            
            # Extract new rotation and translation
            R_new = pose_new[:3, :3]
            t_new = pose_new[:3, 3]
            
            # Convert rotation matrix to quaternion
            # Note: We need to handle potential numerical instabilities
            from scipy.spatial.transform import Rotation
            r = Rotation.from_matrix(R_new)
            qw, qx, qy, qz = r.as_quat()[[3, 0, 1, 2]]  # Reorder to w,x,y,z
            
            # Write first line: image info
            f.write(f'{image_id} {qw} {qx} {qy} {qz} {t_new[0]} {t_new[1]} {t_new[2]} {image.camera_id} {image.name}\n')
            
            # print(image.point3D_ids.shape)
            # Write second line: points2D info
            # points2D_line = image.point3D_ids.flatten()
            f.write('\n')

def main(args):
    base_dir = args.base_dir
    output_dir = args.output_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read point clouds
    print("Reading point clouds...")
    source = read_points3d(os.path.join(base_dir, "points3D.txt"), base_dir)
    target = read_obj_vertices(os.path.join(base_dir, "../../../../../scene.obj"))
    
    # Perform ICP registration
    transformation = register_point_clouds(source, target)
    
    # Save transformed point cloud
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(transformation)
    o3d.io.write_point_cloud(os.path.join(output_dir, "points3D.ply"), source_transformed)
    
    # Copy and transform necessary files
    shutil.copy2(os.path.join(base_dir, "cameras.txt"), os.path.join(output_dir, "cameras.txt"))
    shutil.copy2(os.path.join(base_dir, "points3D.txt"), os.path.join(output_dir, "points3D.txt"))
    
    # Transform camera poses
    print("Transforming camera poses...")
    transform_images(
        os.path.join(base_dir, "images.txt"),
        transformation,
        os.path.join(output_dir, "images.txt")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform ICP alignment and transform camera poses.')
    parser.add_argument('--base_dir', type=str, required=True,
                      help='Base directory containing points3D.txt and images.txt')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for transformed files')
    args = parser.parse_args()
    main(args)