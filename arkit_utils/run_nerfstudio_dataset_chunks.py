import os
import shutil
import argparse
from pathlib import Path
import math
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from background_color import get_background_color
from dataclasses import dataclass
import subprocess

@dataclass
class TrainingConfig:
    """Base configuration for training parameters."""
    background_color: Optional[str] = None
    use_mesh_initialization: bool = True
    combine_mesh_sfm: bool = True
    rasterize_mode: str = "antialiased"
    use_scale_regularization: bool = False
    camera_optimizer_mode: str = "SO3xR3"
    train_cameras_sampling: str = "fps"
    sh_degree: int = 2
    use_bilateral_grid: bool = True
    num_downscales: int = 2
    auto_scale_poses: bool = False
    center_method: str = "none"
    orientation_method: str = "none"
    eval_mode: str = "fraction"
    train_split_fraction: float = 1.0
    make_share_url: bool = True
    visualization: str = "viewer+tensorboard"
    quit_on_completion: bool = True

@dataclass
class ChunkTrainingConfig(TrainingConfig):
    """Configuration specific to chunk training."""
    max_iterations: int = 10000
    stop_split_at: Optional[int] = None
    camera_optimizer_mode: str = "off"
    
@dataclass
class FinalTrainingConfig(TrainingConfig):
    """Configuration specific to final training."""
    max_iterations: int = 10000
    load_dir: Optional[str] = None
    num_downscales:int = 0
    

def create_directory(path):
    os.makedirs(path, exist_ok=True)

def parse_colmap_points3D(points3D_path: Path) -> Dict[int, Tuple[np.ndarray, List[int]]]:
    """Parse COLMAP points3D.txt file.
    Returns dict mapping point3D_id to (xyz, list of image_ids where point is visible)
    """
    points3D = {}
    with open(points3D_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            data = line.split()
            point3D_id = int(data[0])
            xyz = np.array([float(x) for x in data[1:4]])
            # Skip RGB
            image_ids = [int(x) for x in data[8::2]]  # Every other element starting from 8th
            points3D[point3D_id] = (xyz, image_ids)
    return points3D

def filter_points3D_for_chunk(points3D: Dict[int, Tuple[np.ndarray, List[int]]], 
                            chunk_image_ids: List[int]) -> Tuple[Dict[int, Tuple[np.ndarray, List[int]]], np.ndarray]:
    """Filter points3D to only keep those visible in chunk images.
    Returns filtered points dict and bounding box (min_xyz, max_xyz)
    """
    chunk_image_ids = set(chunk_image_ids)
    filtered_points = {}
    
    xyz_points = []
    for point_id, (xyz, image_ids) in points3D.items():
        # Keep point if visible in any chunk image
        if any(img_id in chunk_image_ids for img_id in image_ids):
            filtered_points[point_id] = (xyz, image_ids)
            xyz_points.append(xyz)
    
    xyz_points = np.stack(xyz_points, axis=0)
    min_xyz = np.min(xyz_points, axis=0)
    max_xyz = np.max(xyz_points, axis=0)
    
    # Add small padding to bounding box (5% of size)
    size = max_xyz - min_xyz
    padding = size * 0.05
    min_xyz -= padding
    max_xyz += padding
    
    bbox = np.stack([min_xyz, max_xyz], axis=0)
    return filtered_points, bbox

def write_points3D(points3D: Dict[int, Tuple[np.ndarray, List[int]]], output_path: Path):
    """Write points3D to COLMAP format."""
    with open(output_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        
        for point_id, (xyz, image_ids) in points3D.items():
            # Use placeholder RGB values
            rgb = "128 128 128"
            error = "1.0"
            # Create track string
            track = " ".join([f"{img_id} 0" for img_id in image_ids])
            line = f"{point_id} {xyz[0]} {xyz[1]} {xyz[2]} {rgb} {error} {track}\n"
            f.write(line)

def filter_gaussians_by_bbox(checkpoint: Dict, device: str = "cuda", check_optimization: bool = True) -> Dict:
    """Filter gaussians to keep only those inside the bounding box and check optimization status."""
    means = checkpoint['pipeline']['_model.gauss_params.means']
    
    # Calculate scene centroid and std deviation
    scene_centroid = torch.mean(means, dim=0)
    scene_std = torch.std(means - scene_centroid, dim=0)
    
    # Create initial mask
    inside_mask = torch.ones(len(means), dtype=torch.bool, device=device)
    
    # Filter 1: Remove distant gaussians (3 sigma rule)
    distance = torch.norm(means - scene_centroid, dim=1)
    std_distance = torch.std(distance)
    distance_mask = distance < 3 * std_distance
    inside_mask &= distance_mask
    
    # Filter 2: Remove under-optimized gaussians
    if check_optimization and 'optimizers' in checkpoint:
        opt_state = checkpoint['optimizers']['means']['state'][0]
        if 'exp_avg' in opt_state:
            exp_avg = opt_state['exp_avg']
            exp_avg_sq = opt_state['exp_avg_sq']
            unoptimized_mask = (torch.abs(exp_avg).max(dim=-1)[0] < 1e-6) & \
                              (exp_avg_sq.max(dim=-1)[0] < 1e-6)
            inside_mask &= ~unoptimized_mask
            
    # Filter 3: Remove low opacity gaussians
    opacities = checkpoint['pipeline']['_model.gauss_params.opacities']
    opacity_mask = torch.sigmoid(opacities).squeeze(-1) > 0.01
    inside_mask &= opacity_mask
    
    # Apply all filters
    filtered_checkpoint = checkpoint.copy()
    gauss_keys = [
        '_model.gauss_params.means',
        '_model.gauss_params.scales',
        '_model.gauss_params.quats',
        '_model.gauss_params.features_dc',
        '_model.gauss_params.features_rest',
        '_model.gauss_params.opacities'
    ]
    
    for key in gauss_keys:
        filtered_checkpoint['pipeline'][key] = checkpoint['pipeline'][key][inside_mask]
    
    # Filter optimizer states
    if 'optimizers' in checkpoint:
        opt_names = ['means', 'scales', 'quats', 'features_dc', 'features_rest', 'opacities']
        for opt_name in opt_names:
            if opt_name in checkpoint['optimizers']:
                state = checkpoint['optimizers'][opt_name]['state']
                if 0 in state:
                    for key in ['exp_avg', 'exp_avg_sq']:
                        if key in state[0]:
                            filtered_checkpoint['optimizers'][opt_name]['state'][0][key] = \
                                state[0][key][inside_mask]
    
    return filtered_checkpoint

def copy_scene_obj(input_path: Path, chunk_dir: Path):
    try:
        scene_obj = input_path / "scene.obj"
        if scene_obj.exists():
            shutil.copy2(scene_obj, chunk_dir / "scene.obj")
        else:
            raise FileNotFoundError(f"scene.obj not found at {scene_obj}")
    except (OSError, shutil.Error) as e:
        print(f"Error copying scene.obj: {e}")

def split_colmap_dataset(colmap_path: Path, output_base: Path, input_path: Path, chunk_size: int = 500) -> List[Tuple[Path, np.ndarray]]:
    """Split a COLMAP dataset into chunks of specified size.
    Returns list of (chunk_path, chunk_bbox) tuples.
    """
    # Read COLMAP data
    images_txt = colmap_path / "images.txt"
    cameras_txt = colmap_path / "cameras.txt"
    points3D_txt = colmap_path / "points3D.txt"
    
    if not all(p.exists() for p in [images_txt, cameras_txt, points3D_txt]):
        raise FileNotFoundError(f"Missing required COLMAP files in {colmap_path}")

    # Parse points3D first
    points3D = parse_colmap_points3D(points3D_txt)

    # Read and parse images.txt
    with open(images_txt, 'r') as f:
        lines = f.readlines()
    
    # Parse image entries
    image_entries = []
    image_ids = []
    for i, line in enumerate(lines):
        if line.startswith('#') or i % 2 == 1:
            continue
        fields = line.strip().split()
        image_id = int(fields[0])
        image_ids.append(image_id)
        image_entries.append(line.strip())
    
    # Calculate number of chunks
    num_images = len(image_entries)
    num_chunks = math.ceil(num_images / chunk_size)
    chunk_info = []

    print(f"Splitting {num_images} images into {num_chunks} chunks of size {chunk_size}")

    # Create chunks
    for i in range(num_chunks):
        chunk_dir = output_base / f"chunk_{i:03d}/sparse/0"
        create_directory(chunk_dir)
        copy_scene_obj(input_path, chunk_dir)

        # Get chunk image IDs and entries
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_images)
        chunk_image_ids = image_ids[start_idx:end_idx]
        chunk_entries = image_entries[start_idx:end_idx]
                
        # Copy cameras.txt
        shutil.copy2(cameras_txt, chunk_dir / "cameras.txt")
        # shutil.copy2(points3D_txt, chunk_dir / "points3D.txt")
        
        # filter points3D
        filtered_points3D, _ = filter_points3D_for_chunk(points3D, chunk_image_ids)
        write_points3D(filtered_points3D, chunk_dir / "points3D.txt")

        # Write chunk's images.txt
        with open(chunk_dir / "images.txt", 'w') as f:
            for j, image_entry in enumerate(chunk_entries):
                f.write(image_entry + '\n')
                # Find and write the corresponding image data line
                data_line = lines[lines.index(image_entry + '\n') + 1]
                f.write(data_line)
        
        # NEW: Copy corresponding depth and normal files
        chunk_image_dir = chunk_dir.parent.parent / "images"
        create_directory(chunk_image_dir)
        
        # Create depth/normal directories for chunk
        chunk_depth_dir = chunk_dir.parent.parent / "mono_depth"
        chunk_normal_dir = chunk_dir.parent.parent / "normals_from_pretrain"
        create_directory(chunk_depth_dir)
        create_directory(chunk_normal_dir)
        
        # Copy matching files
        for image_entry in chunk_entries:
            image_name = image_entry.split()[-1].split('/')[-1]
            base_name = Path(image_name).stem
            # Copy depth file

            src_depth = input_path.parent / f"{input_path.parent.name}_nerfstudio" / "mono_depth" / f"{base_name}.npy"
            if src_depth.exists():
                shutil.copy2(src_depth, chunk_depth_dir / f"{base_name}.npy")
            
            
            # Copy normal file
            src_normal = input_path.parent / f"{input_path.parent.name}_nerfstudio" / "normals_from_pretrain" / f"{base_name}.png"
            if src_normal.exists():
                shutil.copy2(src_normal, chunk_normal_dir / f"{base_name}.png")
        
        chunk_info.append(chunk_dir)

    return chunk_info

def cull_invisible_gaussians(checkpoint_path: Path, chunk_colmap_path: Path, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Cull gaussians that are not visible in any training view of the chunk."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract gaussian parameters from pipeline
    pipeline_state = checkpoint['pipeline']
    gauss_params = {
        'means': pipeline_state['_model.gauss_params.means'],
        'scales': pipeline_state['_model.gauss_params.scales'],
        'quats': pipeline_state['_model.gauss_params.quats'],
        'features_dc': pipeline_state['_model.gauss_params.features_dc'],
        'features_rest': pipeline_state['_model.gauss_params.features_rest'],
        'opacities': pipeline_state['_model.gauss_params.opacities']
    }
    
    # Extract gaussian parameters
    means = gauss_params['means']
    scales = gauss_params['scales']
    quats = gauss_params['quats']
    features_dc = gauss_params['features_dc']
    features_rest = gauss_params['features_rest']
    opacities = gauss_params['opacities']
    
    # Read COLMAP cameras
    with open(chunk_colmap_path / "images.txt", 'r') as f:
        lines = f.readlines()
    
    # Parse camera poses (we only need camera centers for frustum culling)
    camera_centers = []
    for i, line in enumerate(lines):
        # Skip comments and data lines (every other line contains image data)
        if line.startswith('#') or i % 2 == 1:
            continue
            
        # Split the line and ensure we have enough fields
        fields = line.strip().split()
        if len(fields) < 8:
            print(f"Warning: Skipping malformed line {i}: {line}")
            continue
            
        try:
            # COLMAP format: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            # Position is at index 5,6,7
            tx, ty, tz = map(float, fields[5:8])
            camera_centers.append([tx, ty, tz])
        except (ValueError, IndexError) as e:
            print(f"Warning: Error parsing line {i}: {line}")
            print(f"Error details: {e}")
            continue
    
    if not camera_centers:
        raise RuntimeError(f"No valid camera positions found in {chunk_colmap_path}/images.txt")
    
    print(f"Found {len(camera_centers)} valid camera positions")
    camera_centers = torch.tensor(camera_centers, device=device)
    
    # For each gaussian, check if it's visible in any view
    visible_mask = torch.zeros(len(means), dtype=torch.bool, device=device)
    
    # Parameters for visibility test
    max_distance = 10.0  # Maximum distance from camera to consider gaussian visible
    min_opacity = 0.01  # Minimum opacity to consider gaussian visible
    
    for cam_center in camera_centers:
        # Calculate distances from gaussians to camera
        distances = torch.norm(means - cam_center.unsqueeze(0), dim=1)
        
        # Consider gaussian visible if:
        # 1. It's within max_distance of any camera
        # 2. Its opacity is above threshold
        visible_mask |= (distances < max_distance) & (torch.sigmoid(opacities).squeeze(-1) > min_opacity)
    
    # Keep only visible gaussians
    culled_params = {
        'means': means[visible_mask],
        'scales': scales[visible_mask],
        'quats': quats[visible_mask],
        'features_dc': features_dc[visible_mask],
        'features_rest': features_rest[visible_mask],
        'opacities': opacities[visible_mask]
    }
    
    print(f"Culled {(~visible_mask).sum().item()}/{len(means)} gaussians")
    
    return culled_params

def split_colmap_dataset_by_region(colmap_path: Path, output_base: Path, input_path: Path, m_region: int = 2, n_region: int = 2) -> List[Tuple[Path, np.ndarray]]:
    """Split a COLMAP dataset into m x n regions based on camera positions.
    Returns list of (chunk_path, chunk_bbox) tuples.
    """
    # Read COLMAP data
    images_txt = colmap_path / "images.txt"
    cameras_txt = colmap_path / "cameras.txt"
    points3D_txt = colmap_path / "points3D.txt"
    
    if not all(p.exists() for p in [images_txt, cameras_txt, points3D_txt]):
        raise FileNotFoundError(f"Missing required COLMAP files in {colmap_path}")

    # Parse points3D
    points3D = parse_colmap_points3D(points3D_txt)

    # Read camera positions from images.txt
    camera_positions = []
    image_entries = []
    image_ids = []
    
    with open(images_txt, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if line.startswith('#') or i % 2 == 1:
            continue
        fields = line.strip().split()
        image_id = int(fields[0])
        # Get camera position (last 3 numbers before CAMERA_ID)
        tx, ty, tz = map(float, fields[5:8])
        
        camera_positions.append([tx, tz])  # Only use x,z for 2D partitioning
        image_ids.append(image_id)
        image_entries.append(line.strip())

    camera_positions = np.array(camera_positions)
    
    # Get region boundaries
    x_min, x_max = camera_positions[:, 0].min(), camera_positions[:, 0].max()
    z_min, z_max = camera_positions[:, 1].min(), camera_positions[:, 1].max()
    
    x_step = (x_max - x_min) / m_region
    z_step = (z_max - z_min) / n_region
    
    chunk_info = []
    
    # Create m x n regions
    for i in range(m_region):
        for j in range(n_region):
            region_x_min = x_min + i * x_step
            region_x_max = x_min + (i + 1) * x_step
            region_z_min = z_min + j * z_step 
            region_z_max = z_min + (j + 1) * z_step
            
            # Find cameras in this region
            mask = ((camera_positions[:, 0] >= region_x_min) & 
                   (camera_positions[:, 0] < region_x_max) &
                   (camera_positions[:, 1] >= region_z_min) &
                   (camera_positions[:, 1] < region_z_max))
            
            region_image_ids = [id for id, m in zip(image_ids, mask) if m]
            region_entries = [entry for entry, m in zip(image_entries, mask) if m]
            
            if not region_image_ids:
                print(f"Warning: No cameras in region {i}_{j}")
                continue
                
            # Create chunk directory and copy scene.obj
            chunk_dir = output_base / f"chunk_{i}_{j}/sparse/0"
            create_directory(chunk_dir)
            copy_scene_obj(input_path, chunk_dir)
            
            # Filter points3D for this region
            region_points3D, region_bbox = filter_points3D_for_chunk(points3D, region_image_ids)
            
            # Copy cameras.txt
            shutil.copy2(cameras_txt, chunk_dir / "cameras.txt")
            
            # Write region's images.txt
            with open(chunk_dir / "images.txt", 'w') as f:
                for k, image_entry in enumerate(region_entries):
                    if k > 0:
                        f.write('\n')
                    f.write(image_entry + '\n')
                    # Find and write corresponding data line
                    data_line = lines[lines.index(image_entry + '\n') + 1]
                    f.write(data_line)
                    
            # Write filtered points3D.txt
            write_points3D(region_points3D, chunk_dir / "points3D.txt")
            
            # NEW: Copy corresponding depth and normal files
            chunk_image_dir = chunk_dir.parent.parent / "images"
            create_directory(chunk_image_dir)
            
            # Create depth/normal directories for chunk
            chunk_depth_dir = chunk_dir.parent.parent / "mono_depth"
            chunk_normal_dir = chunk_dir.parent.parent / "normals_from_pretrain"
            create_directory(chunk_depth_dir)
            create_directory(chunk_normal_dir)
            
            # Copy matching files
            for image_entry in region_entries:
                image_name = image_entry.split()[-1].split('/')[-1]
                base_name = Path(image_name).stem
                
                # Copy depth file
                src_depth = input_path / "mono_depth" / f"{base_name}.npy"
                if src_depth.exists():
                    shutil.copy2(src_depth, chunk_depth_dir / f"{base_name}.npy")
                
                # Copy normal file
                src_normal = input_path / "normals_from_pretrain" / f"{base_name}.png"
                if src_normal.exists():
                    shutil.copy2(src_normal, chunk_normal_dir / f"{base_name}.png")
            
            chunk_info.append((chunk_dir, region_bbox))
            print(f"Region {i}_{j}: {len(region_image_ids)} images, {len(region_points3D)} points")
            print(f"Region bbox: min={region_bbox[0]}, max={region_bbox[1]}")
            
    return chunk_info

def split_colmap_dataset_by_cluster(colmap_path: Path, output_base: Path, input_path: Path, n_clusters: int = 4) -> List[Tuple[Path, np.ndarray]]:
    """Split a COLMAP dataset into chunks using K-means clustering on camera positions.
    Returns list of (chunk_path, chunk_bbox) tuples.
    """
    # Read COLMAP data
    images_txt = colmap_path / "images.txt"
    cameras_txt = colmap_path / "cameras.txt"
    points3D_txt = colmap_path / "points3D.txt"
    
    if not all(p.exists() for p in [images_txt, cameras_txt, points3D_txt]):
        raise FileNotFoundError(f"Missing required COLMAP files in {colmap_path}")

    # Parse points3D
    points3D = parse_colmap_points3D(points3D_txt)

    # Read camera positions from images.txt
    camera_positions = []
    image_entries = []
    image_ids = []
    
    with open(images_txt, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if line.startswith('#') or i % 2 == 1:
            continue
        fields = line.strip().split()
        image_id = int(fields[0])
        # Get camera position (last 3 numbers before CAMERA_ID)
        tx, ty, tz = map(float, fields[5:8])
        
        camera_positions.append([tx, tz])  # Only use x,z for 2D clustering
        image_ids.append(image_id)
        image_entries.append(line.strip())

    camera_positions = np.array(camera_positions)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(camera_positions)
    
    chunk_info = []
    
    # Create chunks based on clusters
    for cluster_idx in range(n_clusters):
        # Get cameras in this cluster
        cluster_mask = (cluster_labels == cluster_idx)
        cluster_image_ids = [id for id, m in zip(image_ids, cluster_mask) if m]
        cluster_entries = [entry for entry, m in zip(image_entries, cluster_mask) if m]
        
        if not cluster_image_ids:
            print(f"Warning: No cameras in cluster {cluster_idx}")
            continue
            
        # Create chunk directory and copy scene.obj
        chunk_dir = output_base / f"chunk_cluster_{cluster_idx}/sparse/0"
        create_directory(chunk_dir)
        copy_scene_obj(input_path, chunk_dir)
        
        # Filter points3D for this cluster
        cluster_points3D, cluster_bbox = filter_points3D_for_chunk(points3D, cluster_image_ids)
        
        # Copy cameras.txt
        shutil.copy2(cameras_txt, chunk_dir / "cameras.txt")
        
        # Write cluster's images.txt
        with open(chunk_dir / "images.txt", 'w') as f:
            for k, image_entry in enumerate(cluster_entries):
                if k > 0:
                    f.write('\n')
                f.write(image_entry + '\n')
                # Find and write corresponding data line
                data_line = lines[lines.index(image_entry + '\n') + 1]
                f.write(data_line)
                
        # Write filtered points3D.txt
        write_points3D(cluster_points3D, chunk_dir / "points3D.txt")
        
        # NEW: Copy corresponding depth and normal files
        chunk_image_dir = chunk_dir.parent.parent / "images"
        create_directory(chunk_image_dir)
        
        # Create depth/normal directories for chunk
        chunk_depth_dir = chunk_dir.parent.parent / "mono_depth"
        chunk_normal_dir = chunk_dir.parent.parent / "normals_from_pretrain"
        create_directory(chunk_depth_dir)
        create_directory(chunk_normal_dir)
        
        # Copy matching files
        for image_entry in cluster_entries:
            image_name = image_entry.split()[-1].split('/')[-1]
            base_name = Path(image_name).stem
            
            # Copy depth file
            src_depth = input_path / "mono_depth" / f"{base_name}.npy"
            if src_depth.exists():
                shutil.copy2(src_depth, chunk_depth_dir / f"{base_name}.npy")
            
            # Copy normal file
            src_normal = input_path / "normals_from_pretrain" / f"{base_name}.png"
            if src_normal.exists():
                shutil.copy2(src_normal, chunk_normal_dir / f"{base_name}.png")
        
        chunk_info.append((chunk_dir, cluster_bbox))
        print(f"Cluster {cluster_idx}: {len(cluster_image_ids)} images, {len(cluster_points3D)} points")
        print(f"Cluster bbox: min={cluster_bbox[0]}, max={cluster_bbox[1]}")
        
        # Optionally visualize the cluster
        cluster_positions = camera_positions[cluster_mask]
        plt.scatter(cluster_positions[:, 0], cluster_positions[:, 1], label=f'Cluster {cluster_idx}')
    
    plt.title('Camera Positions Clustering')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.legend()
    plt.savefig(output_base / 'camera_clusters.png')
    plt.close()
        
    return chunk_info

def get_training_command(config: Union[ChunkTrainingConfig, FinalTrainingConfig], 
                        data_path: Path,
                        output_dir: Optional[Path] = None,
                        colmap_path: Optional[str] = None,
                        is_chunk: bool = False) -> str:
    """
    Generate training command based on configuration.
    
    Args:
        config: Training configuration
        data_path: Path to dataset
        output_dir: Output directory (optional)
        colmap_path: Path to COLMAP data (optional)
        is_chunk: Whether this is for chunk training
    
    Returns:
        Command string for training
    """
    cmds = ['ns-train splatfacto']
    
    # Add data path
    cmds.append(f'--data {data_path}')
    
    # Add output directory if specified
    if output_dir:
        cmds.append(f'--output-dir {output_dir}')
    
    # Add iterations
    cmds.append(f'--max-num-iterations {config.max_iterations}')
    
    # Add load directory for final training
    if isinstance(config, FinalTrainingConfig) and config.load_dir:
        cmds.append(f'--load-dir {config.load_dir}')
    
    # Add background color if specified
    if config.background_color:
        cmds.append('--pipeline.model.background-color custom')
        cmds.append(f'--pipeline.model.custom-background-color {config.background_color}')
    
    # Add model configuration
    cmds.append(f'--pipeline.model.num-downscales {config.num_downscales}')
    if config.use_mesh_initialization:
        cmds.append('--pipeline.model.use-mesh-initialization True')
    if config.combine_mesh_sfm:
        cmds.append('--pipeline.model.combine-mesh-sfm True')
    if config.rasterize_mode:
        cmds.append(f'--pipeline.model.rasterize-mode {config.rasterize_mode}')
    if config.use_scale_regularization:
        cmds.append('--pipeline.model.use-scale-regularization True')
    if config.camera_optimizer_mode:
        cmds.append(f'--pipeline.model.camera-optimizer.mode {config.camera_optimizer_mode}')
    if config.train_cameras_sampling:
        cmds.append(f'--pipeline.datamanager.train-cameras-sampling-strategy {config.train_cameras_sampling}')
    if config.sh_degree:
        cmds.append(f'--pipeline.model.sh-degree {config.sh_degree}')
    if config.use_bilateral_grid:
        cmds.append('--pipeline.model.use-bilateral-grid True')
    if config.num_downscales:
        cmds.append(f'--pipeline.model.num-downscales {config.num_downscales}')
    
    # Add chunk-specific parameters
    if is_chunk and isinstance(config, ChunkTrainingConfig):
        if config.stop_split_at:
            cmds.append(f'--pipeline.model.stop_split_at {config.stop_split_at}')
    
    # Add visualization settings
    if config.make_share_url:
        cmds.append('--viewer.make-share-url True')
    if config.visualization:
        cmds.append(f'--vis {config.visualization}')
    if config.quit_on_completion:
        cmds.append('--viewer.quit-on-train-completion True')
    
    # Add COLMAP settings
    cmds.append('colmap')
    if colmap_path:
        cmds.append(f'--colmap_path "{colmap_path}"')
    cmds.append(f'--auto_scale_poses {"True" if config.auto_scale_poses else "False"}')
    cmds.append(f'--center_method {config.center_method}')
    cmds.append(f'--orientation_method {config.orientation_method}')
    cmds.append(f'--eval-mode {config.eval_mode}')
    cmds.append(f'--train-split-fraction {config.train_split_fraction}')
    
    return ' '.join(cmds)

def main(args):
    root_path = Path(args.input_path).resolve()
    parent_name = root_path.parent.name
    output_root = root_path.parent / f"{parent_name}_nerfstudio"
    pose_method = args.method
    if args.use_icp:
        pose_method = f"{pose_method}_ICP"

    # Add background color detection
    images_dir = output_root / "images"
    try:
        bg_color = get_background_color(str(images_dir))
        bg_color_str = f"{bg_color[0]},{bg_color[1]},{bg_color[2]}"
        print(f"Detected background color: {bg_color_str}")
    except Exception as e:
        print(f"Warning: Could not detect background color: {e}")
        bg_color_str = None

    # Create chunks directory
    chunks_dir = output_root / "chunks"
    create_directory(chunks_dir)
    
    # Split dataset into chunks based on partition method
    if not args.skip_chunk_training:
        if args.partition_method == 'region':
            chunk_info = split_colmap_dataset_by_region(
                output_root / f"colmap/{pose_method}/0", 
                chunks_dir,
                root_path,
                args.m_region,
                args.n_region
            )
        elif args.partition_method == 'cluster':
            chunk_info = split_colmap_dataset_by_cluster(
                output_root / f"colmap/{pose_method}/0",
                chunks_dir,
                root_path,
                args.n_clusters
            )
        else:  # 'chunk_size'
            chunk_info = split_colmap_dataset(
                output_root / f"colmap/{pose_method}/0", 
                chunks_dir,
                root_path,
                args.chunk_size
            )
    else:
        # If skipping chunk training, find existing chunks
        chunk_info = []
        for chunk_dir in chunks_dir.glob("chunk_*/sparse/0"):
            # Read bbox from a saved file or reconstruct it
            points3D_txt = chunk_dir / "points3D.txt"
            if points3D_txt.exists():
                points3D = parse_colmap_points3D(points3D_txt)
                xyz_points = np.stack([xyz for xyz, _ in points3D.values()], axis=0)
                min_xyz = np.min(xyz_points, axis=0)
                max_xyz = np.max(xyz_points, axis=0)
                # Add padding
                size = max_xyz - min_xyz
                padding = size * 0.05
                min_xyz -= padding
                max_xyz += padding
                bbox = np.stack([min_xyz, max_xyz], axis=0)
                chunk_info.append(chunk_dir)
        
        if not chunk_info:
            raise RuntimeError("No existing chunks found in directory")
        print(f"Found {len(chunk_info)} existing chunks")

    # Create chunk training config
    chunk_config = ChunkTrainingConfig(
        max_iterations=args.chunk_iterations,
        stop_split_at=args.chunk_iterations,
        background_color=bg_color_str if 'bg_color_str' in locals() else None
    )

    # Train each chunk or load existing checkpoints
    chunk_checkpoints = []
    for chunk_path in chunk_info:
        chunk_name = chunk_path.parent.parent.name
        output_dir = output_root / 'chunks' / f"{chunk_name}" / 'output'
        model_dir = output_dir / output_root.name / 'splatfacto'

        if not args.skip_chunk_training:
            # Get training command for chunk
            cmd = get_training_command(
                config=chunk_config,
                data_path=output_root,
                output_dir=output_dir,
                colmap_path=f"chunks/{chunk_name}/sparse/0",
                is_chunk=True
            )
            print(f"Training chunk {chunk_name} with command: {cmd}")
            os.system(cmd)

        # Find latest checkpoint
        timestamp_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        if not timestamp_dirs:
            raise RuntimeError(f"No timestamp directories found in {model_dir}")
        
        latest_timestamp_dir = max(timestamp_dirs, key=lambda x: x.stat().st_mtime)
        checkpoint_dir = latest_timestamp_dir / "nerfstudio_models"
        
        checkpoints = list(checkpoint_dir.glob("step-*.ckpt"))
        if not checkpoints:
            raise RuntimeError(f"No checkpoint found for chunk {chunk_name}")
        latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('-')[1]))
        
        # Load and optionally filter checkpoint
        checkpoint = torch.load(latest_checkpoint, map_location="cuda")
        if args.filter_gaussians:
            filtered_checkpoint = filter_gaussians_by_bbox(checkpoint)
            chunk_checkpoints.append(filtered_checkpoint)
        else:
            chunk_checkpoints.append(checkpoint)

    # Create combined checkpoint directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    combined_dir = output_root / parent_name / 'splatfacto' / timestamp
    create_directory(combined_dir / "nerfstudio_models")
    
    # Instead of directly calling combine_checkpoints, use the CLI command
    combine_cmd = [
        "python", 
        str(Path(__file__).parent / "combine_chunk_checkpoints.py"),
        "--chunks_dir", str(chunks_dir),
        "--output_dir", str(combined_dir),
        "--device", "cuda"
    ]
    
    if args.debug:
        combine_cmd.append("--debug")
    
    print("\nRunning combine_chunk_checkpoints with command:", " ".join(combine_cmd))
    subprocess.run(combine_cmd, check=True)

    # Create final training config
    final_config = FinalTrainingConfig(
        max_iterations=args.final_iterations,
        load_dir=str(combined_dir / "nerfstudio_models"),
        background_color=bg_color_str if 'bg_color_str' in locals() else None
    )

    # Get final training command
    final_cmd = get_training_command(
        config=final_config,
        data_path=output_root,
        output_dir=str(output_root),
        colmap_path=f"colmap/{pose_method}/0",
        is_chunk=False
    )

    print("Running final training with command:", final_cmd)
    subprocess.run(final_cmd, shell=True, check=True)

    # Clean up intermediate checkpoint after final training is complete
    try:
        shutil.rmtree(combined_dir)
        print(f"\nCleaned up intermediate checkpoint directory: {combined_dir}")
    except Exception as e:
        print(f"\nWarning: Failed to clean up intermediate checkpoint: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train nerfstudio model on chunked dataset")
    parser.add_argument("--input_path", help="Path to the root directory of run_arkit_3dgs.sh output")
    parser.add_argument("--method", type=str, default='arkit', help="Choose pose optimization methods")
    parser.add_argument("--use_icp", action='store_true', default=False, help="use ICP for mesh and point3D")
    parser.add_argument("--model_type", type=str, default='splatfacto', help="Choose model type")
    parser.add_argument("--chunk_size", type=int, default=500, help="Number of images per chunk")
    parser.add_argument("--chunk_iterations", type=int, default=10000, help="Number of iterations for training each chunk")
    parser.add_argument("--final_iterations", type=int, default=30000, help="Number of iterations for final training")
    parser.add_argument("--partition_method", type=str, default='chunk_size',
                      choices=['chunk_size', 'region', 'cluster'],
                      help="Method to partition the dataset")
    parser.add_argument("--m_region", type=int, default=2,
                      help="Number of regions in x direction")
    parser.add_argument("--n_region", type=int, default=2, 
                      help="Number of regions in z direction")
    parser.add_argument("--n_clusters", type=int, default=4,
                      help="Number of clusters for clustering-based partition")
    parser.add_argument("--skip-chunk-training", action="store_true",
                      help="Skip chunk training and use existing checkpoints")
    parser.add_argument("--filter-gaussians", action="store_true", default=True,
                      help="Whether to filter gaussians when combining chunks")
    parser.add_argument("--debug", action="store_true", default=False,
                      help="Enable debug mode for combine_chunk_checkpoints")
    args = parser.parse_args()
    print("args", args)
    main(args) 