import sqlite3
import cv2
import numpy as np
import os
from pathlib import Path
import argparse
import logging
import colorlog
from tabulate import tabulate
import json
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import struct
import shutil

# Set up colored logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)s:%(name)s:%(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))

logger = colorlog.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def get_image_pairs(db_path):
    """獲取所有有效的影像配對"""
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT pair_id, data FROM matches WHERE rows > 0")
    
    # 解析 pair_id 為圖片 ID
    pairs = []
    for pair_id, _ in cursor:
        # COLMAP 使用以下公式編碼 pair_id
        # pair_id = image_id1 * MAX_IMAGE_ID + image_id2
        image_id2 = pair_id % 2147483647
        image_id1 = (pair_id - image_id2) // 2147483647
        pairs.append((image_id1, image_id2))
    
    # 獲取圖片路徑
    images = {}
    cursor = conn.execute("SELECT image_id, name FROM images")
    for image_id, name in cursor:
        images[image_id] = name
    
    conn.close()
    return pairs, images

def draw_matches(img1_path, img2_path, matches_data, output_path, inlier_mask=None):
    """繪製兩張圖片之間的匹配關係，內點和外點用不同顏色表示"""
    # 讀取圖片
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    # 調整圖片大小以便顯示
    max_height = 800
    scale1 = max_height / img1.shape[0]
    scale2 = max_height / img2.shape[0]
    
    img1 = cv2.resize(img1, None, fx=scale1, fy=scale1)
    img2 = cv2.resize(img2, None, fx=scale2, fy=scale2)
    
    # 水平拼接圖片
    combined = np.hstack((img1, img2))
    
    # 在拼接圖上畫線
    for idx, match in enumerate(matches_data):
        pt1 = (int(match[0] * scale1), int(match[1] * scale1))
        pt2 = (int(match[2] * scale2) + img1.shape[1], int(match[3] * scale2))
        
        # 如果有內點/外點標記，使用不同顏色
        if inlier_mask is not None:
            color = (0, 255, 0) if inlier_mask[idx] else (0, 0, 255)  # 綠色為內點，紅色為外點
        else:
            color = (0, 255, 0)
            
        cv2.line(combined, pt1, pt2, color, 1)
    
    # Add inlier/outlier count annotation
    if inlier_mask is not None:
        n_inliers = np.sum(inlier_mask)
        n_outliers = len(inlier_mask) - n_inliers
        text = f"Inliers: {n_inliers}, Outliers: {n_outliers}"
        cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2)  # White text with black outline
        cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 0), 1)
    
    # 保存結果
    cv2.imwrite(str(output_path), combined)

def find_outliers(matches_coords, threshold=2.0):
    """使用RANSAC和基礎矩陣估計來找出外點"""
    # 將匹配點轉換為numpy數組
    pts1 = np.float32([[match[0], match[1]] for match in matches_coords])
    pts2 = np.float32([[match[2], match[3]] for match in matches_coords])
    
    # 使用RANSAC估計基礎矩陣
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 
                                    ransacReprojThreshold=threshold, 
                                    confidence=0.999)
    
    return mask.ravel().astype(bool)

def compute_epipolar_error(F, pts1, pts2):
    """計算極線誤差"""
    if F is None or F.shape != (3, 3):
        return np.full(len(pts1), np.inf)  # 返回無限大的誤差
        
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    
    # 計算點到極線的距離
    errors = []
    for pt, line in zip(pts2, lines2):
        # 點到線距離公式: |ax + by + c| / sqrt(a^2 + b^2)
        error = abs(line[0] * pt[0] + line[1] * pt[1] + line[2]) / np.sqrt(line[0]**2 + line[1]**2)
        errors.append(error)
    return np.array(errors)

def draw_matches_with_stats(img1_path, img2_path, matches_data, output_path):
    """Draw matches between two images with basic statistics"""
    # Read images
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    # Prepare match points
    pts1 = np.float32([[match[0], match[1]] for match in matches_data])
    pts2 = np.float32([[match[2], match[3]] for match in matches_data])
    
    # Resize images for display
    max_height = 800
    scale1 = max_height / img1.shape[0]
    scale2 = max_height / img2.shape[0]
    img1 = cv2.resize(img1, None, fx=scale1, fy=scale1)
    img2 = cv2.resize(img2, None, fx=scale2, fy=scale2)
    
    # Create combined image
    combined = np.hstack((img1, img2))
    
    # Draw match lines
    color = (0, 255, 0)  # Green color for all matches
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        pt1 = (int(x1 * scale1), int(y1 * scale1))
        pt2 = (int(x2 * scale2) + img1.shape[1], int(y2 * scale2))
        cv2.line(combined, pt1, pt2, color, 1)
    
    # Add statistics
    info_text = [
        f"Total matches: {len(pts1)}",
    ]
    
    y_offset = 30
    for i, text in enumerate(info_text):
        y = y_offset + i * 30
        # White text with black outline for better visibility
        cv2.putText(combined, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2)
        cv2.putText(combined, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 0), 1)
    
    # Save result
    cv2.imwrite(str(output_path), combined)
    
    return {
        'total_matches': len(pts1),
    }

def write_filtered_images_txt(original_txt_path, output_txt_path, image_ids_to_keep):
    """Write a new images.txt containing only the specified image IDs"""
    with open(original_txt_path, 'r') as f_in, open(output_txt_path, 'w') as f_out:
        # Copy header comments
        line = f_in.readline()
        while line.startswith('#'):
            f_out.write(line)
            line = f_in.readline()
        
        # Go back to start after headers
        f_in.seek(0)
        
        # Skip header comments again
        line = f_in.readline()
        while line.startswith('#'):
            line = f_in.readline()
        
        # Process each image entry (2 lines per image)
        while line:
            if line.strip():  # Skip empty lines
                data = line.strip().split()
                image_id = int(data[0])
                
                if image_id in image_ids_to_keep:
                    # Write the image line
                    f_out.write(line)
                    # Write the points line
                    points_line = f_in.readline()
                    f_out.write(points_line)
                else:
                    # Skip the points line
                    f_in.readline()
            
            line = f_in.readline()

def write_images_txt(images_txt_path, output_txt_path, image_ids_to_keep):
    """Convert images.txt to a new images.txt containing only specified image IDs"""
    images = []
    
    with open(images_txt_path, 'r') as f:
        # Skip header comments
        line = f.readline()
        while line.startswith('#'):
            line = f.readline()
        
        # Process each image entry (2 lines per image)
        while line:
            if line.strip():  # Skip empty lines
                data = line.strip().split()
                image_id = int(data[0])
                
                if image_id in image_ids_to_keep:
                    # Parse image data
                    qvec = [float(x) for x in data[1:5]]  # [qw, qx, qy, qz]
                    tvec = [float(x) for x in data[5:8]]  # [tx, ty, tz]
                    camera_id = int(data[8])
                    name = data[9]
                    
                    # Read points line
                    points_line = f.readline().strip()
                    
                    images.append({
                        'id': image_id,
                        'qvec': qvec,
                        'tvec': tvec,
                        'camera_id': camera_id,
                        'name': name,
                        'points_line': points_line
                    })
                else:
                    # Skip points line
                    f.readline()
            
            line = f.readline()
    
    # Write text file
    with open(output_txt_path, 'w') as f:
        # Write header
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        
        # Write each image
        for image in images:
            # Write image line
            f.write(f"{image['id']} {' '.join(map(str, image['qvec']))} "
                   f"{' '.join(map(str, image['tvec']))} {image['camera_id']} "
                   f"{image['name']}\n")
            
            # Write points line
            f.write(f"{image['points_line']}\n")

def copy_cameras_txt(src_dir, dst_dir, image_ids_to_keep):
    """Copy and filter cameras.txt to only include cameras used by kept images"""
    src_path = Path(src_dir) / 'cameras.txt'
    images_path = Path(src_dir) / 'images.txt'
    dst_path = Path(dst_dir) / 'cameras.txt'
    
    if not src_path.exists() or not images_path.exists():
        return
    
    # First get the camera IDs used by kept images
    used_camera_ids = set()
    with open(images_path, 'r') as f:
        # Skip header comments
        line = f.readline()
        while line.startswith('#'):
            line = f.readline()
        
        # Process each image entry (2 lines per image)
        while line:
            if line.strip():  # Skip empty lines
                data = line.strip().split()
                image_id = int(data[0])
                if image_id in image_ids_to_keep:
                    camera_id = int(data[8])  # Camera ID is the 9th field
                    used_camera_ids.add(camera_id)
                # Skip points line
                f.readline()
            line = f.readline()
    
    # Now copy only the used cameras
    with open(src_path, 'r') as f_in, open(dst_path, 'w') as f_out:
        # Copy header comments
        line = f_in.readline()
        while line.startswith('#'):
            f_out.write(line)
            line = f_in.readline()
        
        # Process each camera
        while line:
            if line.strip():  # Skip empty lines
                data = line.strip().split()
                camera_id = int(data[0])
                if camera_id in used_camera_ids:
                    f_out.write(line)
            line = f_in.readline()
    
    logger.info(f"Wrote filtered cameras.txt with {len(used_camera_ids)} cameras")

def copy_points3D_txt(src_dir, dst_dir, image_ids_to_keep):
    """Copy and filter points3D.txt to only include points visible in kept images"""
    src_path = Path(src_dir) / 'points3D.txt'
    dst_path = Path(dst_dir) / 'points3D.txt'
    
    if not src_path.exists():
        return
    
    kept_points = set()
    points_data = {}
    
    # First pass: identify points visible in kept images
    with open(src_path, 'r') as f:
        # Skip header comments
        line = f.readline()
        while line.startswith('#'):
            line = f.readline()
        
        # Process each point
        while line:
            if line.strip():
                parts = line.strip().split()
                point3D_id = int(parts[0])
                xyz = list(map(float, parts[1:4]))
                rgb = list(map(int, parts[4:7]))
                error = float(parts[7])
                image_ids_str = parts[8::3]  # Every third element starting from index 8
                point_image_ids = set(map(int, image_ids_str))
                
                # If point is visible in any kept image, save it
                if point_image_ids & image_ids_to_keep:
                    kept_points.add(point3D_id)
                    points_data[point3D_id] = {
                        'xyz': xyz,
                        'rgb': rgb,
                        'error': error,
                        'track': [(int(parts[i]), int(parts[i+1])) 
                                 for i in range(8, len(parts), 3)]
                    }
            line = f.readline()
    
    # Write filtered points3D.txt
    with open(dst_path, 'w') as f:
        # Write header
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        
        # Write points
        for point3D_id in sorted(kept_points):
            point = points_data[point3D_id]
            track_str = ' '.join(f'{image_id} {point2D_idx} 1' 
                               for image_id, point2D_idx in point['track'])
            f.write(f"{point3D_id} {' '.join(map(str, point['xyz']))} "
                   f"{' '.join(map(str, point['rgb']))} {point['error']} {track_str}\n")
    
    logger.info(f"Wrote filtered points3D.txt with {len(kept_points)} points")

def create_filtered_model(sparse_dir, output_dir, image_ids_to_keep):
    """Create a filtered copy of the sparse model"""
    sparse_dir = Path(sparse_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Copy and filter each file
    copy_cameras_txt(sparse_dir, output_dir, image_ids_to_keep)
    write_filtered_images_txt(sparse_dir / 'images.txt', 
                            output_dir / 'images.txt',
                            image_ids_to_keep)
    copy_points3D_txt(sparse_dir, output_dir, image_ids_to_keep)
    
    logger.info(f"Created filtered model in {output_dir}")

def visualize_all_pairs(db_path, images_dir, output_dir, sparse_dir=None, query_image_ids=None, 
                       original_images_txt=None, skip_vis=False, skip_query_pairs=False):
    """視覺化影像配對和相對位姿"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 獲取所有配對
    pairs, images = get_image_pairs(db_path)
    
    # Create sets for query and matched images
    query_image_ids_set = set(query_image_ids) if query_image_ids else set()
    print("query_image_ids_set", query_image_ids_set)
    matched_image_ids = set()
    
    # Filter pairs to only include those with query images
    filtered_pairs = []
    if query_image_ids:
        for id1, id2 in pairs:
            # Skip pairs where both images are query images if skip_query_pairs is True
            if skip_query_pairs and id1 in query_image_ids_set and id2 in query_image_ids_set:
                continue
                
            if id1 in query_image_ids_set or id2 in query_image_ids_set:
                filtered_pairs.append((id1, id2))
                if id1 in query_image_ids_set:
                    matched_image_ids.add(id2)
                if id2 in query_image_ids_set:
                    matched_image_ids.add(id1)
    else:
        filtered_pairs = pairs
    
    # Remove query images from matched set
    matched_image_ids -= query_image_ids_set
    print("matched_image_ids", matched_image_ids)
    
    # If original_images_txt is provided, create filtered versions
    if original_images_txt:
        # Create filtered text files
        filtered_txt_path = output_dir / 'filtered_images.txt'
        write_filtered_images_txt(original_images_txt, filtered_txt_path, 
                                matched_image_ids | query_image_ids_set)
        
        # Create text files for query and matched images
        if query_image_ids:
            write_images_txt(original_images_txt, 
                           output_dir / 'query_images.txt', 
                           query_image_ids_set)
            write_images_txt(original_images_txt, 
                           output_dir / 'matched_images.txt', 
                           matched_image_ids)
            
            logger.info(f"Created text files:")
            logger.info(f"  - query_images.txt: {len(query_image_ids_set)} images")
            logger.info(f"  - matched_images.txt: {len(matched_image_ids)} images")
    
    conn = sqlite3.connect(db_path)
    
    stats = []
    for i, (id1, id2) in enumerate(filtered_pairs):
        # Get image paths
        img1_path = Path(images_dir) / images[id1]
        img2_path = Path(images_dir) / images[id2]
        
        # Get match data
        pair_id = id1 * 2147483647 + id2
        cursor = conn.execute("SELECT data FROM two_view_geometries WHERE pair_id=?", (pair_id,))
        matches_data = cursor.fetchone()[0]
        matches = np.frombuffer(matches_data, dtype=np.uint32).reshape(-1, 2)
        
        # Get keypoints
        cursor = conn.execute("SELECT data FROM keypoints WHERE image_id=?", (id1,))
        keypoints1_data = cursor.fetchone()[0]
        keypoints1 = np.frombuffer(keypoints1_data, dtype=np.float32).reshape(-1, 2)
        
        cursor = conn.execute("SELECT data FROM keypoints WHERE image_id=?", (id2,))
        keypoints2_data = cursor.fetchone()[0]
        keypoints2 = np.frombuffer(keypoints2_data, dtype=np.float32).reshape(-1, 2)
        
        # Prepare match coordinates
        matches_coords = []
        for idx1, idx2 in matches:
            kp1 = keypoints1[idx1]
            kp2 = keypoints2[idx2]
            matches_coords.append([kp1[0], kp1[1], kp2[0], kp2[1]])
        
        # Only save visualization if not skipped
        if not skip_vis:
            result_stats = draw_matches_with_stats(
                img1_path, img2_path, matches_coords, 
                output_dir / f"match_{id1}_{id2}.jpg"
            )
        else:
            result_stats = {'total_matches': len(matches_coords)}
        
        stats.append({
            'image_pair': f"{images[id1]} - {images[id2]}",
            **result_stats
        })
        
        print(f"Processed pair {i+1}/{len(filtered_pairs)}: {images[id1]} - {images[id2]}")
    
    # Print statistics table
    print("\nMatching Statistics:")
    headers = ['Image Pair', 'Total Matches']
    table = [[s['image_pair'], s['total_matches']] for s in stats]
    print(tabulate(table, headers=headers, tablefmt='grid'))
    
    # Create filtered sparse model if sparse_dir is provided
    if sparse_dir:
        filtered_model_dir = output_dir / 'filtered_sparse'
        create_filtered_model(sparse_dir, filtered_model_dir, 
                            matched_image_ids | query_image_ids_set)
    
    conn.close()

def parse_query_range(query_str):
    """Parse query range from string like '1009-1169' into a list of integers"""
    if not query_str:
        return None
    
    # Handle individual numbers
    if query_str.isdigit():
        return [int(query_str)]
    
    # Handle ranges with hyphen
    if '-' in query_str:
        start, end = map(int, query_str.split('-'))
        return list(range(start, end + 1))
    
    return None

def read_image_ids_from_txt(images_txt_path):
    """Read image IDs from a COLMAP images.txt file"""
    image_ids = set()
    
    with open(images_txt_path, 'r') as f:
        # Skip header comments
        line = f.readline()
        while line.startswith('#'):
            line = f.readline()
        
        # Process each image entry (2 lines per image)
        while line:
            if line.strip():  # Skip empty lines
                data = line.strip().split()
                image_id = int(data[0])
                image_ids.add(image_id)
                # Skip points line
                f.readline()
            line = f.readline()
    
    return image_ids

def main():
    # 設置命令行參數
    parser = argparse.ArgumentParser(description='視覺化 COLMAP 資料庫中的影像配對')
    parser.add_argument('--database', '-d', required=True,
                        help='COLMAP 資料庫路徑 (database.db)')
    parser.add_argument('--images', '-i', required=True,
                        help='原始圖片目錄路徑')
    parser.add_argument('--output', '-o', required=True,
                        help='輸出結果目錄路徑')
    parser.add_argument('--query', '-q', type=str, nargs='+',
                        help='指定要視覺化的圖片索引 (支援範圍表示法，例如: 1009-1169)')
    parser.add_argument('--query-images-txt', type=str,
                        help='Path to images.txt file containing query images')
    parser.add_argument('--sparse', '-s', type=str,
                        help='COLMAP稀疏重建模型目錄路徑')
    parser.add_argument('--original-images-txt', type=str,
                        help='Original COLMAP images.txt file path to create filtered version')
    parser.add_argument('--skip-vis', action='store_true',
                        help='Skip saving visualization images')
    parser.add_argument('--skip-query-pairs', action='store_true',
                        help='Skip pairs where both images are query images')

    args = parser.parse_args()

    # Get query IDs from either command line arguments or images.txt
    query_ids = []
    if args.query:
        for q in args.query:
            ids = parse_query_range(q)
            if ids:
                query_ids.extend(ids)
    elif args.query_images_txt:
        query_ids = list(read_image_ids_from_txt(args.query_images_txt))
        logger.info(f"Read {len(query_ids)} query image IDs from {args.query_images_txt}")
    
    visualize_all_pairs(
        args.database, 
        args.images, 
        args.output, 
        args.sparse, 
        query_ids if query_ids else None,
        args.original_images_txt,
        args.skip_vis,
        args.skip_query_pairs
    )

if __name__ == "__main__":
    main()