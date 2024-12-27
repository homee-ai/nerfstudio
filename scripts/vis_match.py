import sqlite3
import cv2
import numpy as np
import os
from pathlib import Path
import argparse
import logging
import colorlog
from tabulate import tabulate

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

def draw_matches_with_F_comparison(img1_path, img2_path, matches_data, output_path):
    """繪製匹配並比較RANSAC前後的F矩陣效果"""
    # 讀取圖片
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    # 準備匹配點
    pts1 = np.float32([[match[0], match[1]] for match in matches_data])
    pts2 = np.float32([[match[2], match[3]] for match in matches_data])
    
    # 檢查是否有足夠的點來計算F矩陣（至少需要8個點）
    if len(pts1) < 8:
        logger.warning(f"Not enough matches ({len(pts1)}) to compute F matrix (minimum 8 required)")
        return {
            'total_matches': len(pts1),
            'ransac_inliers': 0,
            'ransac_outliers': 0,
            'mean_error_no_ransac': float('inf'),
            'mean_error_ransac': float('inf')
        }
    
    # 計算F矩陣（不使用RANSAC）
    F_no_ransac = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)[0]
    errors_no_ransac = compute_epipolar_error(F_no_ransac, pts1, pts2)
    
    # 計算F矩陣（使用RANSAC）
    F_ransac, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 
                                           ransacReprojThreshold=3.0, 
                                           confidence=0.999)
    
    # 檢查F矩陣是否成功計算
    if F_ransac is None:
        logger.warning("Failed to compute F matrix with RANSAC")
        mask = np.zeros(len(pts1), dtype=bool)
        errors_ransac = np.full(len(pts1), np.inf)
        mean_error_ransac = float('inf')
    else:
        mask = mask.ravel().astype(bool)
        errors_ransac = compute_epipolar_error(F_ransac, pts1, pts2)
        # 只使用內點計算平均誤差
        mean_error_ransac = np.mean(errors_ransac[mask]) if np.sum(mask) > 0 else float('inf')
    
    mean_error_no_ransac = np.mean(errors_no_ransac) if not np.all(np.isinf(errors_no_ransac)) else float('inf')
    
    # 調整圖片大小
    max_height = 800
    scale1 = max_height / img1.shape[0]
    scale2 = max_height / img2.shape[0]
    img1 = cv2.resize(img1, None, fx=scale1, fy=scale1)
    img2 = cv2.resize(img2, None, fx=scale2, fy=scale2)
    
    # 創建結果圖像
    combined = np.hstack((img1, img2))
    
    # 繪製匹配線和統計信息
    for idx, ((x1, y1), (x2, y2)) in enumerate(zip(pts1, pts2)):
        pt1 = (int(x1 * scale1), int(y1 * scale1))
        pt2 = (int(x2 * scale2) + img1.shape[1], int(y2 * scale2))
        
        # 使用不同顏色表示RANSAC內點和外點
        if mask[idx]:
            color = (0, 255, 0)  # 綠色為內點
        else:
            color = (0, 0, 255)  # 紅色為外點
            
        cv2.line(combined, pt1, pt2, color, 1)
    
    # 添加統計信息
    info_text = [
        f"Total matches: {len(pts1)}",
        f"RANSAC inliers: {np.sum(mask)}",
        f"RANSAC outliers: {len(mask) - np.sum(mask)}",
        f"Mean error (no RANSAC): {mean_error_no_ransac:.2f}px" if not np.isinf(mean_error_no_ransac) else "Mean error (no RANSAC): N/A",
        f"Mean error (RANSAC inliers): {mean_error_ransac:.2f}px" if not np.isinf(mean_error_ransac) else "Mean error (RANSAC): N/A",
    ]
    
    y_offset = 30
    for i, text in enumerate(info_text):
        y = y_offset + i * 30
        cv2.putText(combined, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2)
        cv2.putText(combined, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 0), 1)
    
    # 保存結果
    cv2.imwrite(str(output_path), combined)
    
    return {
        'total_matches': len(pts1),
        'ransac_inliers': np.sum(mask),
        'ransac_outliers': len(mask) - np.sum(mask),
        'mean_error_no_ransac': mean_error_no_ransac,
        'mean_error_ransac': mean_error_ransac
    }

def visualize_all_pairs(db_path, images_dir, output_dir):
    """視覺化所有影像配對"""
    # 創建輸出目錄
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 獲取所有配對
    pairs, images = get_image_pairs(db_path)
    
    # 連接資料庫獲取匹配數據
    conn = sqlite3.connect(db_path)
    
    stats = []
    for i, (id1, id2) in enumerate(pairs):
        # 獲取圖片路徑
        img1_path = Path(images_dir) / images[id1]
        img2_path = Path(images_dir) / images[id2]
        
        # 獲取匹配數據
        pair_id = id1 * 2147483647 + id2
        cursor = conn.execute("SELECT data FROM matches WHERE pair_id=?", (pair_id,))
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
        
        # 使用新的繪圖函數
        result_stats = draw_matches_with_F_comparison(
            img1_path, img2_path, matches_coords, 
            output_dir / f"match_{id1}_{id2}.jpg"
        )
        
        stats.append({
            'image_pair': f"{images[id1]} - {images[id2]}",
            **result_stats
        })
        
        print(f"Processed pair {i+1}/{len(pairs)}: {images[id1]} - {images[id2]}")
    
    # 打印統計信息表格
    print("\nMatching Statistics:")
    headers = ['Image Pair', 'Total Matches', 'RANSAC Inliers', 'RANSAC Outliers', 
              'Mean Error (no RANSAC)', 'Mean Error (RANSAC)']
    table = [[
        s['image_pair'], s['total_matches'], s['ransac_inliers'],
        s['ransac_outliers'], f"{s['mean_error_no_ransac']:.2f}",
        f"{s['mean_error_ransac']:.2f}"
    ] for s in stats]
    print(tabulate(table, headers=headers, tablefmt='grid'))
    
    conn.close()

def main():
    # 設置命令行參數
    parser = argparse.ArgumentParser(description='視覺化 COLMAP 資料庫中的影像��配')
    parser.add_argument('--database', '-d', required=True,
                        help='COLMAP 資料庫路徑 (database.db)')
    parser.add_argument('--images', '-i', required=True,
                        help='原始圖片目錄路徑')
    parser.add_argument('--output', '-o', required=True,
                        help='輸出結果目錄路徑')
    parser.add_argument('--max-height', type=int, default=800,
                        help='輸出圖片的最大高度 (預設: 800)')
    parser.add_argument('--line-thickness', type=int, default=1,
                        help='匹配線條粗細 (預設: 1)')
    parser.add_argument('--line-color', nargs=3, type=int, default=[0, 255, 0],
                        help='匹配線條顏色，BGR格式 (預設: 0 255 0)')

    args = parser.parse_args()

    db_path = args.database
    images_dir = args.images
    output_dir = args.output
    max_height = args.max_height
    line_thickness = args.line_thickness
    line_color = args.line_color

    visualize_all_pairs(db_path, images_dir, output_dir)

if __name__ == "__main__":
    main()