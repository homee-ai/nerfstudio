from dataclasses import dataclass
from pathlib import Path
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy.stats import zscore
import argparse
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
import os
import shutil

@dataclass
class ReprojectionError:
    image_id: int
    image_name: str 
    point2D_idx: int
    point3D_id: int
    x: float  # 原始2D點x坐標
    y: float  # 原始2D點y坐標
    proj_x: float  # 投影點x坐標
    proj_y: float  # 投影點y坐標
    error: float  # 重投影誤差

def parse_reprojection_errors(file_path: Path) -> List[ReprojectionError]:
    """
    解析重投影誤差文件
    
    Args:
        file_path: 重投影誤差文件路徑
        
    Returns:
        包含所有重投影誤差數據的列表
    """
    errors = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # 跳過註釋行
            if line.startswith('#'):
                continue
                
            # 解析數據行
            parts = line.strip().split()
            if len(parts) == 9:
                error = ReprojectionError(
                    image_id=int(parts[0]),
                    image_name=parts[1],
                    point2D_idx=int(parts[2]), 
                    point3D_id=int(parts[3]),
                    x=float(parts[4]),
                    y=float(parts[5]),
                    proj_x=float(parts[6]),
                    proj_y=float(parts[7]),
                    error=float(parts[8])
                )
                errors.append(error)
                
    return errors

def compute_error_statistics(errors: List[ReprojectionError]):
    """
    計算重投影誤差的統計信息
    
    Args:
        errors: 重投影誤差列表
        
    Returns:
        包含均值、中位數、標準差等統計信息的字典
    """
    error_values = np.array([e.error for e in errors])
    
    stats = {
        'mean': float(np.mean(error_values)),
        'median': float(np.median(error_values)), 
        'std': float(np.std(error_values)),
        'min': float(np.min(error_values)),
        'max': float(np.max(error_values)),
        'count': len(errors)
    }
    
    return stats

def compute_per_image_mean_reprojection_error(errors: List[ReprojectionError]) -> Dict[int, float]:
    """
    計算每張影像的平均reprojection error
    
    Args:
        errors: 重投影誤差列表
        
    Returns:
        Dict[image_id, mean_error]
    """
    image_errors = defaultdict(list)
    for error in errors:
        image_errors[error.image_id].append(error.error)
        
    return {img_id: np.mean(errs) for img_id, errs in image_errors.items()}

def compute_per_image_point_distribution(errors: List[ReprojectionError]) -> Dict[int, float]:
    """
    計算每張影像的2d point分佈發散程度(使用標準差)
    
    Args:
        errors: 重投影誤差列表
        
    Returns:
        Dict[image_id, distribution_std]
    """
    image_points = defaultdict(list)
    for error in errors:
        image_points[error.image_id].append((error.x, error.y))
        
    distribution_scores = {}
    for img_id, points in image_points.items():
        points = np.array(points)
        # 計算點的x和y座標的標準差，取較大值作為發散程度
        std_x = np.std(points[:, 0])
        std_y = np.std(points[:, 1])
        distribution_scores[img_id] = max(std_x, std_y)
        
    return distribution_scores

def compute_per_image_point_count(errors: List[ReprojectionError]) -> Dict[int, int]:
    """
    計算每張影像的2d point數量
    
    Args:
        errors: 重投影誤差列表
        
    Returns:
        Dict[image_id, point_count]
    """
    image_points = defaultdict(set)
    for error in errors:
        image_points[error.image_id].add(error.point2D_idx)
        
    return {img_id: len(points) for img_id, points in image_points.items()}

def compute_mean_reprojection_error(errors: List[ReprojectionError]) -> float:
    """
    計算所有點的平均重投影誤差
    
    Args:
        errors: 重投影誤差列表
        
    Returns:
        mean_error
    """
    return np.mean([error.error for error in errors])

def compute_reprojection_error_std(errors: List[ReprojectionError]) -> float:
    """
    計算所有點的重投影誤差的標準差
    
    Args:
        errors: 重投影誤差列表
        
    Returns:
        error_std
    """
    return np.std([error.error for error in errors])

def find_outlier_points(errors: List[ReprojectionError], 
                       threshold: float = 3.0) -> List[Tuple[int, int, float]]:
    """
    找出異常大的重投影誤差的點
    
    Args:
        errors: 重投影誤差列表
        threshold: z-score閾值，預設為3(3個標準差)
        
    Returns:
        List of (image_id, point2D_idx, error_value) for outlier points
    """
    error_values = np.array([error.error for error in errors])
    z_scores = zscore(error_values)
    
    outliers = []
    for error, z_score in zip(errors, z_scores):
        if z_score > threshold:
            outliers.append((error.image_id, error.point2D_idx, error.error))
            
    return sorted(outliers, key=lambda x: x[2], reverse=True)  # 按誤差大小排序

def visualize_points_on_image(image_path: Path, 
                            points: List[Tuple[float, float]], 
                            output_path: Path,
                            title: str = None):
    """
    在影像上繪製2D特徵點
    
    Args:
        image_path: 原始影像路徑
        points: 2D點列表 [(x, y), ...]
        output_path: 輸出影像路徑
        title: 影像標題
    """
    # 讀取影像
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return
        
    # 轉換為RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 創建圖形
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    
    # 繪製點
    points = np.array(points)
    plt.scatter(points[:, 0], points[:, 1], c='red', s=20, alpha=0.6)
    
    if title:
        plt.title(title)
    
    # 保存圖片
    plt.savefig(output_path)
    plt.close()

def analyze_images_with_least_points(errors: List[ReprojectionError], 
                                   image_dir: Path,
                                   output_dir: Path,
                                   top_n: int = 10):
    """
    分析並視覺化點數最少的影像
    
    Args:
        errors: 重投影誤差列表
        image_dir: 原始影像目錄
        output_dir: 輸出目錄
        top_n: 要分析的影像數量
    """
    # 計算每張影像的點數
    point_counts = compute_per_image_point_count(errors)
    
    # 按點數排序
    sorted_images = sorted(point_counts.items(), key=lambda x: x[1])
    
    # 創建輸出目錄
    vis_dir = output_dir / "images_with_least_points"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集分析結果
    analysis_results = []
    
    # 處理前N張點數最少的影像
    for img_id, point_count in sorted_images[:top_n]:
        # 找出該影像的所有點
        image_points = [(e.x, e.y) for e in errors if e.image_id == img_id]
        
        # 找到對應的影像文件
        image_name = next((e.image_name for e in errors if e.image_id == img_id), None)
        if image_name:
            image_path = image_dir / image_name
            if image_path.exists():
                output_path = vis_dir / f"least_points_{img_id}_{point_count}points.png"
                
                # 繪製並保存視覺化結果
                visualize_points_on_image(
                    image_path,
                    image_points,
                    output_path,
                    f"Image {img_id}: {point_count} points"
                )
                
                analysis_results.append({
                    'image_id': img_id,
                    'image_name': image_name,
                    'point_count': point_count,
                    'visualization': output_path
                })
            
    return analysis_results

class Point3DAnalysis:
    """用於3D點分析的類，包含優化的數據結構"""
    def __init__(self, errors: List[ReprojectionError]):
        # 建立高效率的查詢結構
        self.point3d_to_2d = defaultdict(list)  # 3D點到對應2D點的映射
        self.point3d_errors = defaultdict(list)  # 3D點對應的重投影誤差
        self.image_points = defaultdict(list)    # 每張影像的2D點
        
        # 初始化數據結構
        for error in errors:
            point2d = {
                'image_id': error.image_id,
                'image_name': error.image_name,
                'point2D_idx': error.point2D_idx,
                'x': error.x,
                'y': error.y,
                'proj_x': error.proj_x,
                'proj_y': error.proj_y,
                'error': error.error
            }
            self.point3d_to_2d[error.point3D_id].append(point2d)
            self.point3d_errors[error.point3D_id].append(error.error)
            self.image_points[error.image_id].append(point2d)

    def get_corresponding_2d_points(self, point3d_id: int) -> List[Dict]:
        """查詢對應於特定3D點的所有2D點"""
        return self.point3d_to_2d.get(point3d_id, [])

    def get_2d_point_count(self, point3d_id: int) -> int:
        """獲取3D點對應的2D點數量"""
        return len(self.point3d_to_2d.get(point3d_id, []))

    def get_mean_reprojection_error(self, point3d_id: int) -> float:
        """計算3D點對應2D點的平均重投影誤差"""
        errors = self.point3d_errors.get(point3d_id, [])
        return np.mean(errors) if errors else 0.0

    def get_high_error_points3d(self, top_k: int = 10) -> List[Tuple[int, float, List[Dict]]]:
        """獲取平均重投影誤差最高的前K個3D點"""
        point3d_mean_errors = [
            (point3d_id, np.mean(errors), self.point3d_to_2d[point3d_id])
            for point3d_id, errors in self.point3d_errors.items()
            if len(errors) >= 2  # 可選：過濾掉觀測次數過少的點
        ]
        return sorted(point3d_mean_errors, key=lambda x: x[1], reverse=True)[:top_k]

def visualize_3d_point_observations(
    point3d_id: int,
    observations: List[Dict],
    image_dir: Path,
    output_dir: Path,
    max_images_per_point: int = 9
) -> List[Path]:
    """
    為3D點的所有觀測生成視覺化結果
    
    Args:
        point3d_id: 3D點ID
        observations: 該3D點的所有2D觀測
        image_dir: 原始影像目錄
        output_dir: 輸出目錄
        max_images_per_point: 每個3D點最多顯示的影像數
        
    Returns:
        生成的視覺化文件路徑列表
    """
    # 創建該3D點的輸出目錄
    point_dir = output_dir / f"point3d_{point3d_id}"
    point_dir.mkdir(parents=True, exist_ok=True)
    
    visualization_paths = []
    
    # 按誤差大小排序觀測
    sorted_obs = sorted(observations, key=lambda x: x['error'], reverse=True)
    
    # 限制處理的影像數量
    for idx, obs in enumerate(sorted_obs[:max_images_per_point]):
        image_path = image_dir / obs['image_name']
        if not image_path.exists():
            continue
            
        # 生成視覺化
        output_path = point_dir / f"image_{obs['image_id']}_error_{obs['error']:.3f}.png"
        points = [(obs['x'], obs['y'])]  # 原始點
        proj_points = [(obs['proj_x'], obs['proj_y'])]  # 投影點
        
        # 讀取並轉換影像
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 創建圖形
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        
        # 繪製原始點和投影點
        plt.scatter(np.array(points)[:, 0], np.array(points)[:, 1], 
                   c='red', s=100, label='Original', marker='o')
        plt.scatter(np.array(proj_points)[:, 0], np.array(proj_points)[:, 1], 
                   c='blue', s=100, label='Projected', marker='x')
        
        # 添加連線
        for p1, p2 in zip(points, proj_points):
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g--', alpha=0.5)
        
        plt.title(f"Point3D {point3d_id} in Image {obs['image_id']}\nReprojection Error: {obs['error']:.3f}")
        plt.legend()
        
        # 保存圖片
        plt.savefig(output_path)
        plt.close()
        
        visualization_paths.append(output_path)
    
    return visualization_paths

def analyze_high_error_3d_points(
    errors: List[ReprojectionError],
    image_dir: Path,
    output_dir: Path,
    top_k: int = 10
) -> Dict:
    """
    分析重投影誤差最高的3D點
    
    Args:
        errors: 重投影誤差列表
        image_dir: 原始影像目錄
        output_dir: 輸出目錄
        top_k: 分析的3D點數量
    
    Returns:
        分析結果字典
    """
    # 初始化分析器
    analyzer = Point3DAnalysis(errors)
    
    # 獲取高誤差的3D點
    high_error_points = analyzer.get_high_error_points3d(top_k)
    
    # 創建輸出目錄
    high_error_dir = output_dir / "high_error_points3d"
    if high_error_dir.exists():
        shutil.rmtree(high_error_dir)
    high_error_dir.mkdir(parents=True)
    
    # 分析結果
    analysis_results = []
    
    # 處理每個高誤差3D點
    for point3d_id, mean_error, observations in high_error_points:
        # 生成視覺化
        vis_paths = visualize_3d_point_observations(
            point3d_id,
            observations,
            image_dir,
            high_error_dir
        )
        
        # 收集結果
        result = {
            'point3d_id': point3d_id,
            'mean_error': mean_error,
            'num_observations': len(observations),
            'visualizations': vis_paths
        }
        analysis_results.append(result)
    
    return analysis_results

def analyze_high_error_images(
    errors: List[ReprojectionError],
    image_dir: Path,
    output_dir: Path,
    top_k: int = 10
) -> List[Dict]:
    """
    分析並視覺化重投影誤差最高的影像
    
    Args:
        errors: 重投影誤差列表
        image_dir: 原始影像目錄
        output_dir: 輸出目錄
        top_k: 要分析的影像數量
    """
    # 計算每張影像的平均誤差
    image_errors = compute_per_image_mean_reprojection_error(errors)
    
    # 按誤差排序
    sorted_images = sorted(image_errors.items(), key=lambda x: x[1], reverse=True)
    
    # 創建輸出目錄
    vis_dir = output_dir / "high_error_images"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集分析結果
    analysis_results = []
    
    # 處理前K張誤差最高的影像
    for img_id, mean_error in sorted_images[:top_k]:
        # 找出該影像的所有點
        image_points = []
        proj_points = []
        for error in errors:
            if error.image_id == img_id:
                image_points.append((error.x, error.y))
                proj_points.append((error.proj_x, error.proj_y))
        
        # 找到對應的影像文件
        image_name = next((e.image_name for e in errors if e.image_id == img_id), None)
        if image_name:
            image_path = image_dir / image_name
            if image_path.exists():
                output_path = vis_dir / f"high_error_image_{img_id}_error_{mean_error:.3f}.png"
                
                # 讀取影像
                img = cv2.imread(str(image_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 創建圖形
                plt.figure(figsize=(12, 8))
                plt.imshow(img)
                
                # 繪製原始點和投影點
                points = np.array(image_points)
                proj = np.array(proj_points)
                plt.scatter(points[:, 0], points[:, 1], 
                          c='red', s=20, alpha=0.6, label='Original')
                plt.scatter(proj[:, 0], proj[:, 1], 
                          c='blue', s=20, alpha=0.6, label='Projected')
                
                # 添加連線
                for p1, p2 in zip(points, proj):
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g--', alpha=0.2)
                
                plt.title(f"Image {img_id}: Mean Error {mean_error:.3f} pixels")
                plt.legend()
                
                # 保存圖片
                plt.savefig(output_path)
                plt.close()
                
                analysis_results.append({
                    'image_id': img_id,
                    'image_name': image_name,
                    'mean_error': mean_error,
                    'point_count': len(image_points),
                    'visualization': output_path
                })
    
    return analysis_results

def analyze_error_vs_point_count(
    errors: List[ReprojectionError],
    output_dir: Path
) -> Dict:
    """
    分析影像重投影誤差與特徵點數量的關係
    
    Args:
        errors: 重投影誤差列表
        output_dir: 輸出目錄
        
    Returns:
        分析結果字典，包含相關係數等統計信息
    """
    # 計算每張影像的平均誤差和特徵點數量
    image_errors = compute_per_image_mean_reprojection_error(errors)
    point_counts = compute_per_image_point_count(errors)
    
    # 準備數據進行相關性分析
    images = list(set(image_errors.keys()) & set(point_counts.keys()))
    error_values = [image_errors[img_id] for img_id in images]
    count_values = [point_counts[img_id] for img_id in images]
    
    # 計算相關係數
    correlation = np.corrcoef(error_values, count_values)[0, 1]
    
    # 創建散點圖
    plt.figure(figsize=(12, 8))
    plt.scatter(count_values, error_values, alpha=0.5)
    
    # 添加趨勢線
    z = np.polyfit(count_values, error_values, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(count_values), max(count_values), 100)
    plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'Trend line (slope: {z[0]:.6f})')
    
    plt.xlabel('Number of 2D Points')
    plt.ylabel('Mean Reprojection Error (pixels)')
    plt.title('Reprojection Error vs Number of 2D Points')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加相關係數文字
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # 保存圖片
    output_path = output_dir / "error_vs_points_correlation.png"
    plt.savefig(output_path)
    plt.close()
    
    # 計算其他統計信息
    stats = {
        'correlation': correlation,
        'trend_slope': z[0],
        'trend_intercept': z[1],
        'mean_error': np.mean(error_values),
        'mean_point_count': np.mean(count_values),
        'visualization': output_path
    }
    
    return stats

def compute_pose_quality_metrics(errors: List[ReprojectionError]) -> Dict[int, Dict]:
    """
    計算每張影像的pose質量指標
    
    Args:
        errors: 重投影誤差列表
        
    Returns:
        Dict[image_id, metrics]，其中metrics包含多個評估指標
    """
    # 按影像ID組織數據
    image_errors = defaultdict(list)
    image_points = defaultdict(list)
    
    for error in errors:
        image_errors[error.image_id].append(error.error)
        image_points[error.image_id].append((error.x, error.y))
    
    # 計算每張影像的指標
    metrics = {}
    for img_id in image_errors.keys():
        error_values = np.array(image_errors[img_id])
        points = np.array(image_points[img_id])
        
        # 計算點的空間分布
        if len(points) > 1:
            points_std_x = np.std(points[:, 0])
            points_std_y = np.std(points[:, 1])
            points_coverage = points_std_x * points_std_y
        else:
            points_coverage = 0
            
        metrics[img_id] = {
            'mean_error': np.mean(error_values),
            'error_std': np.std(error_values),
            'max_error': np.max(error_values),
            'point_count': len(points),
            'points_coverage': points_coverage,
            # 綜合分數：結合多個指標
            'quality_score': np.mean(error_values) * (1 + np.std(error_values)) / 
                           (np.log1p(len(points)) * np.log1p(points_coverage))
        }
    
    return metrics

def find_problematic_poses(
    errors: List[ReprojectionError],
    image_dir: Path,
    output_dir: Path,
    top_k: int = 10
) -> List[Dict]:
    """
    找出pose準確性最差的影像
    
    Args:
        errors: 重投影誤差列表
        image_dir: 原始影像目錄
        output_dir: 輸出目錄
        top_k: 要分析的影像數量
        
    Returns:
        問題影像的分析結果列表
    """
    # 計算pose質量指標
    pose_metrics = compute_pose_quality_metrics(errors)
    
    # 按質量分數排序（分數越高表示pose越不準確）
    sorted_images = sorted(
        pose_metrics.items(),
        key=lambda x: x[1]['quality_score'],
        reverse=True
    )
    
    # 創建輸出目錄
    vis_dir = output_dir / "problematic_poses"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集分析結果
    analysis_results = []
    
    # 處理前K張最有問題的影像
    for img_id, metrics in sorted_images[:top_k]:
        # 找出該影像的所有點
        image_points = []
        proj_points = []
        for error in errors:
            if error.image_id == img_id:
                image_points.append((error.x, error.y))
                proj_points.append((error.proj_x, error.proj_y))
        
        # 找到對應的影像文件
        image_name = next((e.image_name for e in errors if e.image_id == img_id), None)
        if image_name:
            image_path = image_dir / image_name
            if image_path.exists():
                output_path = vis_dir / f"bad_pose_{img_id}_score_{metrics['quality_score']:.3f}.png"
                
                # 讀取影像
                img = cv2.imread(str(image_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 創建圖形
                plt.figure(figsize=(12, 8))
                plt.imshow(img)
                
                # 繪製原始點和投影點
                points = np.array(image_points)
                proj = np.array(proj_points)
                plt.scatter(points[:, 0], points[:, 1], 
                          c='red', s=20, alpha=0.6, label='Original')
                plt.scatter(proj[:, 0], proj[:, 1], 
                          c='blue', s=20, alpha=0.6, label='Projected')
                
                # 添加連線
                for p1, p2 in zip(points, proj):
                    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g--', alpha=0.2)
                
                plt.title(f"Image {img_id}\n" + 
                         f"Quality Score: {metrics['quality_score']:.3f}\n" +
                         f"Mean Error: {metrics['mean_error']:.3f} ± {metrics['error_std']:.3f}")
                plt.legend()
                
                # 保存圖片
                plt.savefig(output_path)
                plt.close()
                
                analysis_results.append({
                    'image_id': img_id,
                    'image_name': image_name,
                    'metrics': metrics,
                    'visualization': output_path
                })
    
    return analysis_results

def save_problematic_images(
    errors: List[ReprojectionError],
    image_dir: Path,
    output_dir: Path,
    top_k: int = 10
) -> List[Dict]:
    """
    找出並儲存pose準確性最差的影像
    
    Args:
        errors: 重投影誤差列表
        image_dir: 原始影像目錄
        output_dir: 輸出目錄
        top_k: 要儲存的影像數量
        
    Returns:
        問題影像的資訊列表
    """
    # 計算pose質量指標
    pose_metrics = compute_pose_quality_metrics(errors)
    
    # 按質量分數排序（分數越高表示pose越不準確）
    sorted_images = sorted(
        pose_metrics.items(),
        key=lambda x: x[1]['quality_score'],
        reverse=True
    )
    
    # 創建輸出目錄
    save_dir = output_dir / "problematic_images"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 儲存結果資訊
    saved_images = []
    
    # 處理前K張最有問題的影像
    for img_id, metrics in sorted_images[:top_k]:
        # 找到對應的影像文件
        image_name = next((e.image_name for e in errors if e.image_id == img_id), None)
        if image_name:
            src_path = image_dir / image_name
            if src_path.exists():
                # 構建輸出文件名（包含質量分數）
                output_name = f"bad_pose_{img_id}_score_{metrics['quality_score']:.3f}_{image_name}"
                dst_path = save_dir / output_name
                
                # 複製影像文件
                shutil.copy2(src_path, dst_path)
                
                saved_images.append({
                    'image_id': img_id,
                    'original_name': image_name,
                    'saved_name': output_name,
                    'quality_score': metrics['quality_score'],
                    'metrics': metrics,
                    'saved_path': dst_path
                })
    
    # 生成描述文件
    description_path = save_dir / "problematic_images_info.txt"
    with open(description_path, 'w', encoding='utf-8') as f:
        f.write("=== 問題影像清單 ===\n\n")
        for img in saved_images:
            f.write(f"Image ID: {img['image_id']}\n")
            f.write(f"Original name: {img['original_name']}\n")
            f.write(f"Saved as: {img['saved_name']}\n")
            f.write(f"Quality score: {img['quality_score']:.3f}\n")
            f.write(f"Mean error: {img['metrics']['mean_error']:.3f} ± {img['metrics']['error_std']:.3f}\n")
            f.write(f"Point count: {img['metrics']['point_count']}\n")
            f.write(f"Points coverage: {img['metrics']['points_coverage']:.1f}\n")
            f.write("\n")
    
    return saved_images

def analyze_3d_point_observations_vs_error(errors: List[ReprojectionError], 
                                         output_dir: Path) -> Dict:
    """
    分析3D點的觀測次數與重投影誤差的關係
    
    Args:
        errors: 重投影誤差列表
        output_dir: 輸出目錄
        
    Returns:
        包含相關性分析結果的字典
    """
    # 初始化分析器
    analyzer = Point3DAnalysis(errors)
    
    # 收集每個3D點的觀測次數和平均誤差
    observations = []
    mean_errors = []
    
    for point3d_id in analyzer.point3d_to_2d.keys():
        obs_count = analyzer.get_2d_point_count(point3d_id)
        mean_error = analyzer.get_mean_reprojection_error(point3d_id)
        observations.append(obs_count)
        mean_errors.append(mean_error)
    
    # 計算相關係數
    correlation = np.corrcoef(observations, mean_errors)[0, 1]
    
    # 創建散點圖
    plt.figure(figsize=(12, 8))
    plt.scatter(observations, mean_errors, alpha=0.5)
    
    # 添加趨勢線
    z = np.polyfit(observations, mean_errors, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(observations), max(observations), 100)
    plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, 
            label=f'Trend line (slope: {z[0]:.6f})')
    
    plt.xlabel('Number of Observations (2D points)')
    plt.ylabel('Mean Reprojection Error (pixels)')
    plt.title('3D Point Mean Reprojection Error vs Number of Observations')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加相關係數文字
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # 保存圖片
    output_path = output_dir / "3d_point_error_vs_observations.png"
    plt.savefig(output_path)
    plt.close()
    
    # 計算一些統計信息
    stats = {
        'correlation': correlation,
        'trend_slope': z[0],
        'trend_intercept': z[1],
        'mean_observations': np.mean(observations),
        'median_observations': np.median(observations),
        'max_observations': max(observations),
        'visualization': output_path
    }
    
    return stats

def gen_analysis_report(file_path: Path, output_path: Path = None, image_dir: Optional[Path] = None) -> str:
    """
    生成詳盡的重投影誤差分析報告
    
    Args:
        file_path: 重投影誤差文件路徑
        output_path: 報告輸出路徑(可選)
        image_dir: 原始影像目錄路徑(可選)
    """
    errors = parse_reprojection_errors(file_path)
    
    # 準備報告內容
    report = []
    report.append("=== 重投影誤差分析報告 ===")
    report.append(f"分析文件: {file_path}")
    report.append(f"分析時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n1. 基本統計信息")
    report.append("-----------------")
    
    # 1. 基本統計信息
    stats = compute_error_statistics(errors)
    report.append(f"總特徵點數: {stats['count']}")
    report.append(f"平均重投影誤差: {stats['mean']:.3f}")
    report.append(f"中位數誤差: {stats['median']:.3f}")
    report.append(f"誤差標準差: {stats['std']:.3f}")
    report.append(f"最小誤差: {stats['min']:.3f}")
    report.append(f"最大誤差: {stats['max']:.3f}")
    
    # 2. 每張影像的分析
    report.append("\n2. 影像級別分析")
    report.append("-----------------")
    
    image_errors = compute_per_image_mean_reprojection_error(errors)
    point_distributions = compute_per_image_point_distribution(errors)
    point_counts = compute_per_image_point_count(errors)
    
    # 找出誤差最大和最小的影像
    max_error_img = max(image_errors.items(), key=lambda x: x[1])
    min_error_img = min(image_errors.items(), key=lambda x: x[1])
    
    report.append(f"影像數量: {len(image_errors)}")
    report.append(f"每張影像平均特徵點數: {np.mean(list(point_counts.values())):.1f}")
    report.append(f"特徵點數範圍: {min(point_counts.values())} - {max(point_counts.values())}")
    report.append(f"影像平均誤差範圍: {min(image_errors.values()):.3f} - {max(image_errors.values()):.3f}")
    report.append(f"\n誤差最大的影像:")
    report.append(f"- Image {max_error_img[0]}: {max_error_img[1]:.3f} pixels")
    report.append(f"- 特徵點數: {point_counts[max_error_img[0]]}")
    report.append(f"- 點分佈標準差: {point_distributions[max_error_img[0]]:.3f}")
    
    report.append(f"\n誤差最小的影像:")
    report.append(f"- Image {min_error_img[0]}: {min_error_img[1]:.3f} pixels")
    report.append(f"- 特徵點數: {point_counts[min_error_img[0]]}")
    report.append(f"- 點分佈標準差: {point_distributions[min_error_img[0]]:.3f}")
    
    # 3. 異常點分析
    report.append("\n3. 異常點分析")
    report.append("-----------------")
    
    outliers = find_outlier_points(errors)
    outlier_ratio = len(outliers) / len(errors) * 100
    
    report.append(f"檢測到的異常點數量: {len(outliers)} ({outlier_ratio:.2f}%)")
    report.append("\n前10個最誤差的點:")
    for i, (img_id, point_id, error) in enumerate(outliers[:10], 1):
        report.append(f"{i}. Image {img_id}, Point {point_id}: {error:.3f} pixels")
    
    # 4. 分佈分析
    report.append("\n4. 誤差分佈分析")
    report.append("-----------------")
    
    error_values = np.array([e.error for e in errors])
    percentiles = np.percentile(error_values, [25, 50, 75, 90, 95, 99])
    
    report.append(f"25th percentile: {percentiles[0]:.3f}")
    report.append(f"50th percentile (median): {percentiles[1]:.3f}")
    report.append(f"75th percentile: {percentiles[2]:.3f}")
    report.append(f"90th percentile: {percentiles[3]:.3f}")
    report.append(f"95th percentile: {percentiles[4]:.3f}")
    report.append(f"99th percentile: {percentiles[5]:.3f}")
    
    # 5. 建議
    report.append("\n5. 分析建議")
    report.append("-----------------")
    
    # 根據統計結果給出建議
    if stats['mean'] > 2.0:
        report.append("- 平均重投影誤差偏大，建議檢查相機標定參數")
    if outlier_ratio > 5:
        report.append("- 異常點比例較高，建議進行異常點過濾")
    if max(image_errors.values()) > 3 * stats['mean']:
        report.append("- 部分影像誤差顯偏大，建議檢查這些影像的特徵點品質")
    
    # 在影像級別分析部分後添加高誤差影像視覺化
    if image_dir and image_dir.exists():
        report.append("\n6. 高重投影誤差影像分析")
        report.append("-----------------")
        
        # 創建輸出目錄
        vis_output_dir = output_path.parent if output_path else Path(".")
        
        # 分析高誤差影像
        high_error_images = analyze_high_error_images(
            errors,
            image_dir,
            vis_output_dir
        )
        
        report.append(f"\n分析了 {len(high_error_images)} 張高誤差影像:")
        for result in high_error_images:
            report.append(f"\n- Image {result['image_id']} ({result['image_name']})")
            report.append(f"  平均重投影誤差: {result['mean_error']:.3f}")
            report.append(f"  特徵點數量: {result['point_count']}")
            report.append(f"  視覺化結果: {result['visualization']}")
    
    # 添加點數最少影像分析部分
    if image_dir and image_dir.exists():
        report.append("\n7. 點數最少影像分析")
        report.append("-----------------")
        
        # 創建輸出目錄
        vis_output_dir = output_path.parent if output_path else Path(".")
        
        # 分析點數最少的影像
        least_points_analysis = analyze_images_with_least_points(
            errors,
            image_dir,
            vis_output_dir
        )
        
        report.append(f"\n分析了 {len(least_points_analysis)} 張點數最少的影像:")
        for result in least_points_analysis:
            report.append(f"\n- Image {result['image_id']} ({result['image_name']})")
            report.append(f"  點數: {result['point_count']}")
            report.append(f"  視覺化結果: {result['visualization']}")
    
    # 添加3D點分析部分
    if image_dir and image_dir.exists():
        report.append("\n8. 高重投影誤差3D點分析")
        report.append("-----------------")
        
        high_error_analysis = analyze_high_error_3d_points(
            errors,
            image_dir,
            output_path.parent if output_path else Path(".")
        )
        
        report.append(f"\n分析了 {len(high_error_analysis)} 個高誤差3D點:")
        for result in high_error_analysis:
            report.append(f"\n- Point3D {result['point3d_id']}")
            report.append(f"  平均重投影誤差: {result['mean_error']:.3f}")
            report.append(f"  觀測次數: {result['num_observations']}")
            report.append(f"  視覺化結果數量: {len(result['visualizations'])}")
    
    # 添加誤差與特徵點數量關係分析
    report.append("\n8. 重投影誤差與特徵點數量關係分析")
    report.append("-----------------")
    
    # 創建輸出目錄
    vis_output_dir = output_path.parent if output_path else Path(".")
    
    # 進行相關性分析
    correlation_analysis = analyze_error_vs_point_count(errors, vis_output_dir)
    
    report.append(f"\n相關性分析結果:")
    report.append(f"- 相關係數: {correlation_analysis['correlation']:.3f}")
    report.append(f"- 趨勢線斜率: {correlation_analysis['trend_slope']:.6f}")
    report.append(f"- 趨勢線截距: {correlation_analysis['trend_intercept']:.3f}")
    report.append(f"- 平均重投影誤差: {correlation_analysis['mean_error']:.3f} pixels")
    report.append(f"- 平均特徵點數量: {correlation_analysis['mean_point_count']:.1f}")
    
    # 解釋相關性
    correlation = correlation_analysis['correlation']
    if abs(correlation) < 0.1:
        report.append("\n分析結論: 重投影誤差與特徵點數量幾乎無相關性")
    elif abs(correlation) < 0.3:
        report.append("\n分析結論: 重投影誤差與特徵點數量呈弱相關性")
    elif abs(correlation) < 0.5:
        report.append("\n分析結論: 重投影誤差與特徵點數量呈中等相關性")
    else:
        report.append("\n分析結論: 重投影誤差與特徵點數量呈強相關性")
        
    if correlation > 0:
        report.append("相關方向: 特徵點數量越多，重投影誤差傾向於越大")
    else:
        report.append("相關方向: 特徵點數量越多，重投影誤差傾向於越小")
    
    report.append(f"\n視覺化結果: {correlation_analysis['visualization']}")
    
    # 添加pose準確性分析
    if image_dir and image_dir.exists():
        report.append("\n9. Camera Pose準確性分析")
        report.append("-----------------")
        
        problematic_poses = find_problematic_poses(
            errors,
            image_dir,
            output_path.parent if output_path else Path(".")
        )
        
        report.append(f"\n找出 {len(problematic_poses)} 個可能有問題的camera pose:")
        for result in problematic_poses:
            metrics = result['metrics']
            report.append(f"\n- Image {result['image_id']} ({result['image_name']})")
            report.append(f"  質量分數: {metrics['quality_score']:.3f}")
            report.append(f"  平均重投影誤差: {metrics['mean_error']:.3f} ± {metrics['error_std']:.3f}")
            report.append(f"  最大重投影誤差: {metrics['max_error']:.3f}")
            report.append(f"  特徵點數量: {metrics['point_count']}")
            report.append(f"  點分布覆蓋度: {metrics['points_coverage']:.1f}")
            report.append(f"  視覺化結果: {result['visualization']}")
    
    # 在 Camera Pose 準確性分析部分後添加影像儲存
    if image_dir and image_dir.exists():
        report.append("\n10. 問題影像儲存")
        report.append("-----------------")
        
        # 儲存問題影像
        saved_images = save_problematic_images(
            errors,
            image_dir,
            output_path.parent if output_path else Path(".")
        )
        
        report.append(f"\n已儲存 {len(saved_images)} 張問題影像:")
        for img in saved_images:
            report.append(f"\n- Image {img['image_id']} ({img['original_name']})")
            report.append(f"  儲存為: {img['saved_name']}")
            report.append(f"  質量分數: {img['quality_score']:.3f}")
            report.append(f"  儲存路徑: {img['saved_path']}")
    # 添加3D點觀測次數與誤差關係分析
    report.append("\n11. 3D點觀測次數與重投影誤差關係分析")
    report.append("-----------------")
    
    observation_analysis = analyze_3d_point_observations_vs_error(
        errors,
        output_path.parent if output_path else Path(".")
    )
    
    report.append(f"\n相關性分析結果:")
    report.append(f"- 相關係數: {observation_analysis['correlation']:.3f}")
    report.append(f"- 趨勢線斜率: {observation_analysis['trend_slope']:.6f}")
    report.append(f"- 趨勢線截距: {observation_analysis['trend_intercept']:.3f}")
    report.append(f"- 平均觀測次數: {observation_analysis['mean_observations']:.1f}")
    report.append(f"- 中位數觀測次數: {observation_analysis['median_observations']:.1f}")
    report.append(f"- 最大觀測次數: {observation_analysis['max_observations']}")
    
    # 解釋相關性
    correlation = observation_analysis['correlation']
    if abs(correlation) < 0.1:
        report.append("\n分析結論: 3D點的重投影誤差與其觀測次數幾乎無相關性")
    elif abs(correlation) < 0.3:
        report.append("\n分析結論: 3D點的重投影誤差與其觀測次數呈弱相關性")
    elif abs(correlation) < 0.5:
        report.append("\n分析結論: 3D點的重投影誤差與其觀測次數呈中等相關性")
    else:
        report.append("\n分析結論: 3D點的重投影誤差與其觀測次數呈強相關性")
        
    if correlation > 0:
        report.append("相關方向: 觀測次數越多，重投影誤差傾向於越大")
    else:
        report.append("相關方向: 觀測次數越多，重投影誤差傾向於越小")
    
    report.append(f"\n視覺化結果: {observation_analysis['visualization']}")
    
    # 將報告轉換為字符串
    report_text = "\n".join(report)
    
    # 如果提供輸出路徑，保存報告
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
    
    return report_text

# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分析重投影誤差並生成報告')
    
    parser.add_argument('input', type=str,
                        help='重投影誤差文件的路徑')
    
    parser.add_argument('-o', '--output', type=str,
                        help='報告輸出文件的路徑 (可選)',
                        default=None)
    
    parser.add_argument('-i', '--image_dir', type=str,
                        help='原始影像目錄路徑 (可選，用於生成視覺化結果)',
                        default=None)
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if args.output is None:
        output_path = input_path.parent / f"analysis_report_{input_path.stem}.txt"
    else:
        output_path = Path(args.output)
    
    image_dir = Path(args.image_dir) if args.image_dir else None
    
    report = gen_analysis_report(input_path, output_path, image_dir)
    print(report)