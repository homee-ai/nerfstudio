import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import glob

def get_background_color(image_dir: str, sample_size: int = 1000) -> Tuple[int, int, int]:
    """
    Determine the background color by analyzing color histograms of images.
    Returns the least frequent color as the likely background color.
    
    Args:
        image_dir: Directory containing the images
        sample_size: Number of random pixels to sample from each image
    
    Returns:
        Tuple of (R,G,B) values representing the background color
    """
    # Get list of image files
    image_files = glob.glob(str(Path(image_dir) / "*.jpg")) + \
                 glob.glob(str(Path(image_dir) / "*.png"))
    
    if not image_files:
        raise ValueError(f"No images found in {image_dir}")

    # Initialize array to store all sampled colors
    all_colors = []
    
    for img_path in image_files:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions
        h, w = img.shape[:2]
        
        # Randomly sample pixels
        y_coords = np.random.randint(0, h, sample_size)
        x_coords = np.random.randint(0, w, sample_size)
        
        # Get colors at sampled coordinates
        sampled_colors = img[y_coords, x_coords]
        all_colors.extend(sampled_colors)

    if not all_colors:
        raise ValueError("No valid images processed")
        
    # Convert to numpy array
    all_colors = np.array(all_colors)
    
    # Create color histogram
    # Using bins for each RGB channel
    hist, edges = np.histogramdd(
        all_colors, 
        bins=(32, 32, 32),  # 32 bins for each channel
        range=((0, 256), (0, 256), (0, 256))
    )
    
    # Find the bin with the least frequency (excluding empty bins)
    hist_flat = hist.flatten()
    non_zero_mask = hist_flat > 0
    if not non_zero_mask.any():
        raise ValueError("No valid color data found")
    
    min_freq_idx = hist_flat[non_zero_mask].argmin()
    min_freq_coords = np.unravel_index(min_freq_idx, hist.shape)
    
    # Convert bin indices to color values
    r = int((edges[0][min_freq_coords[0]] + edges[0][min_freq_coords[0] + 1]) / 2)
    g = int((edges[1][min_freq_coords[1]] + edges[1][min_freq_coords[1] + 1]) / 2)
    b = int((edges[2][min_freq_coords[2]] + edges[2][min_freq_coords[2] + 1]) / 2)
    
    return (r, g, b)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Determine background color from images")
    parser.add_argument("--image_dir", required=True, help="Directory containing input images")
    parser.add_argument("--sample_size", type=int, default=1000, 
                        help="Number of pixels to sample from each image")
    
    args = parser.parse_args()
    
    try:
        bg_color = get_background_color(args.image_dir, args.sample_size)
        print(f"Detected background color (RGB): {bg_color}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 