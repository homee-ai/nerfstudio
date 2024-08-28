import json
import os
import numpy as np
from PIL import Image, ImageDraw
import glob
import argparse

def generate_mask(polygons, width, height):
    mask = Image.new('L', (width, height), 1)
    draw = ImageDraw.Draw(mask)
    
    for polygon in polygons:
        if polygon['contentType'] == 'polygon' and len(polygon['content']) > 2:
            points = [(point['x'], point['y']) for point in polygon['content']]
            draw.polygon(points, fill=0)
    
    return np.array(mask)

def process_image(image_path, json_path, output_dir):
    img = Image.open(image_path)
    width, height = img.size
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        mask = 255 * generate_mask(data, width, height)
    else:
        mask = 255 * np.ones((height, width), dtype=np.uint8)
    
    output_filename = os.path.splitext(os.path.basename(image_path))[0] + '.png'
    output_path = os.path.join(output_dir, output_filename)
    
    Image.fromarray(mask).save(output_path)

def main(image_dir, json_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_files = glob.glob(os.path.join(image_dir, '*.*'))
    
    for image_file in image_files:
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        json_file = os.path.join(json_dir, base_name + '.json')
        process_image(image_file, json_file, output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成掩码图像')
    parser.add_argument('--image_dir', help='输入图像文件目录')
    parser.add_argument('--json_dir', help='输入JSON文件目录')
    parser.add_argument('--output_dir', help='输出掩码图像目录')
    args = parser.parse_args()

    main(args.image_dir, args.json_dir, args.output_dir)