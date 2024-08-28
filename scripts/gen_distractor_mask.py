import json
import numpy as np
import cv2
import os

def generate_binary_mask(json_file, output_file, width=1260, height=704):
    """
    Generate a binary mask from a JSON file with polygon data.

    Args:
    - json_file (str): Path to the JSON file containing polygon data.
    - output_file (str): Path to save the generated binary mask image.
    - width (int): Width of the mask image. Default is 1260.
    - height (int): Height of the mask image. Default is 704.
    """
    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Create an empty mask
    mask = np.ones((height, width), dtype=np.uint8)

    # Draw polygons on the mask
    for item in data:
        points = np.array([[point['x'], point['y']] for point in item['content']], dtype=np.int32)
        cv2.fillPoly(mask, [points], 0)

    # Save the generated binary mask as an image file
    cv2.imwrite(output_file, mask * 255)

    print(f"Binary mask for {json_file} generated and saved as {output_file}.")

def process_multiple_jsons(json_folder, output_folder, width=1260, height=704):
    """
    Process multiple JSON files in a folder to generate binary masks.

    Args:
    - json_folder (str): Path to the folder containing JSON files.
    - output_folder (str): Path to the folder where binary mask images will be saved.
    - width (int): Width of the mask images. Default is 1260.
    - height (int): Height of the mask images. Default is 704.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each JSON file in the folder
    for json_file in os.listdir(json_folder):
        if not json_file.endswith('.json'):
            json_path = os.path.join(json_folder, json_file)
            output_path = os.path.join(output_folder, json_file.replace('.json', '.png'))
            generate_binary_mask(json_path, output_path, width, height)

    print("All JSON files processed successfully.")

# Example usage:
# process_multiple_json​⬤

def main():
    json_folder = "data/island_arkit_colmap/mask_jsons"
    output_folder = "data/island_arkit_colmap/masks"
    image_folder = "data/island_arkit_colmap/images"

    json_names = [img_name.replace(".png", ".json") for img_name in os.listdir(image_folder)]

    width=1260
    height=704

    for json_name in json_names:
        json_path = os.path.join(json_folder, json_name)
        output_path = os.path.join(output_folder, json_name.replace('.json', '.png'))
        if not os.path.exists(json_path):
            # create a white mask 
            mask = np.ones((height, width), dtype=np.uint8)
            cv2.imwrite(output_path, mask * 255)
        else:
            generate_binary_mask(json_path, output_path, width, height)

if __name__ == "__main__":
    main()
    
