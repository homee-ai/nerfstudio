import cv2
import numpy as np
import json
import os
import argparse
from concurrent.futures import ProcessPoolExecutor
import math
import tqdm

# Import CUDA-related modules only if available
try:
    from numba import cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

global undistort_data
undistort_data = None

class UndistortData:
    def __init__(self, crop_left: int, crop_top: int, crop_width: int, crop_height: int):
        self.crop_left = crop_left
        self.crop_top = crop_top
        self.crop_width = crop_width
        self.crop_height = crop_height

class Data:
    def __init__(self, intrinsic_matrix, intrinsic_matrix_reference_dimensions, lens_distortion_center, inverse_lens_distortion_lookup_table, lens_distortion_lookup_table):
        self.intrinsic_matrix = intrinsic_matrix
        self.intrinsic_matrix_reference_dimensions = intrinsic_matrix_reference_dimensions
        self.lens_distortion_center = lens_distortion_center
        self.inverse_lens_distortion_lookup_table = inverse_lens_distortion_lookup_table
        self.lens_distortion_lookup_table = lens_distortion_lookup_table

def readCalibrationJson(path):
    with open(path, "r") as f:
        data = json.load(f)

    intrinsic_matrix = data["calibration_data"]["intrinsic_matrix"]
    intrinsic_matrix_reference_dimensions = data["calibration_data"]["intrinsic_matrix_reference_dimensions"]
    lens_distortion_center = data["calibration_data"]["lens_distortion_center"]
    inverse_lut = data["calibration_data"]["inverse_lens_distortion_lookup_table"]
    lut = data["calibration_data"]["lens_distortion_lookup_table"]

    data = Data(intrinsic_matrix, intrinsic_matrix_reference_dimensions, lens_distortion_center, inverse_lut, lut)
    return data

def get_lens_distortion_point(point, lookup_table, distortion_center, image_size):
    radius_max_x = min(distortion_center[0], image_size[0] - distortion_center[0])
    radius_max_y = min(distortion_center[1], image_size[1] - distortion_center[1])
    radius_max = np.sqrt(radius_max_x**2 + radius_max_y**2)

    radius_point = np.sqrt(np.square(point[0] - distortion_center[0]) + np.square(point[1] - distortion_center[1]))

    magnification = lookup_table[-1]
    if radius_point < radius_max:
        relative_position = radius_point / radius_max * (len(lookup_table) - 1)
        frac = relative_position - np.floor(relative_position)
        lower_lookup = lookup_table[int(np.floor(relative_position))]
        upper_lookup = lookup_table[int(np.ceil(relative_position))]
        magnification = lower_lookup * (1.0 - frac) + upper_lookup * frac

    mapped_point = np.array([distortion_center[0] + (point[0] - distortion_center[0]) * (1.0 + magnification),
                              distortion_center[1] + (point[1] - distortion_center[1]) * (1.0 + magnification)])
    return mapped_point

def remove_black_pixels(lookup_table, distortion_center, width, height):
    roi_min_x = 0
    roi_min_y = 0
    roi_max_x = width
    roi_max_y = height

    # Find the boundary of the undistorted image
    left_max_x = float('-inf')
    right_min_x = float('inf')

    for y in range(roi_min_y, roi_max_y):
        p1 = np.array([0.5, y + 0.5])
        p1_undistorted = get_lens_distortion_point(p1, lookup_table, distortion_center, [width, height])
        left_max_x = max(left_max_x, p1_undistorted[0])

        p2 = np.array([width - 0.5, y + 0.5])
        p2_undistorted = get_lens_distortion_point(p2, lookup_table, distortion_center, [width, height])
        right_min_x = min(right_min_x, p2_undistorted[0])

    top_max_y = float('-inf')
    bottom_min_y = float('inf')

    for x in range(roi_min_x, roi_max_x):
        p1 = np.array([x + 0.5, 0.5])
        p1_undistorted = get_lens_distortion_point(p1, lookup_table, distortion_center, [width, height])
        top_max_y = max(top_max_y, p1_undistorted[1])

        p2 = np.array([x + 0.5, height - 0.5])
        p2_undistorted = get_lens_distortion_point(p2, lookup_table, distortion_center, [width, height])
        bottom_min_y = min(bottom_min_y, p2_undistorted[1])

    # use ceil for left and top, floor for right and bottom to avoid maximum black pixels
    left_max_x = math.ceil(left_max_x)
    right_min_x = math.floor(right_min_x)
    top_max_y = math.ceil(top_max_y)
    bottom_min_y = math.floor(bottom_min_y)

    new_width = right_min_x - left_max_x
    new_height = bottom_min_y - top_max_y

    return left_max_x, top_max_y, new_width, new_height

if CUDA_AVAILABLE:
    @cuda.jit
    def rectify_image_kernel(image, rectified_image, lookup_table, distortion_center_x, distortion_center_y, width, height):
        x, y = cuda.grid(2)
        if x < width and y < height:
            radius_max_x = min(distortion_center_x, width - distortion_center_x)
            radius_max_y = min(distortion_center_y, height - distortion_center_y)
            radius_max = (radius_max_x**2 + radius_max_y**2)**0.5

            radius_point = ((x - distortion_center_x)**2 + (y - distortion_center_y)**2)**0.5
            magnification = lookup_table[-1]
            if radius_point < radius_max:
                relative_position = radius_point / radius_max * (len(lookup_table) - 1)
                frac = relative_position - int(relative_position)
                lower_lookup = lookup_table[int(relative_position)]
                upper_lookup = lookup_table[int(relative_position) + 1]
                magnification = lower_lookup * (1.0 - frac) + upper_lookup * frac

            mapped_x = int(distortion_center_x + (x - distortion_center_x) * (1.0 + magnification))
            mapped_y = int(distortion_center_y + (y - distortion_center_y) * (1.0 + magnification))

            if 0 <= mapped_x < width and 0 <= mapped_y < height:
                rectified_image[y, x, 0] = image[mapped_y, mapped_x, 0]
                rectified_image[y, x, 1] = image[mapped_y, mapped_x, 1]
                rectified_image[y, x, 2] = image[mapped_y, mapped_x, 2]

def rectify_single_image(image_path, output_path, distortion_param_json_path):
    """Processes a single image with distortion correction."""
    global undistort_data
    
    image = cv2.imread(image_path)
    height, width, channel = image.shape
    rectified_image = np.zeros((height, width, channel), dtype=image.dtype)

    # read calibration data
    data = readCalibrationJson(distortion_param_json_path)
    lookup_table = data.inverse_lens_distortion_lookup_table
    distortion_center = data.lens_distortion_center
    reference_dimensions = data.intrinsic_matrix_reference_dimensions
    ratio_x = width / reference_dimensions[0]
    ratio_y = height / reference_dimensions[1]
    distortion_center[0] = distortion_center[0] * ratio_x
    distortion_center[1] = distortion_center[1] * ratio_y

    if CUDA_AVAILABLE:
        threads_per_block = (16, 16)
        blocks_per_grid_x = int(np.ceil(width / threads_per_block[0]))
        blocks_per_grid_y = int(np.ceil(height / threads_per_block[1]))
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        d_image = cuda.to_device(image)
        d_rectified_image = cuda.to_device(rectified_image)
        d_lookup_table = cuda.to_device(np.array(lookup_table, dtype=np.float32))
        d_distortion_center_x = distortion_center[0]
        d_distortion_center_y = distortion_center[1]

        rectify_image_kernel[blocks_per_grid, threads_per_block](d_image, d_rectified_image, d_lookup_table, d_distortion_center_x, d_distortion_center_y, width, height)
        rectified_image = d_rectified_image.copy_to_host()
    else:
        for i in range(width):
            for j in range(height):
                rectified_index = np.array([i, j])
                original_index = get_lens_distortion_point(
                    rectified_index, lookup_table, distortion_center, [width, height])

                if (original_index[0] < 0 or original_index[0] >= width or original_index[1] < 0 or original_index[1] >= height):
                    continue

                rectified_image[j, i] = image[int(original_index[1]), int(original_index[0])]

    if undistort_data is None:
        lookup_table = np.array(data.lens_distortion_lookup_table, dtype=np.float32)
        distortion_center = np.array(distortion_center, dtype=np.float32)
        crop_left, crop_top, crop_width, crop_height = remove_black_pixels(lookup_table, distortion_center, width, height)
        undistort_data = UndistortData(crop_left, crop_top, crop_width, crop_height)
    else:
        crop_left = undistort_data.crop_left
        crop_top = undistort_data.crop_top
        crop_width = undistort_data.crop_width
        crop_height = undistort_data.crop_height

    crop_image = rectified_image[crop_top:crop_top+crop_height, crop_left:crop_left+crop_width]
    cv2.imwrite(output_path, crop_image)

def rectify_all_image(image_folder_path, distortion_param_json_path, output_image_folder_path):
    if CUDA_AVAILABLE:
        for filename in tqdm.tqdm(os.listdir(image_folder_path), desc="Processing:"):
            image_path = os.path.join(image_folder_path, filename)
            output_path = os.path.join(output_image_folder_path, filename)
            rectify_single_image(image_path, output_path, distortion_param_json_path)
    else :
        print("Using CPU, it may take a long time")
        with ProcessPoolExecutor() as executor:
            for filename in os.listdir(image_folder_path):
                image_path = os.path.join(image_folder_path, filename)
                output_path = os.path.join(output_image_folder_path, filename)
                executor.submit(
                    rectify_single_image, image_path, output_path, distortion_param_json_path)

def rectified_intrinsic(input_path, output_path):
    num_intrinsic = 0
    with open(input_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_intrinsic += 1

    camera_ids = np.empty((num_intrinsic, 1))
    widths = np.empty((num_intrinsic, 1))
    heights = np.empty((num_intrinsic, 1))
    paramss = np.empty((num_intrinsic, 4))

    count = 0
    with open(input_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = undistort_data.crop_width
                height = undistort_data.crop_height
                params = np.array(tuple(map(float, elems[4:])))
                params[2] = params[2] - undistort_data.crop_left
                params[3] = params[3] - undistort_data.crop_top

                camera_ids[count] = camera_id
                widths[count] = width
                heights[count] = height
                paramss[count] = params

                count = count+1

    with open(output_path, "w") as f:
        for i in range(num_intrinsic):
            line = str(int(camera_ids[i])) + " " + "PINHOLE" + " " + str(int(widths[i]))+ " " + str(int(heights[i]))+ " " + str(paramss[i][0]) + " " + str(paramss[i][1])+ " " + str(paramss[i][2])+ " " +  str(paramss[i][3])
            f.write(line  + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="undistort ARKit image using distortion params get from AVfoundation")
    parser.add_argument("--input_base", type=str)

    args = parser.parse_args()
    base_folder_path = args.input_base
    input_image_folder_path = base_folder_path + "/distort_images"
    distortion_param_json_path = base_folder_path + "/sparse/0/calibration.json"
    output_image_folder_path = base_folder_path + "/post/images/"
    input_camera = base_folder_path + "/sparse/0/distort_cameras.txt"

    if not os.path.exists(output_image_folder_path):
        os.makedirs(output_image_folder_path)

    rectify_all_image(input_image_folder_path, distortion_param_json_path, output_image_folder_path)
    
    output_cameras = [
        base_folder_path + "/post/sparse/online/cameras.txt",
        base_folder_path + "/post/sparse/online_loop/cameras.txt"
    ]

    for output_camera in output_cameras:
        rectified_intrinsic(input_camera, output_camera)
