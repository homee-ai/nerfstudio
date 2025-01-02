#!/usr/bin/env bash
set -e

# Validate the input arguments
if [ $# -lt 1 ]; then
  echo "Usage: $0 <input_base_path> [<method1> <method2> ...] [--icp] [--skip-preprocess]"
  echo "Available methods: arkit colmap, loftr, lightglue, glomap"
  echo "Default method: arkit"
  exit 1
fi

input_base_path=$1
shift
methods=()
use_icp=false
skip_preprocess=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --icp)
      use_icp=true
      shift
      ;;
    --skip-preprocess)
      skip_preprocess=true
      shift
      ;;
    *)
      methods+=("$1")
      shift
      ;;
  esac
done

# If no methods specified, default to arkit
if [ ${#methods[@]} -eq 0 ]; then
  methods=("arkit")
fi

# Print configuration
echo "Configuration:"
echo "  input_base_path: ${input_base_path}"
echo "  methods: ${methods[@]}"
echo "  use_icp: ${use_icp}"
echo "  skip_preprocess: ${skip_preprocess}"

remove_and_create_folder() {
  if [ -d "$1" ]; then
    rm -rf "$1"
  fi
  mkdir -p "$1"
}

# Function to log time spent on each step
log_time() {
  local step_name=$1
  local start_time=$2
  local end_time=$3
  local duration=$((end_time - start_time))
  echo "${step_name},${duration}" >> "${output_csv}"
}

# create output csv
remove_and_create_folder "${input_base_path}/../output"
output_csv="${input_base_path}/../output/duration.csv"
# Create output CSV file if it doesn't exist
if [ ! -f "${output_csv}" ]; then
    touch "${output_csv}"
fi


echo "Step,Duration (secs)" > "${output_csv}"

if [ "$skip_preprocess" = false ]; then
  echo "=== Preprocess ARkit data === "
  
  remove_and_create_folder "${input_base_path}/post"
  remove_and_create_folder "${input_base_path}/post/sparse"
  remove_and_create_folder "${input_base_path}/post/sparse/online"
  remove_and_create_folder "${input_base_path}/post/sparse/online_loop"

  start_time=$(date +%s)
  echo "1. Undistort image using AVFoundation calibration data"
  python arkit_utils/undistort_images/undistort_image.py --input_base ${input_base_path} || { echo "Failed to undistort images"; exit 10; }
  end_time=$(date +%s)
  log_time "Undistort image using AVFoundation calibration data" $start_time $end_time

  start_time=$(date +%s)
  echo "2. Transform ARKit mesh to point3D"
  if [ ! -f "${input_base_path}/../scene.obj" ]; then
    echo "Error: scene.obj not found at ${input_base_path}/../scene.obj"
    exit 1
  fi
  cp ${input_base_path}/../scene.obj ${input_base_path} || { echo "Failed to copy scene.obj"; exit 20; } || { echo "Failed to copy scene.obj"; exit 20; }
  python arkit_utils/mesh_to_points3D/arkitobj2point3D.py --input_base_path ${input_base_path} || { echo "Failed to transform ARKit mesh to point3D"; exit 21; }
  end_time=$(date +%s)
  log_time "Transform ARKit mesh to point3D" $start_time $end_time

  start_time=$(date +%s)
  echo "3. Transform ARKit pose to COLMAP coordinate"
  python arkit_utils/arkit_pose_to_colmap.py --input_database_path ${input_base_path} || { echo "Failed to transform ARKit pose to COLMAP coordinate"; exit 30; }
  end_time=$(date +%s)
  log_time "Transform ARKit pose to COLMAP coordinate" $start_time $end_time

  start_time=$(date +%s)
  echo "4. Optimize pose using selected methods"
  if [[ ! " ${methods[@]} " =~ " arkit " ]]; then
    remove_and_create_folder "${input_base_path}/post/sparse/offline" || { echo "Failed to create offline directory"; exit 40; }
    python arkit_utils/pose_optimization/optimize_pose.py --input_database_path ${input_base_path} --methods "${methods[@]}" || { echo "Failed to optimize pose"; exit 41; }
  else
    echo "Skipping pose optimization"
  fi
  end_time=$(date +%s)
  log_time "Optimize pose using selected methods" $start_time $end_time

  if [ "$use_icp" = true ]; then
    start_time=$(date +%s)
    echo "4.5 ICP registration"
    for method in "${methods[@]}"; do
      if [ "$method" != "arkit" ]; then
        echo "Running ICP for method: ${method}"
        python arkit_utils/icp.py \
          --base_dir "${input_base_path}/post/sparse/offline/${method}/final" \
          --output_dir "${input_base_path}/post/sparse/offline/${method}_ICP/final" || { echo "Failed ICP for method: ${method}"; exit 45; }
      fi
    done
    end_time=$(date +%s)
    log_time "ICP registration" $start_time $end_time
  fi


  start_time=$(date +%s)
  echo "5. Prepare dataset for nerfstudio"
  python arkit_utils/prepare_nerfstudio_dataset.py --input_path ${input_base_path} || { echo "Failed to prepare dataset for nerfstudio"; exit 50; }
  end_time=$(date +%s)
  log_time "Prepare dataset for nerfstudio" $start_time $end_time

  echo "Dataset preparation completed."
fi

start_time=$(date +%s)
echo "6. Start training nerfstudio"
if [ "$use_icp" = true ]; then
  python arkit_utils/run_nerfstudio_dataset.py --input_path ${input_base_path} --method "${methods[@]}" --use_icp || { echo "Failed to start training nerfstudio with ICP"; exit 60; }
else
  python arkit_utils/run_nerfstudio_dataset.py --input_path ${input_base_path} --method "${methods[@]}" || { echo "Failed to start training nerfstudio"; exit 61; }
fi
end_time=$(date +%s)
log_time "Training nerfstudio" $start_time $end_time
