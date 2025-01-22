#!/usr/bin/env bash
set -e

# Validate the input arguments
if [ $# -lt 1 ]; then
  echo "Usage: $0 <input_base_path> [<method1> <method2> ...] [--icp] [--skip-preprocess] [--resume-train <path>]"
  echo "Available methods: arkit colmap, loftr, lightglue, glomap"
  echo "Default method: arkit"
  exit 1
fi

input_base_path=$1
shift
methods=()
use_icp=false
skip_preprocess=false
resume_train=""

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
    --resume-train)
      resume_train="$2"
      shift 2
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
echo "  resume_train: ${resume_train}"

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

# Function to execute a step and log its time
execute_step() {
    local step_name=$1
    shift
    local command=$@
    local start_time=$(date +%s)
    echo "Executing step: $step_name"
    eval $command || { echo "Failed at step: $step_name"; exit 1; }
    local end_time=$(date +%s)
    log_time "$step_name" $start_time $end_time
}

# create output csv
remove_and_create_folder "${input_base_path}/../output"
output_csv="${input_base_path}/../output/duration.csv"
echo "Step,Duration (secs)" > "${output_csv}"

if [ "$skip_preprocess" = false ]; then
  echo "=== Preprocess ARkit data === "
  
  remove_and_create_folder "${input_base_path}/post"
  remove_and_create_folder "${input_base_path}/post/sparse"
  remove_and_create_folder "${input_base_path}/post/sparse/online"
  remove_and_create_folder "${input_base_path}/post/sparse/online_loop"

  execute_step "Undistort image" \
    "python arkit_utils/undistort_images/undistort_image.py --input_base ${input_base_path}"

  if [ ! -f "${input_base_path}/../scene.obj" ]; then
    echo "Error: scene.obj not found at ${input_base_path}/../scene.obj"
    exit 1
  fi

  cp ${input_base_path}/../scene.obj ${input_base_path} || { echo "Failed to copy scene.obj"; exit 1; }
  execute_step "Transform ARKit mesh to point3D" \
    "python arkit_utils/mesh_to_points3D/arkitobj2point3D.py --input_base_path ${input_base_path}"

  execute_step "Transform ARKit pose to COLMAP coordinate" \
    "python arkit_utils/arkit_pose_to_colmap.py --input_database_path ${input_base_path}"

  if [[ ! " ${methods[@]} " =~ " arkit " ]]; then
    remove_and_create_folder "${input_base_path}/post/sparse/offline" || { echo "Failed to create offline directory"; exit 1; }
    execute_step "Optimize pose using selected methods" \
      "python arkit_utils/pose_optimization/optimize_pose.py --input_database_path ${input_base_path} --methods \"${methods[@]}\""
  else
    echo "Skipping pose optimization"
  fi

  if [ "$use_icp" = true ]; then
    for method in "${methods[@]}"; do
      if [ "$method" != "arkit" ]; then
        echo "Running ICP for method: ${method}"
        execute_step "ICP registration" \
          "python arkit_utils/icp.py --base_dir \"${input_base_path}/post/sparse/offline/${method}/final\" --output_dir \"${input_base_path}/post/sparse/offline/${method}_ICP/final\""
      fi
    done
  fi

  execute_step "Prepare dataset for nerfstudio" \
    "python arkit_utils/prepare_nerfstudio_dataset.py --input_path ${input_base_path}"

  echo "Dataset preparation completed."
fi

# Training section
if [ "$use_icp" = true ]; then
  execute_step "Training nerfstudio" \
    "python arkit_utils/run_nerfstudio_dataset.py --input_path ${input_base_path} \
    --method \"${methods[@]}\" --use_icp ${resume_train:+--resume_path \"$resume_train\"}"
else
  execute_step "Training nerfstudio" \
    "python arkit_utils/run_nerfstudio_dataset.py --input_path ${input_base_path} \
    --method \"${methods[@]}\" ${resume_train:+--resume_path \"$resume_train\"}"
fi
