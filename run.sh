#!/usr/bin/env bash
set -e

# Add logging configuration at the top
LOG_PREFIX="[3DGS]"
LOG_INDENT="  "
LOG_INDENT_LEVEL=0
COLOR_RED='\033[0;31m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[1;33m'
COLOR_BLUE='\033[0;34m'
COLOR_RESET='\033[0m'

# Only use colors if connected to a terminal
if [ -t 1 ]; then
    USE_COLORS=true
else
    USE_COLORS=false
fi

# Helper functions - MOVED TO TOP
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
    echo "================================================"
    echo "Executing step: $step_name"
    echo "================================================"
    eval $command || { echo "Failed at step: $step_name"; exit 1; }
    local end_time=$(date +%s)
    log_time "$step_name" $start_time $end_time
}

# Logging functions
log() {
    local level=$1; shift
    local color=$1; shift
    local message=$*
    local timestamp=$(date +"%Y-%m-%d %T")
    local indent=$(printf "%${LOG_INDENT_LEVEL}s" "")
    
    if $USE_COLORS; then
        echo -e "${color}${LOG_PREFIX} ${timestamp} [${level}]${indent} ${message}${COLOR_RESET}"
    else
        echo "${LOG_PREFIX} ${timestamp} [${level}]${indent} ${message}"
    fi
}

log_info() {
    log "INFO" "$COLOR_GREEN" "$@"
}

log_debug() {
    if [ "$VERBOSE" = true ]; then
        log "DEBUG" "$COLOR_BLUE" "$@"
    fi
}

log_warn() {
    log "WARN" "$COLOR_YELLOW" "$@"
}

log_error() {
    log "ERROR" "$COLOR_RED" "$@"
    exit 1
}

# Error handler with stack trace
error_handler() {
    local exit_code=$?
    local line_number=$1
    local command=$2
    
    log_error "Error occurred in ${BASH_SOURCE[1]} at line ${line_number} (exit code ${exit_code})"
    echo "Stack trace:"
    for ((i=0; i < ${#FUNCNAME[@]}; i++)); do
        echo "  ${BASH_SOURCE[$i+1]}:${BASH_LINENO[$i]} - ${FUNCNAME[$i]}"
    done
    exit $exit_code
}

# Set error trap
trap 'error_handler $LINENO "$BASH_COMMAND"' ERR

# Validate the input arguments
if [ $# -lt 1 ]; then
  echo "Usage: $0 <input_base_path> [<method1> <method2> ...] [--icp] [--skip-preprocess] [--resume-train <path>] [--chunked-train] [--is-adaptive] [--chunk-size <size>] [--chunk-iterations <iters>] [--final-iterations <iters>] [--partition-method <method>] [--m-region <size>] [--n-region <size>] [--n-clusters <clusters>] [--skip-chunk-training] [--no-filter-gaussians]"
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
use_chunked_train=false
chunk_size=500
chunk_iterations=10000
final_iterations=30000
partition_method="chunk_size"
m_region=2
n_region=2
n_clusters=4
skip_chunk_training=false
filter_gaussians=true
VERBOSE=false
QUIET=false
is_adaptive=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -v|--verbose)
      VERBOSE=true
      shift
      ;;
    -q|--quiet)
      QUIET=true
      shift
      ;;
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
    --chunked-train)
      use_chunked_train=true
      shift
      ;;
    --chunk-size)
      chunk_size="$2"
      shift 2
      ;;
    --chunk-iterations)
      chunk_iterations="$2"
      shift 2
      ;;
    --final-iterations)
      final_iterations="$2"
      shift 2
      ;;
    --partition-method)
      partition_method="$2"
      shift 2
      ;;
    --m-region)
      m_region="$2"
      shift 2
      ;;
    --n-region)
      n_region="$2"
      shift 2
      ;;
    --n-clusters)
      n_clusters="$2"
      shift 2
      ;;
    --skip-chunk-training)
      skip_chunk_training=true
      shift
      ;;
    --no-filter-gaussians)
      filter_gaussians=false
      shift
      ;;
    --is-adaptive)
      is_adaptive=true
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
log_info "Configuration:"
log_info "  input_base_path: ${input_base_path}"
log_info "  methods: ${methods[@]}"
log_info "  use_icp: ${use_icp}"
log_info "  skip_preprocess: ${skip_preprocess}"
log_info "  resume_train: ${resume_train}"
log_info "  use_chunked_train: ${use_chunked_train}"
log_info "  chunk_size: ${chunk_size}"
log_info "  chunk_iterations: ${chunk_iterations}"
log_info "  final_iterations: ${final_iterations}"
log_info "  partition_method: ${partition_method}"
log_info "  m_region: ${m_region}"
log_info "  n_region: ${n_region}"
log_info "  n_clusters: ${n_clusters}"
log_info "  skip_chunk_training: ${skip_chunk_training}"
log_info "  filter_gaussians: ${filter_gaussians}"
log_info "  is_adaptive: ${is_adaptive}"

# create output csv
remove_and_create_folder "${input_base_path}/../output"
output_csv="${input_base_path}/../output/duration.csv"
echo "Step,Duration (secs)" > "${output_csv}"

# New adaptive training logic moved after preprocessing
if [ "$skip_preprocess" = false ]; then
  echo "=== Preprocess ARkit data === "
  
  remove_and_create_folder "${input_base_path}/post"
  remove_and_create_folder "${input_base_path}/post/sparse"
  remove_and_create_folder "${input_base_path}/post/sparse/online"
  remove_and_create_folder "${input_base_path}/post/sparse/online_loop"

  execute_step "Undistort image" \
    "python arkit_utils/undistort_images/undistort_image.py --input_base ${input_base_path}"

  if [ ! -f "${input_base_path}/../scene.obj" ]; then
    log_error "scene.obj not found at ${input_base_path}/../scene.obj"
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

# Set output_root path based on processed data location
output_root="$(dirname "${input_base_path}")/$(basename "$(dirname "${input_base_path}")")_nerfstudio"

# Adaptive training decision
if [ "$is_adaptive" = true ]; then
    # Only check if not resuming training
    if [ -z "$resume_train" ]; then
        # Verify images directory exists
        if [ ! -d "${output_root}/images" ]; then
            log_error "Images directory ${output_root}/images not found"
            exit 1
        fi
        
        # Count images only if directory exists
        num_images=$(find "${output_root}/images" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) | wc -l)
        adaptive_threshold=$((chunk_size * 2))
        
        if [ "$num_images" -gt "$adaptive_threshold" ]; then
            use_chunked_train=true
            log_info "Enabling chunked training automatically (${num_images} images > 2*${chunk_size})"
        else
            use_chunked_train=false
            log_info "Using normal training (${num_images} images <= 2*${chunk_size})"
        fi
    fi
fi

# Training section
if [ "$use_icp" = true ]; then
  if [ "$use_chunked_train" = true ]; then
    execute_step "Training nerfstudio with chunks" \
      "python arkit_utils/run_nerfstudio_dataset_chunks.py \
      --input_path ${input_base_path} \
      --method ${methods[@]} \
      --use_icp \
      --chunk_size ${chunk_size} \
      --chunk_iterations ${chunk_iterations} \
      --final_iterations ${final_iterations} \
      --partition_method ${partition_method} \
      --m_region ${m_region} \
      --n_region ${n_region} \
      --n_clusters ${n_clusters} \
      $([ "$skip_chunk_training" = true ] && echo "--skip-chunk-training") \
      $([ "$filter_gaussians" = true ] && echo "--filter-gaussians")"
  else
    execute_step "Training nerfstudiExecuting step: Training nerfstudioo" \
      "python arkit_utils/run_nerfstudio_dataset.py --input_path ${input_base_path} \
      --method \"${methods[@]}\" --use_icp ${resume_train:+--resume_path \"$resume_train\"}"
  fi
else
  if [ "$use_chunked_train" = true ]; then
    execute_step "Training nerfstudio with chunks" \
      "python arkit_utils/run_nerfstudio_dataset_chunks.py \
      --input_path ${input_base_path} \
      --method ${methods[@]} \
      --chunk_size ${chunk_size} \
      --chunk_iterations ${chunk_iterations} \
      --final_iterations ${final_iterations} \
      --partition_method ${partition_method} \
      --m_region ${m_region} \
      --n_region ${n_region} \
      --n_clusters ${n_clusters} \
      $([ "$skip_chunk_training" = true ] && echo "--skip-chunk-training") \
      $([ "$filter_gaussians" = true ] && echo "--filter-gaussians")"
  else
    execute_step "Training nerfstudio" \
      "python arkit_utils/run_nerfstudio_dataset.py --input_path ${input_base_path} \
      --method \"${methods[@]}\" ${resume_train:+--resume_path \"$resume_train\"}"
  fi
fi
