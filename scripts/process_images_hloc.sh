#!/bin/bash

set -eux

# Check if the input directory is provided
if [ $# -eq 0 ]; then
    echo "Please provide the directory containing the images as an argument."
    exit 1
fi

# Set the input and output directories
INPUT_DIR="$1"
OUTPUT_DIR="${INPUT_DIR}_colmap"

# Create COLMAP dataset using nerfstudio
ns-process-data images \
    --data "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --verbose \
    --feature-type superpoint_inloc \
    --matcher-type superglue \

echo "COLMAP dataset created successfully in $OUTPUT_DIR"
