#!/bin/bash

set -eux



DIR=$(dirname "$1")

# Set output and render paths
OUTPUT_PATH="$DIR/gs_model"


# export as gaussian splatting
ns-export gaussian-splat --load-config "$1" --output-dir "$OUTPUT_PATH" --rotate-x False