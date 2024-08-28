#!/bin/bash

set -eux

ns-train splatfacto --data "$1" \
--pipeline.model.rasterize-mode antialiased \
--pipeline.model.use-scale-regularization False \
--pipeline.model.camera-optimizer.mode SO3xR3 \
--pipeline.model.use_mesh_initialization True \
colmap \
--auto_scale_poses True \
--center_method none \
--orientation_method none \
--masks_path masks


