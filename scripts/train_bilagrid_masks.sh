#!/bin/bash

set -eux

ns-train splatfacto --data "$1" \
--pipeline.model.rasterize-mode antialiased \
--pipeline.model.use-scale-regularization True  \
--pipeline.model.camera-optimizer.mode SO3xR3 \
--pipeline.model.use_mesh_initialization False \
--pipeline.model.use_bilateral_grid True \
colmap \
--auto_scale_poses False \
--masks_path masks