#!/bin/bash

set -eux

ns-train splatfacto --data "$1" \
--pipeline.model.rasterize-mode antialiased \
--pipeline.model.use-scale-regularization False  \
--pipeline.model.camera-optimizer.mode SO3xR3 \
--pipeline.model.use_mesh_initialization True \
--pipeline.model.use_bilateral_grid True \
--pipeline.model.normal_consistency_loss False \
--pipeline.model.enable_mcmc False \
--pipeline.model.noise_lr 5e5 \
--pipeline.model.cap_max 1000000 \
--pipeline.model.stop_split_at 15000 \
colmap \
--auto_scale_poses False \
--center_method none \
--orientation_method none \
