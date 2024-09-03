#!/bin/bash

set -eux

ns-train splatfacto --data "$1" \
--max-num-iterations 50000 \
--pipeline.model.rasterize-mode antialiased \
--pipeline.model.use-scale-regularization True  \
--pipeline.model.camera-optimizer.mode off \
--pipeline.model.use_mesh_initialization True \
--pipeline.model.use_bilateral_grid True \
--pipeline.model.normal_consistency_loss True \
--pipeline.model.densify_grad_thresh 0.0008 \
--pipeline.model.enable_mcmc False \
--pipeline.model.noise_lr 5e5 \
--pipeline.model.cap_max 1000000 \
--pipeline.model.stop_split_at 15000 \
--pipeline.model.cull_alpha_thresh 0.005 \
colmap \
--auto_scale_poses False \
--center_method none \
--orientation_method none \
