#!/bin/bash

set -eux

# 1. Prepare COLMAP dataset
# data format
# root
#   colmap
#       sparse
#           0
#               cameras.txt (or .bin)
#               images.txt (or .bin)
#               points3D.txt (or .bin)
#               scene.obj (optional, for mesh init)
#   images

# 2. Train gs-in-the-wild model with different parameter set

# vanilla wild
ns-train splatfacto-w --data "$1" \
--pipeline.model.rasterize-mode antialiased \
--pipeline.model.use-scale-regularization False  \
--pipeline.model.camera-optimizer.mode SO3xR3 \
--pipeline.model.use_mesh_initialization True \
--pipeline.model.normal_consistency_loss False \
--pipeline.model.enable_mcmc False \
--pipeline.model.noise_lr 5e5 \
--pipeline.model.cap_max 100_0000 \
--pipeline.model.stop_split_at 15000 \
colmap \
--auto_scale_poses False \
--center_method none \
--orientation_method none \


# vanilla wild + mip
# ns-train splatfacto-w --data "$1" \
# --max-num-iterations 65000 \
# --pipeline.model.rasterize-mode antialiased \
# --pipeline.model.use-scale-regularization False  \
# --pipeline.model.camera-optimizer.mode off \
# colmap \
# --auto_scale_poses True \
# --assume_colmap_world_coordinate_convention False \
# --center_method focus \
# --orientation_method vertical

# vanilla wild + PhysicsGaussian
# ns-train splatfacto-w --data "$1" \
# --max-num-iterations 65000 \
# --pipeline.model.rasterize-mode classic \
# --pipeline.model.use-scale-regularization True  \
# --pipeline.model.camera-optimizer.mode off \
# colmap \
# --auto_scale_poses True \
# --assume_colmap_world_coordinate_convention False \
# --center_method poses \
# --orientation_method up

# vanilla wild + pose_optimization
# ns-train splatfacto-w --data "$1" \
# --max-num-iterations 65000 \
# --pipeline.model.rasterize-mode classic \
# --pipeline.model.use-scale-regularization False  \
# --pipeline.model.camera-optimizer.mode SO3xR3 \
# colmap \
# --auto_scale_poses True \
# --assume_colmap_world_coordinate_convention False \
# --center_method poses \
# --orientation_method up