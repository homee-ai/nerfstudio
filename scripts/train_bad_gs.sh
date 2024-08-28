#!/bin/bash

set -eux

ns-train bad-gaussians \
--data "$1" \
--pipeline.model.camera-optimizer.mode "linear" \
--pipeline.model.camera-optimizer.num_virtual_views 3 \
--pipeline.model.use_mesh_initialization True \
--pipeline.model.rasterize-mode antialiased \
--pipeline.model.use-scale-regularization False  \
--pipeline.model.use_bilateral_grid False \
--pipeline.model.normal_consistency_loss False \
colmap \
--eval_mode "interval" --eval-interval 20
--auto_scale_poses True \
--center_method none \
--orientation_method none \