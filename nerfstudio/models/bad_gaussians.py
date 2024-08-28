"""
BAD-Gaussians model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import open3d as o3d
import open3d.core as o3c
import math
import torch
from gsplat.cuda_legacy._torch_impl import quat_to_rotmat

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from nerfstudio.model_components.lib_bilagrid import BilateralGrid, slice, total_variation_loss
from nerfstudio.utils.general_utils import MeshPointCloud, rot_to_quat_batch
from nerfstudio.model_components import renderers
from nerfstudio.data.scene_box import SceneBox


from nerfstudio.cameras.camera_optimizers import (
    BadCameraOptimizer,
    BadCameraOptimizerConfig,
    TrajSamplingMode,
)
from nerfstudio.model_components.losses import EdgeAwareVariationLoss

def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def resize_image(image: torch.Tensor, d: int):
    """
    Downscale images using the same 'area' method in opencv

    :param image shape [H, W, C]
    :param d downscale factor (must be 2, 4, 8, etc.)

    return downscaled image in shape [H//d, W//d, C]
    """
    import torch.nn.functional as tf

    image = image.to(torch.float32)
    weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)


# @torch_compile()
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat


@dataclass
class BadGaussiansModelConfig(SplatfactoModelConfig):
    """BAD-Gaussians Model config"""

    _target: Type = field(default_factory=lambda: BadGaussiansModel)
    """The target class to be instantiated."""

    rasterize_mode: Literal["classic", "antialiased"] = "antialiased"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    Refs:
    1. https://github.com/nerfstudio-project/gsplat/pull/117
    2. https://github.com/nerfstudio-project/nerfstudio/pull/2888
    3. Yu, Zehao, et al. "Mip-Splatting: Alias-free 3D Gaussian Splatting." arXiv preprint arXiv:2311.16493 (2023).
    """

    camera_optimizer: BadCameraOptimizerConfig = field(default_factory=BadCameraOptimizerConfig)
    """Config of the camera optimizer to use"""

    cull_alpha_thresh: float = 0.005
    """Threshold for alpha to cull gaussians. Default: 0.1 in splatfacto, 0.005 in splatfacto-big."""

    densify_grad_thresh: float = 4e-4
    """[IMPORTANT] Threshold for gradient to densify gaussians. Default: 4e-4. Tune it smaller with complex scenes."""

    continue_cull_post_densification: bool = False
    """Whether to continue culling after densification. Default: True in splatfacto, False in splatfacto-big."""

    resolution_schedule: int = 250
    """training starts at 1/d resolution, every n steps this is doubled.
    Default: 250. Use 3000 with high resolution images (e.g. higher than 1920x1080).
    """

    num_downscales: int = 0
    """at the beginning, resolution is 1/2^d, where d is this number. Default: 0. Use 2 with high resolution images."""

    enable_absgrad: bool = False
    """Whether to enable absgrad for gaussians. (It affects param tuning of densify_grad_thresh)
    Default: False. Ref: (https://github.com/nerfstudio-project/nerfstudio/pull/3113)
    """

    tv_loss_lambda: Optional[float] = None
    """weight of total variation loss"""

    use_mesh_initialization: bool = False
    """If enable, will try to initialize gaussain from mesh"""

    use_bilateral_grid: bool = False
    """If True, use bilateral grid to handle the ISP changes in the image space"""
    
    gird_shape: Tuple[int, int, int] = (16, 16, 8)
    """Shape of the bilateral grid (X, Y, W)"""

    normal_consistency_loss: bool = False
    """If True, use normal consistency loss"""
    normal_consistency_lambda: float = 0.05


class BadGaussiansModel(SplatfactoModel):
    """BAD-Gaussians Model

    Args:
        config: configuration to instantiate model
    """

    config: BadGaussiansModelConfig
    camera_optimizer: BadCameraOptimizer

    def __init__(
        self,
        config: BadGaussiansModelConfig,
        scene_box: SceneBox,
        num_train_data: int,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> None:
        super().__init__(config=config, scene_box=scene_box, num_train_data=num_train_data, seed_points=seed_points, **kwargs)
        # Scale densify_grad_thresh by the number of virtual views
        self.config.densify_grad_thresh /= self.config.camera_optimizer.num_virtual_views
        # (Experimental) Total variation loss
        self.tv_loss = EdgeAwareVariationLoss(in1_nc=3)

    def populate_modules(self) -> None:
        super().populate_modules()
        self.camera_optimizer: BadCameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        if self.config.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                num=self.num_train_data,
                grid_X=self.config.gird_shape[0],
                grid_Y=self.config.gird_shape[1],
                grid_W=self.config.gird_shape[2],
            )

    def forward(
            self,
            camera: Cameras,
            mode: TrajSamplingMode = "uniform",
    ) -> Dict[str, Union[torch.Tensor, List]]:
        return self.get_outputs(camera, mode)

    def get_outputs(
        self,
        camera: Cameras,
        mode: TrajSamplingMode = "uniform",
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Camera and returns a dictionary of outputs.

        Args:
            camera: Input camera. This camera should have all the needed information to compute the outputs.
            mode: The trajectory sampling mode for the camera optimizer.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"

        is_training = self.training and torch.is_grad_enabled()

        # BAD-Gaussians: get virtual cameras
        virtual_cameras = self.camera_optimizer.apply_to_camera(camera, mode)

        background = self._get_background_color(is_training)

        if self.crop_box is not None and not is_training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                rgb = background.repeat(int(camera.height.item()), int(camera.width.item()), 1)
                depth = background.new_ones(*rgb.shape[:2], 1) * 10
                accumulation = background.new_zeros(*rgb.shape[:2], 1)
                return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}
        else:
            crop_ids = None

        camera_downscale = self._get_downscale_factor()

        for cam in virtual_cameras:
            cam.rescale_output_resolution(1 / camera_downscale)

        # BAD-Gaussians: render virtual views
        virtual_views_rgb = []
        virtual_views_alpha = []
        for cam in virtual_cameras:
            viewmat = self._get_viewmat(cam).unsqueeze(0)
            W, H = int(cam.width.item()), int(cam.height.item())
            self.last_size = (H, W)

            means_crop, colors_crop, opacities_crop, scales_crop, quats_crop = self._get_crop_gaussians(crop_ids)

            BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
            render_mode = "RGB+ED" if self.config.output_depth_during_training or not self.training else "RGB"
            sh_degree_to_use = self._get_sh_degree_to_use(colors_crop)

            rade = self.config.normal_consistency_loss and (self.step >= 5000)
            render, alpha, expected_depths, median_depths, render_normals, info = rasterization(
                means=means_crop,
                quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
                scales=torch.exp(scales_crop),
                opacities=torch.sigmoid(opacities_crop).squeeze(-1),
                colors=colors_crop,
                viewmats=viewmat,  # [1, 4, 4]
                Ks=cam.get_intrinsics_matrices().cuda(),  # [1, 3, 3]
                width=W,
                height=H,
                tile_size=BLOCK_WIDTH,
                packed=False,
                near_plane=0.01,
                far_plane=1e10,
                render_mode=render_mode,
                sh_degree=sh_degree_to_use,
                sparse_grad=False,
                absgrad=True,
                rasterize_mode=self.config.rasterize_mode,
                require_rade=rade
            )

            if self.training and info["means2d"].requires_grad:
                info["means2d"].retain_grad()
            self.xys = info["means2d"]  # [1, N, 2]
            self.radii = info["radii"][0]  # [N]
            alpha = alpha[:, ...]

            rgb = render[:, ..., :3] + (1 - alpha) * background
            rgb = torch.clamp(rgb, 0.0, 1.0)

            # apply bilateral grid
            if self.config.use_bilateral_grid and self.training:
                if cam.metadata is not None and "cam_idx" in cam.metadata:
                    rgb = self._apply_bilateral_grid(rgb, cam.metadata["cam_idx"], H, W)

            virtual_views_rgb.append(rgb)
            virtual_views_alpha.append(alpha)

        rgb = torch.stack(virtual_views_rgb, dim=0).mean(dim=0)
        alpha = torch.stack(virtual_views_alpha, dim=0).mean(dim=0)

        depth_im = self._compute_depth(render, alpha) if render_mode == "RGB+ED" else None

        normal_loss = self._compute_normal_loss(expected_depths, median_depths, render_normals, cam) if rade else torch.tensor(0.0).to(self.device)

        return {
            "rgb": rgb.squeeze(0),
            "depth": depth_im,
            "accumulation": alpha.squeeze(0),
            "background": background,
            "normal_loss": normal_loss,
        }

    def _get_background_color(self, is_training):
        if is_training:
            if self.config.background_color == "random":
                return torch.rand(3, device=self.device)
            elif self.config.background_color == "white":
                return torch.ones(3, device=self.device)
            elif self.config.background_color == "black":
                return torch.zeros(3, device=self.device)
            else:
                return self.background_color.to(self.device)
        else:
            return renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device) if renderers.BACKGROUND_COLOR_OVERRIDE is not None else self.background_color.to(self.device)

    def _get_viewmat(self, cam):
        R = cam.camera_to_worlds[0, :3, :3]  # 3 x 3
        T = cam.camera_to_worlds[0, :3, 3:4]  # 3 x 1
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        R = R @ R_edit
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        return viewmat

    def _get_crop_gaussians(self, crop_ids):
        if crop_ids is not None:
            return (
                self.means[crop_ids],
                torch.cat((self.features_dc[crop_ids][:, None, :], self.features_rest[crop_ids]), dim=1),
                self.opacities[crop_ids],
                self.scales[crop_ids],
                self.quats[crop_ids],
            )
        else:
            return (
                self.means,
                torch.cat((self.features_dc[:, None, :], self.features_rest), dim=1),
                self.opacities,
                self.scales,
                self.quats,
            )

    def _get_sh_degree_to_use(self, colors_crop):
        if self.config.sh_degree > 0:
            return min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            return None

    def _apply_bilateral_grid(self, rgb, cam_idx, H, W):
        grid = self.bil_grids.grids[cam_idx]
        rgb = rgb.permute(0, 3, 1, 2)  # [1, H, W, 3] -> [1, 3, H, W]
        rgb = slice(rgb, grid)
        rgb = rgb.permute(0, 2, 3, 1)  # [1, 3, H, W] -> [1, H, W, 3]
        return rgb

    def _compute_depth(self, render, alpha):
        depth_im = render[:, ..., 3:4]
        depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        return depth_im

    def _compute_normal_loss(self, expected_depths, median_depths, render_normals, cam):
        K = cam.get_intrinsics_matrices().cuda()
        W, H = int(cam.width.item()), int(cam.height.item())
        grid_x, grid_y = torch.meshgrid(torch.arange(W)+0.5, torch.arange(H)+0.5, indexing='xy')
        points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(1, -1, 3).float().cuda()
        rays_d = points @ torch.linalg.inv(K.transpose(2,1)) # 1, M, 3
        points_e = expected_depths.reshape(K.shape[0],-1,1) * rays_d
        points_m = median_depths.reshape(K.shape[0],-1,1) * rays_d
        points_e = points_e.reshape_as(render_normals)
        points_m = points_m.reshape_as(render_normals)
        normal_map_e = torch.zeros_like(points_e)
        dx = points_e[...,2:, 1:-1,:] - points_e[...,:-2, 1:-1,:]
        dy = points_e[...,1:-1, 2:,:] - points_e[...,1:-1, :-2,:]
        normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
        normal_map_e[...,1:-1, 1:-1, :] = normal_map
        normal_map_m = torch.zeros_like(points_m)
        dx = points_m[...,2:, 1:-1,:] - points_m[...,:-2, 1:-1,:]
        dy = points_m[...,1:-1, 2:,:] - points_m[...,1:-1, :-2,:]
        normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
        normal_map_m[...,1:-1, 1:-1, :] = normal_map
        normal_error_map_e = (1 - (render_normals * normal_map_e).sum(dim=-1))
        normal_error_map_m = (1 - (render_normals * normal_map_m).sum(dim=-1))
        normal_loss = self.config.normal_consistency_lambda * (0.4 * normal_error_map_e.mean() + 0.6 * normal_error_map_m.mean())
        return normal_loss.to(self.device)
    
    def after_train(self, step: int):
        assert step == self.step
        # to save some training time, we no longer need to update those stats post refinement
        if self.step >= self.config.stop_split_at:
            return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            # BAD-Gaussians: use absgrad if enabled
            if self.config.enable_absgrad:
                assert self.xys.absgrad is not None  # type: ignore
                grads = self.xys.absgrad[0][visible_mask].norm(dim=-1)  # type: ignore
            else:
                assert self.xys.grad is not None
                grads = self.xys.grad[0][visible_mask].norm(dim=-1)
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(self.num_points, device=self.device, dtype=torch.float32)
                self.vis_counts = torch.ones(self.num_points, device=self.device, dtype=torch.float32)
            assert self.vis_counts is not None
            self.vis_counts[visible_mask] += 1
            self.xys_grad_norm[visible_mask] += grads
            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )
            
    @torch.no_grad()
    def get_outputs_for_camera(
            self,
            camera: Cameras,
            obb_box: Optional[OrientedBox] = None,
            mode: TrajSamplingMode = "mid",
    ) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        # BAD-Gaussians: camera.to(device) will drop metadata
        metadata = camera.metadata
        camera = camera.to(self.device)
        camera.metadata = metadata
        outs = self.get_outputs(camera, mode=mode)
        return outs  # type: ignore

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        # Add total variation loss
        rgb = outputs["rgb"].permute(2, 0, 1).unsqueeze(0)  # H, W, 3 to 1, 3, H, W
        if self.config.tv_loss_lambda is not None:
            loss_dict["tv_loss"] = self.tv_loss(rgb) * self.config.tv_loss_lambda
        if self.config.use_bilateral_grid:
            loss_dict["bilagrid_loss"] = 10 * total_variation_loss(self.bil_grids.grids)
        if self.config.normal_consistency_loss:
            loss_dict["normal_loss"] = outputs["normal_loss"]
        # Add loss from camera optimizer
        self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        metrics_dict = super().get_metrics_dict(outputs, batch)
        # Add metrics from camera optimizer
        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        param_groups = super().get_param_groups()
        self.camera_optimizer.get_param_groups(param_groups=param_groups)
        if self.config.use_bilateral_grid:
            param_groups["bilateral_grid"] = list(self.bil_grids.parameters())
        return param_groups
