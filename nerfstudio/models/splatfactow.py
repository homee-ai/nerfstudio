"""
Gaussian Splatting Model in the Wild implementation in nerfstudio.
https://kevinxu02.github.io/gsw.github.io/
"""

from dataclasses import dataclass, field
from typing import Dict, List, Type, Union

import torch
from torch.nn import Parameter

from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from nerfstudio.fields.splatfactow_field import SplatfactoWField
from nerfstudio.cameras.cameras import Cameras


@dataclass
class SplatfactoWModelConfig(SplatfactoModelConfig):
    """SplatfactoW Model Config, extending Splatfacto with additional features"""

    _target: Type = field(default_factory=lambda: SplatfactoWModel)

    appearance_embed_dim: int = 48
    """Dimension of the appearance embedding, if 0, no appearance embedding is used"""

    appearance_features_dim: int = 72
    """Dimension of the appearance feature"""

    enable_alpha_loss: bool = True
    """Whether to enable the alpha loss for punishing gaussians from occupying background space"""

    enable_robust_mask: bool = True
    """Whether to enable robust mask for calculating the loss"""

    robust_mask_percentage: tuple = (0.0, 0.40)
    """The percentage of the entire image to mask out for robust loss calculation"""

    robust_mask_reset_interval: int = 6000
    """The interval to reset the mask"""

    never_mask_upper: float = 0.4
    """Whether to mask out the upper part of the image, which is usually the sky"""

    start_robust_mask_at: int = 6000
    """The step to start masking"""


class SplatfactoWModel(SplatfactoModel):
    """Extension of Splatfacto with additional features for appearance modeling"""

    config: SplatfactoWModelConfig

    def populate_modules(self):
        # Initialize base model first
        super().populate_modules()
        
        # Add appearance embedding
        assert self.config.appearance_embed_dim > 0
        self.appearance_embeds = torch.nn.Embedding(self.num_train_data, self.config.appearance_embed_dim)
        
        # Add color neural network
        self.color_nn = SplatfactoWField(
            appearance_embed_dim=self.config.appearance_embed_dim,
            appearance_features_dim=self.config.appearance_features_dim,
            implementation=self.config.implementation,
        )

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        # Get appearance embedding for this camera
        if camera.metadata is not None and "cam_idx" in camera.metadata:
            cam_idx = camera.metadata["cam_idx"]
            appearance_embed = self.appearance_embeds(torch.tensor(cam_idx, device=self.device))
        else:
            appearance_embed = self.appearance_embeds(torch.tensor(0, device=self.device))

        # Get base outputs
        outputs = super().get_outputs(camera)
        
        # Process colors through appearance network
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            appearance_features_crop = self.appearance_features[crop_ids]
        else:
            appearance_features_crop = self.appearance_features
            
        colors_crop = self.color_nn(
            appearance_embed.repeat(appearance_features_crop.shape[0], 1),
            appearance_features_crop,
        ).float()
        
        # Update outputs with processed colors
        outputs["rgb"] = colors_crop
        
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        
        # Add alpha loss if enabled
        if self.config.enable_alpha_loss:
            alpha_loss = torch.tensor(0.0).to(self.device)
            background = outputs["background"]
            # Add alpha loss calculation here
            loss_dict["alpha_loss"] = alpha_loss
            
        # Add robust mask if enabled
        if self.step >= self.config.start_robust_mask_at and self.config.enable_robust_mask:
            # Add robust mask calculation here
            pass
            
        return loss_dict
