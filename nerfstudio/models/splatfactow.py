from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
from dataclasses import dataclass, field
import torch
from nerfstudio.fields.splatfactow_field import SplatfactoWField
from nerfstudio.fields.background_field import BGField
from nerfstudio.cameras.cameras import Cameras



@dataclass
class SplatfactoWModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: SplatfactoWModel)
    # Add any new configuration options specific to SplatfactoWModel here
    appearance_embed_dim: int = 48
    appearance_features_dim: int = 72
    enable_bg_model: bool = True
    implementation: Literal["tcnn", "torch"] = "tcnn"

class SplatfactoWModel(SplatfactoModel):
    config: SplatfactoWModelConfig

    def populate_modules(self):
        super().populate_modules()
        
        # Add new modules specific to SplatfactoWModel
        assert self.config.appearance_embed_dim > 0
        self.appearance_embeds = torch.nn.Embedding(self.num_train_data, self.config.appearance_embed_dim)

        if self.config.enable_bg_model:
            self.bg_model = BGField(
                appearance_embedding_dim=self.config.appearance_embed_dim,
                implementation=self.config.implementation,
            )
        else:
            self.bg_model = None

        self.color_nn = SplatfactoWField(
            appearance_embed_dim=self.config.appearance_embed_dim,
            appearance_features_dim=self.config.appearance_features_dim,
            implementation=self.config.implementation,
        )

    # Override or add new methods as needed
    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        # Implement the new get_outputs method
        # This will likely involve calling the parent method and then modifying the results
        outputs = super().get_outputs(camera)
        # Add modifications here
        return outputs

    # Add any other new methods or overrides as needed

# Remove any duplicate methods or properties that are already in SplatfactoModel