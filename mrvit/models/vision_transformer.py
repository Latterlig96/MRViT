import timm
import torch
import torch.nn as nn
from mrvit.config import Config

__all__ = ['VisionTransformer',]

class VisionTransformer(nn.Module):

    """
        VisionTransformer

        Implementation (using `timm` library) of VisionTransformerTiny

        Args:
        :config: Config - config instance with parameters to apply during instance initialization
        
    """

    def __init__(self, config: Config):
        super().__init__()
        self._config = config
        self._backbone = timm.create_model(
            'vit_tiny_r_s16_p8_224',
            pretrained=self._config.model.use_pretrained, 
            num_classes=self._config.model.num_classes,
            in_chans=self._config.model.input_channels
        )
        num_features = self._backbone.num_features
        self.fc = nn.Linear(num_features, self._config.model.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.squeeze(x, dim=0) 
        x = self._backbone(x)
        flattened_features = torch.max(x, 0, keepdim=True)[0]
        out = self.fc(flattened_features)
        return out 
