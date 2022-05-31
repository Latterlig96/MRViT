import timm
import torch
import torch.nn as nn
from mrvit.config import Config

__all__ = ['SwinTransformer',]

class SwinTransformer(nn.Module):

    """
        SwinTransformer

        Implementation (using `timm` library) of SwinTransformerTiny

        Args:
        :config: Config - config instance with parameters to apply during instance initialization
        
    """

    def __init__(self, config: Config):
        super().__init__()
        self._config = config
        self._backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=self._config.model.use_pretrained,
            in_chans=self._config.model.input_channels
        )
        self._head = nn.Linear(768, self._config.model.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.squeeze(x, dim=0) 
        x = self._backbone.forward_features(x)
        x = self._head(x)
        x = torch.max(x, 0, keepdim=True)[0]
        return x 
