import timm
import torch
import torch.nn as nn
from mrvit.config import Config

__all__ = ['EfficientNet',]

class EfficientNet(nn.Module):

    """
        EfficientNet

        Implementation (using `timm` library) of EfficientNetB2

        Args:
        :config: Config - config instance with parameters to apply during instance initialization

    """
    
    def __init__(self, config: Config):
        super().__init__()
        self._config = config
        self._backbone = timm.create_model(
            'efficientnet_b2',
            pretrained=self._config.model.use_pretrained, 
            num_classes=self._config.model.num_classes,
            in_chans=self._config.model.input_channels
        )
        num_features = self._backbone.num_features
        self.fc = nn.Linear(num_features, self._config.model.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.squeeze(x, dim=0) 
        x = self._backbone(x)
        pooled_features = x.view(x.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        out = self.fc(flattened_features)
        return out 
