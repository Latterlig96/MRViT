import torch
import torch.nn as nn
from torchvision import models
from mrvit.config import Config

__all__ = ['MRNet',]

class MRNet(nn.Module):

    """
        MRNet

        Implementation of MRNet based on https://github.com/ahmedbesbes/mrnet/blob/master/model.py

        Args:
        :config: Config - config instance with parameters to apply during instance initialization
        
    """

    def __init__(self, config: Config):
        super().__init__()
        self._config = config
        self.pretrained_model = models.alexnet(pretrained=self._config.model.use_pretrained)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Linear(256, self._config.model.output_dim)

    def forward(self, x):
        x = torch.squeeze(x, dim=0) 
        features = self.pretrained_model.features(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        output = self.classifer(flattened_features)
        return output
