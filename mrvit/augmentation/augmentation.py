from typing import Union
import torch
import torchio as tio
from mrvit.config import Config

__all__ = ['Augmentation',]

class TrainAugmentation:

    """
       TrainAugmentation:

       Object whos instance will be applied during training phase

       Args:
       :config: Config - config instance with parameters to apply during instance initialization

    """

    def __init__(self,
                 config: Config):
        self.augmentation = tio.Compose([
            tio.RandomFlip(),
            tio.RandomMotion(),
            tio.RandomBlur(),
            tio.CropOrPad(target_shape=(config.augmentation.image_size[0],
                                        config.augmentation.image_size[1], 
                                        config.augmentation.image_size[2])),
            tio.ZNormalization()
        ])
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        image = self.augmentation(x)
        return image

class ValAugmentation:

    """
       ValAugmentation:

       Object whos instance will be applied during validation phase

       Args:
       :config: Config - config instance with parameters to apply during instance initialization

    """
    
    def __init__(self,
                 config: Config):
        self.augmentation = tio.Compose([
            tio.CropOrPad(target_shape=(config.augmentation.image_size[0], 
                                        config.augmentation.image_size[1], 
                                        config.augmentation.image_size[2])),
            tio.ZNormalization()
        ])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        image = self.augmentation(x)
        return image

class Augmentation:

    """
        Augmentation:

        Simple manager to operate on TrainAugmentation and ValAugmentation classes and pick
        one based on user need

        Args:
        :config: Config - config instance with parameters to apply during instance initialization

    """
    
    def __init__(self, config: Config):
        self._config = config
    
    def get_by_mode(self, mode: str) -> Union[TrainAugmentation, ValAugmentation]:
        """
            Get certain augmentation instance based on condition

            Args:
            :mode: str - defines phase (either `train` or `val`) and picks up proper augmentation instance based on that condition

            Returns:
            :augmentation: Union[TrainAugmentation, ValAugmentation] - augmentation class instance

        """
        return TrainAugmentation(self._config) if mode == "train" else ValAugmentation(self._config)
