from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from mrvit.dataset.dataset import BaseDataset
from typing import Optional
from mrvit.config import Config
from mrvit.augmentation import Augmentation

__all__ = ['DataModule',]

class DataModule(LightningDataModule):

    """
        DataModule

        Implementation of LightningDataModule to operate on dataset and 
        performing custom operations such as defining train/val datasets and dataloaders

        Args:
        :config: Config - config instance with parameters to apply during instance initialization

    """
    
    def __init__(self,
                 config: Config):
        super().__init__()
        self._config = config
    
    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_data = BaseDataset(train=self._config.train.train,
                                          root_dir=self._config.train.root_dir,
                                          task=self._config.train.task,
                                          plane=self._config.train.plane,
                                          transform=Augmentation(self._config).get_by_mode("train"))
            self.val_data = BaseDataset(train=self._config.valid.train,
                                        root_dir=self._config.valid.root_dir,
                                        task=self._config.valid.task,
                                        plane=self._config.valid.plane,
                                        transform=Augmentation(self._config).get_by_mode("validation"))
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self._config.train.batch_size, shuffle=True, num_workers=self._config.train.num_workers)
    
    def val_dataloader(self) -> DataLoader: 
        return DataLoader(self.val_data, batch_size=self._config.valid.batch_size, num_workers=self._config.valid.num_workers)
