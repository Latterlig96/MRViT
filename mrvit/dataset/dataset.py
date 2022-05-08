from glob import glob
from pathlib import Path
from typing import Callable, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

__all__ = ['Dataset',]

class BaseDataset(Dataset):

    """
        BaseDataset

        Base class to operate on loading/iterating on data during training and validation phase

        Args:
        :train: bool - whether to use class instance as training or validation dataloader
        :root_dir: Path - path to data root directory
        :task: str - task (either ['abnormal', 'acl', 'meniscus'])
        :plane: str -  plane (either ['axial', 'coronal', 'sagital'])
        :transform: Callable - transform class instance (either ['TrainAugmentation', 'ValAugmentation'])

    """

    def __init__(self,
                 train: bool,
                 root_dir: Path,
                 task: str,
                 plane: str,
                 transform: Callable):
        super().__init__()        
        if train:
            self._df = pd.read_csv(root_dir/f'train-{task}.csv', header=None, names=['id', 'label'])
        else:
            self._df = pd.read_csv(root_dir/f'valid-{task}.csv', header=None, names=['id', 'label'])
        self._transform = transform
        self._images = glob(str(root_dir/'train'/plane/"*.npy")) if train else glob(str(root_dir/'valid'/plane/"*.npy"))
        self._labels = self._df['label'].values
        self._weight = self._df['label'].sum()

    def get_data_weight(self) -> float:
        """
            Getter for labels weight to use

            Returns:
            :weight: float - labels weight

        """
        return self._weight

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        image: np.ndarray = np.load(self._images[idx])
        image: torch.Tensor = torch.from_numpy(image).repeat(3, 1, 1, 1).permute(0, 2, 3, 1)
        label: int = self._labels[idx]
        image = self._transform(image).permute(3, 0, 1, 2)
        return image, label.astype('float')
