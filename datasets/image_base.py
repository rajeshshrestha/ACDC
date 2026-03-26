from abc import ABC, abstractmethod
import numpy as np
from torchvision.datasets import VisionDataset

class ImageBaseDataset(ABC, VisionDataset):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def get_shape(self):
        pass

    def get_random_sample(self):
        idx = np.random.randint(len(self))
        return self[idx]