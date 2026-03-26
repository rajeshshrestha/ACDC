from .registry import get_dataset
from .FFHQ import FFHQDataset
from .imagenet import ImageNetDataset

__all__ = [FFHQDataset,ImageNetDataset, get_dataset]
