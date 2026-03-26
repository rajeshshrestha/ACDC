import json
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from .image_base import ImageBaseDataset
import torchvision.transforms as transforms
from .registry import register_dataset

@register_dataset(name='Imagenet')
class ImageNetDataset(ImageBaseDataset):
    def __init__(self,
                 image_root_path: str,
                 valid_extensions: list,
                 resolution: int,
                 start_idx,
                 end_idx,
                 device):
        super().__init__()

        '''Initialize variables'''
        self.valid_extensions = valid_extensions
        self.resolution = resolution
        self.device = device
        self.image_root_path = image_root_path

        '''Get the paths to images'''
        self.fpaths = self.get_relevant_file_paths()

        '''Subselect the data'''
        self.fpaths = self.fpaths[start_idx:end_idx]

        '''Define the transforms'''
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.resolution),
            transforms.CenterCrop(self.resolution)
        ])

    def get_relevant_file_paths(self):
        '''Filter files with valid extensions'''
        extensions = ["*"+ext for ext in self.valid_extensions]
        raw_fpaths = sorted([str(path) for ext in extensions for path in Path(self.image_root_path).rglob(ext)])
        return raw_fpaths

    def __len__(self):
        return len(self.fpaths)

    def get_shape(self):
        return (3, self.resolution, self.resolution)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = (self.transform(Image.open(fpath))*2-1)

        return img
