import json
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from .image_base import ImageBaseDataset
import torchvision.transforms as transforms
from .registry import register_dataset
import os

@register_dataset(name='FFHQ')
class FFHQDataset(ImageBaseDataset):
    def __init__(self,
                 image_root_path: str,
                 data_info_path: str,
                 valid_extensions: list,
                 resolution: int,
                 is_train: bool,
                 start_idx,
                 end_idx,
                 device):
        super().__init__()

        '''Initialize variables'''
        self.valid_extensions = valid_extensions
        self.resolution = resolution
        self.device = device
        self.image_root_path = image_root_path
        self.data_info_path = data_info_path
        self.is_train = is_train

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

        if self.data_info_path and  os.path.exists(self.data_info_path):
            '''Open file info path to filter the files'''
            with open(self.data_info_path, 'r') as fp:
                file_info_dict = json.load(fp)

            check_string = "training" if self.is_train else "validation"
            tqdm_bar = tqdm(raw_fpaths)

            filtered_fpaths = []
            processed_num = 0
            accepted_num = 0
            for path in tqdm_bar:
                file_id = path.split("/")[-1].split(".")[0]
                if file_info_dict[str(int(file_id))]['category'] == check_string:
                    filtered_fpaths.append(path)
                    accepted_num += 1
                processed_num += 1
                tqdm_bar.set_description(
                    f"Processed {processed_num} files and selected {accepted_num} files")

            assert len(
                filtered_fpaths) > 0, "File list is empty. Check the root."

            return filtered_fpaths
        else:
            return raw_fpaths

    def __len__(self):
        return len(self.fpaths)

    def get_shape(self):
        return (3, self.resolution, self.resolution)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = (self.transform(Image.open(fpath))*2-1)

        return img
