import shutil

import numpy as np
import safetensors
import safetensors.torch
import torchvision
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json
import torch.utils.data as data

import os, glob
import matplotlib.image as mpimg
from PIL import Image

import torch

from utils.image import img_crop

class DataLoaderSegmentation(data.Dataset):
    def __init__(self, folder_path):
        super(DataLoaderSegmentation, self).__init__()

        # Load images and masks
        self.img_files = glob.glob(folder_path + "/images/*")
        self.mask_files = glob.glob(folder_path + "/groundtruth/*")

        # Create patches
        self.img_patches = []
        self.mask_patches = []

        for i in range(len(self.img_files)):
            img = Image.open(self.img_files[i])
            mask = Image.open(self.mask_files[i])

            img_patches = img_crop(img, 16, 16)
            mask_patches = img_crop(mask, 16, 16)

            self.img_patches.extend(img_patches)
            self.mask_patches.extend(mask_patches)

            



    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]

            if os.path.exists(img_path) and os.path.exists(mask_path):
                img = mpimg.imread(img_path)
                mask = mpimg.imread(mask_path)

            


            return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return len(self.img_files)


class RoadDataset(BaseDataset):
    def __init__(self, name="train", *args, **kwargs):
        """
        Args:
            name (str): partition name
        """
        index_path = ROOT_PATH / "data" / "road" / name / "index.json"

        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(name)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, name):
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            name (str): partition name
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """


        
        return index