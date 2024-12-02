import glob
import os
import shutil

import matplotlib.image as mpimg
import numpy as np
import safetensors
import safetensors.torch
import torch
import torch.utils.data as data
import torchvision
from PIL import Image
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class RoadDataset(BaseDataset):
    def __init__(self, name="train", *args, **kwargs):
        """
        Args:
            name (str): partition name
        """
        index_path = ROOT_PATH / "data" / name / "index.json"

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
        if name == "train":
            print("Loading train...")
            self.mask_dir = ROOT_PATH / "road_data" / "train" / "groundtruth"
            self.image_dir = ROOT_PATH / "road_data" / "train" / "images"

            self.image_names = os.listdir(self.image_dir)

            images = []
            masks = []

            for i in range(len(self.image_names)):
                image = mpimg.imread(self.image_dir / self.image_names[i])
                mask = mpimg.imread(self.mask_dir / self.image_names[i])
                images.append(image)
                masks.append(mask)

            patch_size = 16

            img_patches = [
                self.img_crop(images[i], patch_size, patch_size)
                for i in range(len(images))
            ]
            mask_patches = [
                self.img_crop(masks[i], patch_size, patch_size)
                for i in range(len(masks))
            ]

            img_patches = np.asarray(
                [
                    img_patches[i][j]
                    for i in range(len(img_patches))
                    for j in range(len(img_patches[i]))
                ]
            )
            mask_patches = np.asarray(
                [
                    mask_patches[i][j]
                    for i in range(len(mask_patches))
                    for j in range(len(mask_patches[i]))
                ]
            )

            y = np.asarray(
                [
                    self.value_to_class(np.mean(mask_patches[i]))
                    for i in range(len(mask_patches))
                ]
            )

            self.data = torch.Tensor(img_patches)
            self.targets = torch.Tensor(y)

            index = []
            data_path = ROOT_PATH / "data" / name
            data_path.mkdir(exist_ok=True, parents=True)

            for i in tqdm(range(len(self.data))):
                img = self.data[i]
                label = self.targets[i]

                save_dict = {"tensor": img}
                save_path = data_path / f"{i:06}.safetensors"
                safetensors.torch.save_file(save_dict, save_path)

                index.append(
                    {"path": str(save_path), "label": torch.Tensor.tolist(label)}
                )

            write_json(index, str(data_path / "index.json"))

            return index

        elif name == "aicrowd":
            print("Loading test...")
            self.image_dir = ROOT_PATH / "road_data" / "aicrowd"
            images = []

            for i in range(1, 51):
                image = mpimg.imread(self.image_dir / f"test_{i}" / f"test_{i}.png")
                images.append(image)

            patch_size = 16

            img_patches = [
                self.img_crop(images[i], patch_size, patch_size)
                for i in range(len(images))
            ]

            img_patches = np.asarray(
                [
                    img_patches[i][j]
                    for i in range(len(img_patches))
                    for j in range(len(img_patches[i]))
                ]
            )

            self.data = torch.Tensor(img_patches)

            index = []
            data_path = ROOT_PATH / "data" / name
            data_path.mkdir(exist_ok=True, parents=True)

            for i in tqdm(range(len(self.data))):
                img = self.data[i]

                save_dict = {"tensor": img}
                save_path = data_path / f"{i:06}.safetensors"
                safetensors.torch.save_file(save_dict, save_path)

                index.append({"path": str(save_path), "label": [1, 0]})

            write_json(index, str(data_path / "index.json"))

            return index

    def extract_features_2d(self, img):
        feat_m = np.mean(img)
        feat_v = np.var(img)
        feat = np.append(feat_m, feat_v)
        return feat

    def value_to_class(self, v):
        foreground_threshold = 0.25
        df = np.sum(v)
        if df > foreground_threshold:
            return np.asarray([0.0, 1.0])
        else:
            return np.asarray([1.0, 0.0])

    def img_crop(self, im, w, h):
        list_patches = []
        imgwidth = im.shape[0]
        imgheight = im.shape[1]
        is_2d = len(im.shape) < 3
        for i in range(0, imgheight, h):
            for j in range(0, imgwidth, w):
                if is_2d:
                    im_patch = im[j : j + w, i : i + h]
                else:
                    im_patch = im[j : j + w, i : i + h, :]
                list_patches.append(im_patch)
        return list_patches
