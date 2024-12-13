import logging
import os
import re

import numpy as np
import torch
import torchvision.io
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ImageRoadDataset(Dataset):
    """
    Dataset for road images. Loads all images from the path into memory.

    Args:
        data_path (str): path to the images.
        target_path (str, optional): path to the masks. Defaults to None.
    """

    def __init__(
        self,
        data_path: str,
        target_path: str | None = None,
    ):
        self.images = self._load_images(data_path)
        self.masks = None
        if target_path is not None:
            self.masks = self._load_images(target_path, grayscale=True)

            assert len(self.images) == len(self.masks)

    def __getitem__(self, index: int) -> dict:
        """
        Get the instance data.

        Args:
            index (int): index of the instance.

        Returns:
            dict: instance data.
        """
        instance_data = {}

        image = self.images[index]
        instance_data["img"] = image

        if self.masks is not None:
            mask = self.masks[index]
            instance_data["mask"] = mask

        return instance_data

    def __len__(self):
        return len(self.images)

    def to(self, device: torch.device):
        """
        Move the dataset to the device.

        Args:
            device (torch.device): device to move the dataset to.
        """
        self.images = self.images.to(device)
        if self.masks is not None:
            self.masks = self.masks.to(device)

    def _load_images(
        self,
        path: str,
        grayscale: bool = False,
    ) -> torch.Tensor:
        """
        Load all images from the path.

        Args:
            path (str): path to the images.
            grayscale (bool, optional): read images in grayscale. Defaults to False.
        """
        if grayscale:
            mode = torchvision.io.ImageReadMode.GRAY
        else:
            mode = torchvision.io.ImageReadMode.UNCHANGED

        files = np.array(os.listdir(path))

        # Extract the ids from the file names
        ids = [int(re.search(r"\d+", file).group()) for file in files]

        # Sort the files by the ids
        order = np.argsort(ids)
        files = files[order]

        images = []

        for file in files:
            image_path = os.path.join(path, file)
            image = torchvision.io.read_image(image_path, mode=mode)
            images.append(image)

        return torch.stack(images)
