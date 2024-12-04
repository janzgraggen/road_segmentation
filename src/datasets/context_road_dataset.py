import logging
import os

import torch
import torchvision.io
import torchvision.transforms.v2 as transforms
import torchvision.tv_tensors
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ContextRoadDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        target_path: str | None = None,
    ):
        self.images = self._load_images(data_path)
        self.mask = None
        if target_path is not None:
            mode = torchvision.io.ImageReadMode.GRAY
            self.masks = self._load_images(target_path, mode)

            assert len(self.images) == len(self.masks)

    def to(self, device: torch.device):
        """
        Move the dataset to the device.

        Args:
            device (torch.device): device to move the dataset to.
        """
        self.images = self.images.to(device)
        if self.masks is not None:
            self.masks = self.masks.to(device)

    def __getitem__(self, index):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            index (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
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

    def _load_images(
        self,
        path: str,
        mode: torchvision.io.ImageReadMode = torchvision.io.ImageReadMode.UNCHANGED,
    ) -> torch.Tensor:
        """Load all images from the path."""

        files = os.listdir(path)
        files.sort()

        images = []

        for i, file in enumerate(files):
            image_path = os.path.join(path, file)
            image = torchvision.io.read_image(image_path, mode=mode)
            images.append(image)

        return torch.stack(images)
