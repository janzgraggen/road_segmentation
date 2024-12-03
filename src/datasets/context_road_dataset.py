import logging
import os
import random

import torch
import torchvision.io
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ContextRoadDataset(Dataset):
    PATCH_SIZE = 16
    ROAD_THRESHOLD = 0.25

    def __init__(
        self,
        size: int,
        data_path: str,
        target_path: str | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ):
        assert size > 0, "Size should be greater than 0."
        assert (
            size % self.PATCH_SIZE == 0
        ), f"Size should be divisible by {self.PATCH_SIZE}."

        random.seed(seed)

        self.size = size
        self.data = []
        self.target = None

        images = os.listdir(data_path)
        images.sort()

        for image_name in images:
            image_path = os.path.join(data_path, image_name)
            image = torchvision.io.read_image(image_path)

            self.data.append(image.float())

        if target_path is not None:
            self.target = []

            masks = os.listdir(target_path)
            masks.sort()

            for mask_name in masks:
                mask_path = os.path.join(target_path, mask_name)

                # Read the mask as a grayscale image
                mode = torchvision.io.ImageReadMode.GRAY
                mask = torchvision.io.read_image(mask_path, mode)

                self.target.append(mask.float())

            assert len(self.data) == len(
                self.target
            ), "Data and target should have the same length."

        self._index = list(range(len(self.data)))

        if shuffle:
            random.shuffle(self._index)

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
        data_index = self._index[index]
        data = self.data[data_index]

        # Crop the image to the desired size
        data = self._process_image(data)

        instance_data = {"img": data}

        if self.target is not None:
            target = self.target[data_index]
            target = self._process_image(target)
            target = self._create_target(target)

            instance_data["labels"] = target

        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

    def _process_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Process image.

        Args:
            image (Tensor): input image.
        Returns:
            image (Tensor): processed image.
        """
        crop = image[:, : self.size, : self.size]

        return crop

    def _create_target(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Create target from mask.

        Args:
            mask (Tensor): input mask.
        Returns:
            target (Tensor): target.
        """

        # Create a patch size kernel of ones to convolve with the mask
        # The kernel has one batch and one channel dimension
        kernel = torch.ones(1, 1, self.PATCH_SIZE, self.PATCH_SIZE)

        # Add batch dimension to the mask
        mask = mask.unsqueeze(0)

        # Convolve the mask with the kernel to get the number of roads in the patch
        with torch.no_grad():
            roads = torch.nn.functional.conv2d(mask, kernel, stride=self.PATCH_SIZE)

        # Remove batch and channel dimension
        roads = roads.squeeze()

        # Threshold the number of roads to get the target
        threshold = self.ROAD_THRESHOLD * self.PATCH_SIZE**2
        target = roads > threshold

        # Flatten and convert to float
        return target.flatten().float()
