import logging
import os
import random

import torch
import torchvision.io
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ContextRoadDataset(Dataset):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        road_threshold: float,
        data_path: str,
        target_path: str | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ):
        assert image_size > 0
        assert image_size % patch_size == 0

        random.seed(seed)

        self.size = image_size
        self.patch_size = patch_size
        self.road_threshold = road_threshold

        self.images = []
        self.masks = None

        image_names = os.listdir(data_path)
        image_names.sort()

        for image_name in image_names:
            image_path = os.path.join(data_path, image_name)

            # Read the image as a tensor and scale it
            image = torchvision.io.read_image(image_path)
            image = self._scale(image)

            self.images.append(image)

        if target_path is not None:
            self.masks = []

            masks = os.listdir(target_path)
            masks.sort()

            for mask_name in masks:
                mask_path = os.path.join(target_path, mask_name)

                # Read the mask as a grayscale image and scale it
                mode = torchvision.io.ImageReadMode.GRAY
                mask = torchvision.io.read_image(mask_path, mode)
                mask = self._scale(mask)

                self.masks.append(mask)

            assert len(self.images) == len(self.masks)

        self._index = list(range(len(self.images)))

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
        image = self.images[data_index]

        # Crop the image to the desired size
        image = self._process_image(image)

        instance = {"img": image}

        if self.masks is not None:
            mask = self.masks[data_index]
            mask = self._process_image(mask)
            target = self._create_target(mask)

            instance["mask"] = mask
            instance["labels"] = target

        return instance

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

    def _scale(self, image: torch.Tensor) -> torch.Tensor:
        """
        Scale the image to [0, 1].

        Args:
            image (Tensor): input image.
        Returns:
            image (Tensor): scaled image.
        """
        return image / 255.0

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
        kernel = torch.ones(1, 1, self.patch_size, self.patch_size)

        # Add batch dimension to the mask
        mask = mask.unsqueeze(0)

        # Convolve the mask with the kernel to get the number of roads in the patch
        with torch.no_grad():
            roads = torch.nn.functional.conv2d(mask, kernel, stride=self.patch_size)

        # Remove batch and channel dimension
        roads = roads.squeeze()

        # Threshold the number of roads to get the target
        threshold = self.road_threshold * self.patch_size**2
        target = roads > threshold

        # Flatten and convert to float
        return target.flatten().float()
