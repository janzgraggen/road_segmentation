import albumentations as A
import torch
import torchvision.transforms.v2 as transforms
import torchvision.tv_tensors as tv_tensors
from albumentations.pytorch import ToTensorV2

from src.utils.train_utils import create_target


class AlbumentationsTransform:
    def __init__(self, patch_size: int, road_threshold: float):
        self.road_threshold = road_threshold

        self.base_transform = transforms.Compose(
            [
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomCrop(height=patch_size, width=patch_size),
                ToTensorV2(),
            ]
        )

        self.image_transform = transforms.Compose(
            [
                A.Normalize(),  # This uses the ImageNet mean and std by default
                A.RandomBrightnessContrast(p=0.5),
                A.GaussianBlur(p=0.5),
            ]
        )

    def __call__(self, sample: dict):
        # Convert to Image such that the transform is applied to both
        sample["img"] = tv_tensors.Image(sample["img"])
        sample["mask"] = tv_tensors.Image(sample["mask"])

        # Applies the base transform to both the image and the mask
        sample = self.base_transform(sample)

        # Additional transforms for the image
        sample["img"] = self.image_transform(sample["img"])

        # Compute the target from the mask
        mask = sample["mask"]
        target = create_target(mask, 16, self.road_threshold)
        sample["labels"] = target

        return sample

    def to(self, device: torch.device):
        self.base_transform.to(device)
        self.image_transform.to(device)
