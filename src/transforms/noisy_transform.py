import albumentations as A
import torch
import torchvision.transforms.v2 as transforms
import torchvision.tv_tensors as tv_tensors

from src.utils.train_utils import create_target


class NoisyTransform:
    """Does random flips, adds random noise and does random erasing."""

    def __init__(self, road_threshold: float = 0.25):
        self.road_threshold = road_threshold

        self.base_transform = transforms.Compose(
            [
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )

        self.image_transform = transforms.Compose(
            [
                transforms.GaussianNoise(sigma=1 / 255),
                transforms.RandomErasing(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
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
