import torch
import torchvision.transforms.v2 as transforms

from src.transforms.noisy_transform import NoisyTransform


class CropTransform(NoisyTransform):
    """Adds random rotation and crops the image to patch_size x patch_size."""

    def __init__(self, patch_size: int, road_threshold: float = 0.25):
        super().__init__(road_threshold=road_threshold)

        self.base_transform = transforms.Compose(
            [
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(180),
                transforms.RandomResizedCrop(
                    (patch_size, patch_size), scale=(0.5, 1.0)
                ),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )

        self.image_transform = transforms.Compose(
            [
                transforms.GaussianNoise(),
                transforms.RandomErasing(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
