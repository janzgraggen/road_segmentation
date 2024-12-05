import torch
import torchvision.transforms.v2 as transforms
import torchvision.tv_tensors as tv_tensors

from src.utils.train_utils import create_target


class CustomTransform:
    def __init__(self, image_size: int, patch_size: int, road_threshold: float):
        self.patch_size = patch_size
        self.road_threshold = road_threshold

        self.base_transform = transforms.Compose(
            [
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(
                    (image_size, image_size), scale=(0.5, 1.0)
                ),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )

        self.image_transform = transforms.Compose(
            [
                transforms.GaussianNoise(),
                transforms.RandomErasing(),
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
        target = create_target(mask, self.patch_size, self.road_threshold)
        sample["labels"] = target

        return sample

    def to(self, device: torch.device):
        self.base_transform.to(device)
        self.image_transform.to(device)
