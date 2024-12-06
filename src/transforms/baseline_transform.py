import torch
import torchvision.transforms.v2 as transforms
import torchvision.tv_tensors as tv_tensors

from src.utils.train_utils import create_target


class BaselineTransform:
    """
    Baseline transform for the image road dataset.

    Assumes images and masks are patches. Scales the images to [0, 1] and
    computes the target from the mask. The target is 1 if the mean of the mask
    is greater than the road threshold, 0 otherwise.

    Args:
        road_threshold (float): threshold for the road mask.
    """

    def __init__(self, road_threshold: float):
        self.road_threshold = road_threshold
        self.transform = transforms.ToDtype(torch.float32, scale=True)

    def __call__(self, sample: dict):
        # Convert to Image such that the transform is applied to both
        sample["img"] = tv_tensors.Image(sample["img"])

        if "mask" in sample:
            sample["mask"] = tv_tensors.Image(sample["mask"])

        # Applies the transform to both the image and the mask
        sample = self.transform(sample)

        if "mask" in sample:
            # Compute the target from the mask
            mask = sample["mask"]
            target = create_target(mask, 16, self.road_threshold)
            sample["labels"] = target

        return sample

    def to(self, device: torch.device):
        self.transform.to(device)
