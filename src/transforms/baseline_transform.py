import torch
import torchvision.transforms.v2 as transforms
import torchvision.tv_tensors as tv_tensors


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
        self.transform = (transforms.ToDtype(torch.float32, scale=True),)

    def __call__(self, sample: dict):
        # Convert to Image such that the transform is applied to both
        sample["img"] = tv_tensors.Image(sample["img"])
        sample["mask"] = tv_tensors.Image(sample["mask"])

        # Applies the transform to both the image and the mask
        sample = self.transform(sample)

        # Compute the target from the mask
        mask = sample["mask"]
        if mask.mean() > self.road_threshold:
            sample["labels"] = torch.tensor([1.0], device=mask.device)
        else:
            sample["labels"] = torch.tensor([0.0], device=mask.device)

        return sample

    def to(self, device: torch.device):
        self.transform.to(device)
