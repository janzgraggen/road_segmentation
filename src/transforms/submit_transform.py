import torch
import torchvision.transforms.v2 as transforms
import torchvision.tv_tensors as tv_tensors


class SubmitTransform:
    def __init__(self, road_threshold: float):
        self.road_threshold = road_threshold

        self.base_transform = transforms.Compose(
            [
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def __call__(self, sample: dict):
        # Convert to Image such that the transform is applied to both
        sample["img"] = tv_tensors.Image(sample["img"])
        sample = self.base_transform(sample)

        return sample

    def to(self, device: torch.device):
        self.base_transform.to(device)
        # self.image_transform.to(device)
