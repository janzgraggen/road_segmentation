import torch.nn as nn
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)


class MobileNetV3(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = deeplabv3_mobilenet_v3_large(
            DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        )
        self.net.classifier[-1] = nn.Conv2d(256, 1, 1)
        self.down_sample = nn.Sequential(
            nn.AvgPool2d(16),
            nn.Flatten(),
        )

        self.pooling = nn.AvgPool2d(16)

    def forward(self, img, **batch):
        """
        Model forward method.

        Args:
            img (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        out = self.net(img)["out"]
        return {"logits": self.down_sample(out)}
