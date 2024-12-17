import os

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.optim import lr_scheduler


class SegLibModel(nn.Module):
    def __init__(self, patch_size, arch, encoder):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder,
            in_channels=3,
            classes=1,
        )

        # self.fc = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(patch_size**2, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, out_dim**2),
        # )

        self.net = nn.Sequential(
            self.model, nn.AvgPool2d(kernel_size=16, stride=16), nn.Flatten()
        )

    def forward(self, img, **batch):
        """
        Model forward method.

        Args:
            img (Tensor): input img.
        Returns:
            output (dict): output dict containing logits.
        """
        out = self.net(img)
        return {"logits": out}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
