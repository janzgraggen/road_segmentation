import os

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler


class SegCNN(nn.Module):
    def __init__(self, patch_size):  # num_classes=2 for binary segmentation
        super(SegCNN, self).__init__()

        # Encoder: Series of convolutional layers to extract features
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Decoder: Transposed convolutions to upsample
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # Upsample
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # Upsample
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                64, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # Restore to input size
        )

        self.net = nn.Sequential(
            self.encoder,
            self.decoder,
            nn.AvgPool2d(kernel_size=16, stride=16),
            nn.Flatten(),
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
