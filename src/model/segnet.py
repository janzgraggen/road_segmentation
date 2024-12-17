import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to handle different input/output dimensions
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)  # Apply dropout after the activation

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)  # Add skip connection
        out = self.relu(out)

        return out


class SegNet(nn.Module):
    def __init__(self, patch_size, dropout_rate=0.1):
        in_channels = 3
        super(SegNet, self).__init__()

        self.encoder = nn.Sequential(
            ResidualBlock(in_channels, 64, stride=1, dropout_rate=dropout_rate),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(64, 128, stride=1, dropout_rate=dropout_rate),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(128, 256, stride=1, dropout_rate=dropout_rate),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(256, 512, stride=1, dropout_rate=dropout_rate),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            ResidualBlock(256, 256, stride=1, dropout_rate=dropout_rate),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            ResidualBlock(128, 128, stride=1, dropout_rate=dropout_rate),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            ResidualBlock(64, 64, stride=1, dropout_rate=dropout_rate),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
        )

        out_dim = patch_size // 16

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(patch_size**2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim**2),
        )

        self.net = nn.Sequential(self.encoder, self.decoder, self.fc)

    def forward(self, img, **batch):
        """
        Model forward method.

        Args:
            img (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        return {"logits": self.net(img)}

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
