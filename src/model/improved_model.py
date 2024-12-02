import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    A Residual Block with convolutional layers, batch normalization, and ReLU activation.
    """

    def __init__(self, in_channels, out_channels, stride=1):
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
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += shortcut
        x = self.relu(x)
        return x


class PixelClassifier(nn.Module):
    """
    Pixel-wise classification network for determining if a pixel belongs to a 'row' or not.
    """

    def __init__(self):
        super(PixelClassifier, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        self.resblock1 = ResidualBlock(64, 128, stride=2)
        self.resblock2 = ResidualBlock(128, 256, stride=2)
        self.resblock3 = ResidualBlock(256, 512, stride=2)

        # Global pooling and fully connected classification head
        self.global_pool = nn.AdaptiveAvgPool2d(
            1
        )  # Pool to [batch_size, channels, 1, 1]
        self.fc = nn.Sequential(
            nn.Flatten(),  # Flatten to [batch_size, channels]
            nn.Linear(512, 128),  # Fully connected layer
            nn.ReLU(inplace=True),  # Non-linearity
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(128, 2),  # Output layer for 2 classes
        )
        self.activation = nn.Softmax(dim=1)

        self.net = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.resblock1,
            self.resblock2,
            self.resblock3,
            self.global_pool,
            self.fc,
            self.activation,
        )

    def forward(self, img, **batch):
        """
        Model forward method.

        Args:
            img (Tensor): input img.
        Returns:
            output (dict): output dict containing logits.
        """
        img = img.permute(0, 3, 1, 2)
        logist = self.net(img)
        return {"logits": logist}

    def __str__(self):
        """
        Custom string representation for the PixelClassifier model.
        """
        num_params = sum(p.numel() for p in self.parameters())
        num_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        details = [
            f"{self.__class__.__name__} Model",
            "-" * 40,
            f"Total Parameters: {num_params:,}",
            f"Trainable Parameters: {num_trainable_params:,}",
            "-" * 40,
            "Architecture:",
            super().__str__(),  # Use PyTorch's built-in string representation for layer structure
        ]
        return "\n".join(details)
