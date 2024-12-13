import torch.nn as nn


# Define Residual Block with Dropout
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


# Define ResNet18 Model with Dropout
class ResNet(nn.Module):
    def __init__(self, patch_size, dropout_rate=0.3):
        super(ResNet, self).__init__()

        assert patch_size % 16 == 0, "Patch size must be a multiple of 16"
        out_dim = patch_size // 16

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-18 layer configurations: [2, 2, 2, 2]
        self.layer1 = self._make_layer(64, 64, 2, 1, dropout_rate)
        self.layer2 = self._make_layer(64, 128, 2, 2, dropout_rate)
        self.layer3 = self._make_layer(128, 256, 2, 2, dropout_rate)
        self.layer4 = self._make_layer(256, 512, 2, 2, dropout_rate)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, out_dim**2)
        self.dropout = nn.Dropout(
            p=dropout_rate
        )  # Apply dropout before the final FC layer

        self.net = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool,
            nn.Flatten(),
            self.dropout,  # Dropout before FC
            self.fc,
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride, dropout_rate):
        layers = [ResidualBlock(in_channels, out_channels, stride, dropout_rate)]
        for _ in range(1, blocks):
            layers.append(
                ResidualBlock(
                    out_channels, out_channels, stride=1, dropout_rate=dropout_rate
                )
            )
        return nn.Sequential(*layers)

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
