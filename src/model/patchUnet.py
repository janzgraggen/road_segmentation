import torch
from torch import nn


class PatchUNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=1,
        use_direct_stride=False,  # Boolean parameter to control stride behavior
        nrChannels1=16,  # Number of channels for the first set of layers
        nrChannels2=32,  # Number of channels for the second set of layers
        nrChannels3=64,  # Number of channels for the third set of layers
        nrChannels4=128,  # Number of channels for the fourth set of layers
    ):
        super(PatchUNet, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder: Each stage consists of two convolutional layers followed by pooling
        self.enc1 = self._block(in_channels, nrChannels1)  # 304x304 -> 152x152
        self.enc2 = self._block(nrChannels1, nrChannels2)  # 152x152 -> 76x76
        self.enc3 = self._block(nrChannels2, nrChannels3)  # 76x76 -> 38x38
        self.enc4 = self._block(nrChannels3, nrChannels4)  # 38x38 -> 19x19

        # Bottleneck
        self.bottleneck = self._block(nrChannels4, nrChannels4 * 2)

        # Decoder: Upsampling + Skip Connections
        self.upconv4 = nn.ConvTranspose2d(
            nrChannels4 * 2, nrChannels4, kernel_size=2, stride=2
        )  # 19 -> 38
        self.dec4 = self._block(nrChannels4 + nrChannels4, nrChannels4)

        self.upconv3 = nn.ConvTranspose2d(
            nrChannels4, nrChannels3, kernel_size=2, stride=2
        )  # 38 -> 76
        self.dec3 = self._block(nrChannels3 + nrChannels3, nrChannels3)

        self.upconv2 = nn.ConvTranspose2d(
            nrChannels3, nrChannels2, kernel_size=2, stride=2
        )  # 76 -> 152
        self.dec2 = self._block(nrChannels2 + nrChannels2, nrChannels2)

        self.upconv1 = nn.ConvTranspose2d(
            nrChannels2, nrChannels1, kernel_size=2, stride=2
        )  # 152 -> 304
        self.dec1 = self._block(nrChannels1 + nrChannels1, nrChannels1)

        # Final output layer with patch-level prediction
        if use_direct_stride:
            # Direct stride 16 convolution (304 -> 19)
            self.final_conv = nn.Conv2d(
                nrChannels1, num_classes, kernel_size=3, stride=16, padding=1
            )  # Direct reduction to 19x19
        else:
            # Multiple stride 2 convolutions (304 -> 152 -> 76 -> 38 -> 19)
            self.final_conv1 = nn.Conv2d(
                nrChannels1, nrChannels1, kernel_size=3, stride=2, padding=1
            )  # 304 -> 152

            self.final_conv2 = nn.Conv2d(
                nrChannels1, nrChannels1, kernel_size=3, stride=2, padding=1
            )  # 152 -> 76

            self.final_conv3 = nn.Conv2d(
                nrChannels1, nrChannels1, kernel_size=3, stride=2, padding=1
            )  # 76 -> 38

            self.final_conv4 = nn.Conv2d(
                nrChannels1, num_classes, kernel_size=3, stride=2, padding=1
            )  # 38 -> 19

    def _block(self, in_channels, out_channels):
        """Defines a block with two convolutions and ReLU activations."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder forward pass
        x1 = self.enc1(x)  # 304x304x3 -> 304x304xnrChannels1
        x2 = self.pool(x1)  # 304 -> 152

        x2 = self.enc2(x2)  # 152x152 -> 152x152xnrChannels2
        x3 = self.pool(x2)  # 152 -> 76

        x3 = self.enc3(x3)  # 76x76 -> 76x76xnrChannels3
        x4 = self.pool(x3)  # 76 -> 38

        x4 = self.enc4(x4)  # 38x38 -> 38x38xnrChannels4
        x5 = self.pool(x4)  # 38 -> 19

        # Bottleneck
        x_bottleneck = self.bottleneck(x5)  # 19x19xnrChannels4*2

        # Decoder forward pass with skip connections
        x = self.upconv4(x_bottleneck)  # 19 -> 38
        x = torch.cat((x, x4), dim=1)  # Concatenate skip connection
        x = self.dec4(x)

        x = self.upconv3(x)  # 38 -> 76
        x = torch.cat((x, x3), dim=1)  # Concatenate skip connection
        x = self.dec3(x)

        x = self.upconv2(x)  # 76 -> 152
        x = torch.cat((x, x2), dim=1)  # Concatenate skip connection
        x = self.dec2(x)

        x = self.upconv1(x)  # 152 -> 304
        x = torch.cat((x, x1), dim=1)  # Concatenate skip connection
        x = self.dec1(x)

        # Apply final convolutions based on the parameter
        if hasattr(self, "final_conv"):
            # Direct stride 16 convolution (304 -> 19)
            return self.final_conv(x)
        else:
            # Multiple stride 2 convolutions (304 -> 152 -> 76 -> 38 -> 19)
            x = self.final_conv1(x)  # 304 -> 152
            x = self.final_conv2(x)  # 152 -> 76
            x = self.final_conv3(x)  # 76 -> 38
            return self.final_conv4(x)  # 38 -> 19
