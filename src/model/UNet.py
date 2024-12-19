import torch
from torch import nn


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=1,
        use_direct_stride=False,  # Boolean parameter to control stride behavior
        nrChannels1=16,  # Number of channels for the first set of layers
        nrChannels2=32,  # Number of channels for the second set of layers
        nrChannels3=64,  # Number of channels for the third set of layers
        nrChannels4=128,  # Number of channels for the fourth set of layers
        drop_prob=0.5,  # Dropout probability
    ):
        super(UNet, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder using nn.Sequential for each block
        self.enc1 = self._block(
            in_channels, nrChannels1, drop_prob
        )  # 304x304 -> 152x152
        self.enc2 = self._block(nrChannels1, nrChannels2, drop_prob)  # 152x152 -> 76x76
        self.enc3 = self._block(nrChannels2, nrChannels3, drop_prob)  # 76x76 -> 38x38
        self.enc4 = self._block(nrChannels3, nrChannels4, drop_prob)  # 38x38 -> 19x19

        # Bottleneck
        self.bottleneck = self._block(nrChannels4, nrChannels4 * 2, drop_prob)

        # Decoder: Upsampling + Skip Connections
        self.upconv4 = nn.ConvTranspose2d(
            nrChannels4 * 2, nrChannels4, kernel_size=2, stride=2
        )  # 19 -> 38
        self.dec4 = self._block(nrChannels4 + nrChannels4, nrChannels4, drop_prob)

        self.upconv3 = nn.ConvTranspose2d(
            nrChannels4, nrChannels3, kernel_size=2, stride=2
        )  # 38 -> 76
        self.dec3 = self._block(nrChannels3 + nrChannels3, nrChannels3, drop_prob)

        self.upconv2 = nn.ConvTranspose2d(
            nrChannels3, nrChannels2, kernel_size=2, stride=2
        )  # 76 -> 152
        self.dec2 = self._block(nrChannels2 + nrChannels2, nrChannels2, drop_prob)

        self.upconv1 = nn.ConvTranspose2d(
            nrChannels2, nrChannels1, kernel_size=2, stride=2
        )  # 152 -> 304
        self.dec1 = self._block(nrChannels1 + nrChannels1, nrChannels1, drop_prob)

        # Final output layer with patch-level prediction

        if use_direct_stride:
            # Direct stride 16 convolution (304 -> 19)
            self.final_conv = nn.AvgPool2d(16)
            # Direct reduction to 19x19
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

    def _block(self, in_channels, out_channels, dropout_prob):
        """Defines a block with two convolutions and ReLU activations."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),  # Dropout added
        )

    def forward(self, img, **batch):
        # Encoder forward pass using sequential layers
        x1 = self.enc1(img)  # 304x304x3 -> 304x304xnrChannels1
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
        img = self.upconv4(x_bottleneck)  # 19 -> 38
        img = torch.cat((img, x4), dim=1)  # Concatenate skip connection
        img = self.dec4(img)

        img = self.upconv3(img)  # 38 -> 76
        img = torch.cat((img, x3), dim=1)  # Concatenate skip connection
        img = self.dec3(img)

        img = self.upconv2(img)  # 76 -> 152
        img = torch.cat((img, x2), dim=1)  # Concatenate skip connection
        img = self.dec2(img)

        img = self.upconv1(img)  # 152 -> 304
        img = torch.cat((img, x1), dim=1)  # Concatenate skip connection
        img = self.dec1(img)

        # Apply final convolutions based on the parameter
        if hasattr(self, "final_conv"):
            # Direct stride 16 convolution (304 -> 19)
            logits = self.final_conv(img)
        else:
            # Multiple stride 2 convolutions (304 -> 152 -> 76 -> 38 -> 19)
            logits = self.final_conv1(img)  # 304 -> 152
            logits = self.final_conv2(logits)  # 152 -> 76
            logits = self.final_conv3(logits)  # 76 -> 38
            logits = self.final_conv4(logits)  # 38 -> 19

        # Flatten logits to match fully connected style output
        logits = logits.view(logits.size(0), -1)  # Flatten to (N, C * H * W)

        # Return output in the same format as ResNet
        return {"logits": logits}

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
