import torch
from torch import nn
from torchvision.models.segmentation import deeplabv3_resnet101


class AdaptedDeepLabV3Plus(nn.Module):
    def __init__(
        self,
        num_classes=1,  # Number of output classes
        backbone="resnet101",  # Backbone for DeepLabV3 (e.g., resnet50, resnet101)
        pretrained=True,  # Use pretrained weights
        extra_convs=True,  # Add extra convolutional layers
    ):
        super(AdaptedDeepLabV3Plus, self).__init__()

        # Load DeepLabV3 model with specified backbone
        if backbone == "resnet101":
            self.deeplab = deeplabv3_resnet101(pretrained=pretrained)
        elif backbone == "resnet50":
            from torchvision.models.segmentation import deeplabv3_resnet50

            self.deeplab = deeplabv3_resnet50(pretrained=pretrained)

        elif backbone == "mobilenetv3":
            from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

            self.deeplab = deeplabv3_mobilenet_v3_large(pretrained=pretrained)
        else:
            raise ValueError("Unsupported backbone. Choose 'resnet50' or 'resnet101'.")

        # Modify the classifier to output the desired number of classes
        self.deeplab.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

        self.down_sample = nn.Sequential(
            nn.AvgPool2d(16),
            nn.Flatten(),
        )

        # Add additional convolutional layers to resize the output from 256x256 to 19x19
        self.extra_convs = nn.Sequential(
            nn.Conv2d(
                num_classes, num_classes, kernel_size=3, stride=2, padding=1
            ),  # 256 -> 128
            nn.ReLU(inplace=True),
            nn.Conv2d(
                num_classes, num_classes, kernel_size=3, stride=2, padding=1
            ),  # 128 -> 64
            nn.ReLU(inplace=True),
            nn.Conv2d(
                num_classes, num_classes, kernel_size=3, stride=2, padding=1
            ),  # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(
                num_classes, num_classes, kernel_size=3, stride=2, padding=1
            ),  # 32 -> 16
            nn.Flatten(),
        )

        if extra_convs:
            self.finish = self.extra_convs
        else:
            self.finish = self.down_sample

    def forward(self, img, **batch):
        # Forward pass through DeepLabV3
        x = self.deeplab(img)["out"]  # Extract the output logits

        # Pass the logits through additional convolutional layers
        logits = self.finish(x)

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
