import torch
from torch import nn
from torchvision.models.segmentation import deeplabv3_resnet101


class AdaptedDeepLabV3Plus(nn.Module):
    def __init__(
        self,
        num_classes=1,  # Number of output classes
        backbone="resnet101",  # Backbone for DeepLabV3 (e.g., resnet50, resnet101)
        pretrained=True,  # Use pretrained weights
        patch_size=256,  # Expected input size
    ):
        super(AdaptedDeepLabV3Plus, self).__init__()

        # Load DeepLabV3 model with specified backbone
        if backbone == "resnet101":
            self.deeplab = deeplabv3_resnet101(pretrained=pretrained)
        elif backbone == "resnet50":
            from torchvision.models.segmentation import deeplabv3_resnet50

            self.deeplab = deeplabv3_resnet50(pretrained=pretrained)
        else:
            raise ValueError("Unsupported backbone. Choose 'resnet50' or 'resnet101'.")

        # # Adjust DeepLabV3 to handle 304x304 input size
        # self.deeplab.backbone.conv1 = nn.Conv2d(
        #     3, 64, kernel_size=7, stride=2, padding=3, bias=False
        # )  # First layer to adapt to 304x304

        # Modify the classifier to output the desired number of classes
        self.deeplab.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

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

        # self.upsample = nn.Upsample(
        #     size=(patch_size/16, patch_size/16), mode="bilinear", align_corners=False
        # )  # Upsample to 19x19

    def forward(self, img, **batch):
        # Forward pass through DeepLabV3
        x = self.deeplab(img)["out"]  # Extract the output logits

        # Pass the logits through additional convolutional layers
        logits = self.extra_convs(x)  
        

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
