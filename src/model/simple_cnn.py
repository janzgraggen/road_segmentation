import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, patch_size):
        super(SimpleCNN, self).__init__()

        assert patch_size % 16 == 0, "Patch size must be a multiple of 16"
        out_dim = patch_size // 16

        # First convolutional block
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.dropout = nn.Dropout(0.5)

        self.conv_skip = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0
        )

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, out_dim**2)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, img, **batch):
        """
        Model forward method.

        Args:
            img (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        x1 = F.relu(self.conv1(img))
        x = self.pool(F.relu(self.conv2(x1)))

        x_skip = self.pool(F.relu(self.conv_skip(x1)))
        x = x + x_skip
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return {"logits": x}

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
