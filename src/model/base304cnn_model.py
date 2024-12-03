from torch import nn
from torch.nn import Sequential


class Base304cnn(nn.Module):
    """
        CNN model for 304x304 images.
        takes in 3x304x304 images and outputs a flattend (19x19) = (361,) array of logits ().
    xxx
    """

    def __init__(self, fc_hidden=512):
        """
        Args:
            n_feats (int): number of input features.
            n_class (int): number of classes.
            fc_hidden (int): number of hidden features.
        """
        super(Base304cnn, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        # first layer is a convolutional layer:
        # input is 3x304x304, output is 64x304-10x304-10 = 64x297x297

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=10,
            stride=1,
            padding=1,
            bias=False,
        )

        # batch normalization 1
        self.bn1 = nn.BatchNorm2d(64)

        # input is 64x297x297, output is 64x148x148
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # input is 64x148x148, output is  128x141x141
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=10,
            stride=1,
            padding=1,
            bias=False,
        )

        # batch normalization 2
        self.bn2 = nn.BatchNorm2d(128)

        # input is 128x141x141, output is 128x1x1
        self.pool2 = nn.AdaptiveAvgPool2d(1)

        # fully connected  block
        self.fc = nn.Sequential(
            # input is 128x1x1, output is 128
            nn.Flatten(),
            # input is 128, output is fc_hidden = 512
            nn.Linear(128, fc_hidden),
            nn.ReLU(),
            # input is fc_hidden = 512, output is 19x19 = 361
            nn.Linear(fc_hidden, 19 * 19),
        )

        self.net = Sequential(
            # convolutional layer 1
            self.conv1,
            self.bn1,
            self.relu,
            self.pool1,
            # convolutional layer 2
            self.conv2,
            self.bn2,
            self.relu,
            self.pool2,
            # fully connected layer
            self.fc,
        )

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
