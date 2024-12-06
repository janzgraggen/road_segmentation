from torch import nn
from torch.nn import Sequential


class CNN2Layer(nn.Module):
    """
    CNN model for full images.
    takes in (3xin_size x in_size) images and
    outputs a flattend (out_size x out_size) = (out^2,) array of logits ().
    """

    def __init__(
        self,
        patch_size,
        out_dim,
        fc_hidden,
        out_channels1,
        kernel1,
        stride1,
        padding1,
        bias1,
        pool1_kernel,
        pool1_stride,
        pool1_padding,
        out_channels2,
        kernel2,
        stride2,
        padding2,
        bias2,
    ):
        """
        Args:
            out_dim (int): output dimension.
            fc_hidden (int): hidden layer dimension.

            out_channels1 (int): number of output channels in the first convolutional layer.
            kernel1 (int): kernel size in the first convolutional layer.
            stride1 (int): stride in the first convolutional layer.
            padding1 (int): padding in the first convolutional layer.
            bias1 (bool): bias in the first convolutional layer.

            pool1_kernel (int): kernel size in the first pooling layer.
            pool1_stride (int): stride in the first pooling layer.
            pool1_padding (int): padding in the first pooling layer.

            out_channels2 (int): number of output channels in the second convolutional layer.
            kernel2 (int): kernel size in the second convolutional layer.
            stride2 (int): stride in the second convolutional layer.
            padding2 (int): padding in the second convolutional layer.
            bias2 (bool): bias in the second convolutional layer.
        """
        # for example:
        # out_dim = 19
        # fc_hidden = 512
        # out_channels1 = 64
        # kernel1 = 10
        # stride1 = 1
        # padding1 = 1
        # bias1 = False

        # pool1_kernel = 2
        # pool1_stride = 2
        # pool1_padding = 0

        # out_channels2 = 128
        # kernel2 = 10
        # stride2 = 1
        # padding2 = 1
        # bias2 = False

        super(CNN2Layer, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        # first layer is a convolutional layer:
        # input is 3x304x304, output is 64x304-10x304-10 = 64x297x297

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=out_channels1,
            kernel_size=kernel1,
            stride=stride1,
            padding=padding1,
            bias=bias1,
        )

        # batch normalization 1
        self.bn1 = nn.BatchNorm2d(out_channels1)

        # input is 64x297x297, output is 64x148x148
        self.pool1 = nn.MaxPool2d(
            kernel_size=pool1_kernel, stride=pool1_stride, padding=pool1_padding
        )

        # input is 64x148x148, output is  128x141x141
        self.conv2 = nn.Conv2d(
            in_channels=out_channels1,
            out_channels=out_channels2,
            kernel_size=kernel2,
            stride=stride2,
            padding=padding2,
            bias=bias2,
        )

        # batch normalization 2
        self.bn2 = nn.BatchNorm2d(out_channels2)

        # input is 128x141x141, output is 128x1x1
        self.pool2 = nn.AdaptiveAvgPool2d(1)

        # fully connected  block
        self.fc = nn.Sequential(
            # input is 128x1x1, output is 128
            nn.Flatten(),
            # input is 128, output is fc_hidden = 512
            nn.Linear(out_channels2, fc_hidden),
            nn.ReLU(),
            # input is fc_hidden = 512, output is 19^2 = 361
            nn.Linear(fc_hidden, out_dim**2),
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
