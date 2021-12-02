import torch
import torch.nn as nn
from torch.functional import Tensor

"""This script defines the network.
"""


class DualPathNetwork(nn.Module):
    """
    Creates a Dual Path Network - a combination of ResNet and DenseNet.
    The output from a block is split up with one split output used like ResNet and the other split used like DenseNet.
    """

    def __init__(self, configs):
        super(DualPathNetwork, self).__init__()
        self.configs = configs

        dense_depths = [16, 32, 24, 128]
        block_size = [3, 4, 10, 3]
        in_channel = [64, 128, 256, 512]
        out_channel = [128, 256, 512, 1024]

        # conversion of features from 3 to 64 --> start layer
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU()

        # define the stack layers
        self.prev_channel = 64
        self.stack1 = self._layer(in_channel[0], out_channel[0], dense_depths[0], 1, block_size[0])
        self.stack2 = self._layer(in_channel[1], out_channel[1], dense_depths[1], 2, block_size[1])
        self.stack3 = self._layer(in_channel[2], out_channel[2], dense_depths[2], 2, block_size[2])
        self.stack4 = self._layer(in_channel[3], out_channel[3], dense_depths[3], 2, block_size[3])

        # output layer
        self.avg_pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(in_features=out_channel[3] + (block_size[3]+1)*dense_depths[3], out_features=10, bias=True)

    def _layer(self, in_channel, out_channel, dense_depth, stride, block_size):
        """
        Creates stack layers of Dual Path Network
        """
        stack_layers = []
        stride_list = [1] * block_size
        # list of strides: all strides for blocks in a stack layer is 1 except the first block stride which is equal to
        # 2 for layers other than layer 1
        stride_list[0] = stride
        first_layer = True
        for i in range(block_size):
            if i != 0:
                first_layer = False
            stack_layers.append(Bottleneck(self.prev_channel, in_channel, out_channel, dense_depth, stride_list[i],
                                           first_layer))
            self.prev_channel = out_channel + (i+2)*dense_depth
        stack = nn.Sequential(*stack_layers)
        return stack

    def forward(self, inputs: Tensor):
        """
        Args:
            inputs: A Tensor representing a batch of input images.
        Return:
            The output Tensor of the network.
        """
        out = self.conv(inputs)
        out = self.bn(out)
        out = self.relu(out)

        out = self.stack1(out)
        out = self.stack2(out)
        out = self.stack3(out)
        out = self.stack4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Bottleneck(nn.Module):
    """
    Creates a bottleneck layer to be used in Dual Path Network
    """

    def __init__(self, prev_channel, in_channel, out_channel, dense_depth, stride, first_layer) -> None:
        super(Bottleneck, self).__init__()
        self.out_channel = out_channel
        self.conv1 = nn.Conv2d(in_channels=prev_channel, out_channels=in_channel, stride=1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=in_channel)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=stride, kernel_size=3,
                               bias=False,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=in_channel)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel + dense_depth, stride=1, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channel + dense_depth)
        self.relu3 = nn.ReLU()

        # to be used in first block only in that stack layer. Only in the first block, the dimension of the image
        # decreases as stride is > 1. For subsequent blocks, the dimension of the image remains same as stride is 1.
        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=prev_channel, out_channels=out_channel + dense_depth, stride=stride,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(num_features=out_channel + dense_depth)
            )
        self.relu4 = nn.ReLU()

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs: A Tensor representing a batch of input images.
        Return:
            The output Tensor of the network.
        """
        # first convolution layer
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu1(out)

        # second convolution layer
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        # third convolution layer
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        inputs = self.shortcut(inputs)
        split = self.out_channel
        # the first out channel is added with the input like resnet
        # the subsequent channels from input is added like densenet, it keeps on increasing with each block
        # the addition of rest output channels remains the same
        out = torch.cat([inputs[:, :split, :, :] + out[:, :split, :, :], inputs[:, split:, :, :], out[:, split:, :, :]],
                        1)
        out = self.relu4(out)
        return out
