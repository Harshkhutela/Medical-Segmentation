"""Reusable building blocks for the U-Net architecture."""

import torch
from torch import nn
from torch.nn import functional as functional


class DoubleConv(nn.Module):
    """Apply two convolution, batch-normalization, and ReLU operations."""

    def __init__(self, in_channels, out_channels):
        """Initialize the two convolutional layers.

        Args:
            in_channels: Number of channels in the input tensor.
            out_channels: Number of channels produced by the block.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        """Extract features while preserving the input height and width."""
        return self.layers(inputs)


class DownBlock(nn.Module):
    """Reduce spatial resolution and then extract richer encoder features."""

    def __init__(self, in_channels, out_channels):
        """Initialize max pooling followed by a DoubleConv block."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, inputs):
        """Halve feature-map resolution and return transformed features."""
        return self.layers(inputs)


class UpBlock(nn.Module):
    """Upsample decoder features, combine encoder features, and refine them."""

    def __init__(self, in_channels, out_channels):
        """Initialize the upsampling layer and convolutional refinement block."""
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, inputs, skip_features):
        """Upsample inputs, concatenate matching encoder features, and refine.

        Padding keeps the block robust when encoder and decoder dimensions differ
        by one pixel due to input dimensions that are not divisible by 16.
        """
        outputs = self.up(inputs)
        height_difference = skip_features.size(2) - outputs.size(2)
        width_difference = skip_features.size(3) - outputs.size(3)
        outputs = functional.pad(
            outputs,
            [
                width_difference // 2,
                width_difference - (width_difference // 2),
                height_difference // 2,
                height_difference - (height_difference // 2),
            ],
        )
        outputs = torch.cat((skip_features, outputs), dim=1)
        return self.conv(outputs)
