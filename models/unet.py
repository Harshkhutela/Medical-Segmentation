"""U-Net model for binary medical image segmentation."""

from torch import nn

from .blocks import DoubleConv, DownBlock, UpBlock


class UNet(nn.Module):
    """Encode an RGB image and decode it into one binary-mask logit map."""

    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        """Initialize the encoder, bottleneck, decoder, and output layer.

        Args:
            in_channels: Number of image channels. Dataset images use RGB (3).
            out_channels: Number of predicted mask channels (1 for binary masks).
            base_channels: Number of filters in the first encoder stage.
        """
        super().__init__()
        self.initial = DoubleConv(in_channels, base_channels)
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)
        self.down4 = DownBlock(base_channels * 8, base_channels * 16)

        self.up1 = UpBlock(base_channels * 16, base_channels * 8)
        self.up2 = UpBlock(base_channels * 8, base_channels * 4)
        self.up3 = UpBlock(base_channels * 4, base_channels * 2)
        self.up4 = UpBlock(base_channels * 2, base_channels)
        self.output_layer = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, inputs):
        """Return raw per-pixel logits with the input spatial dimensions.

        Encoder features are saved and passed to the corresponding decoder stage
        so the decoder can recover fine spatial detail through skip connections.
        """
        encoder1 = self.initial(inputs)
        encoder2 = self.down1(encoder1)
        encoder3 = self.down2(encoder2)
        encoder4 = self.down3(encoder3)
        bottleneck = self.down4(encoder4)

        decoder1 = self.up1(bottleneck, encoder4)
        decoder2 = self.up2(decoder1, encoder3)
        decoder3 = self.up3(decoder2, encoder2)
        decoder4 = self.up4(decoder3, encoder1)
        return self.output_layer(decoder4)
