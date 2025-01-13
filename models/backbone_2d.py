"""
BackboneNeck: 2D CNN backbone with 3 downsampling blocks + FPN neck.

Processes the BEV pseudo-image from PillarFeatureNet and returns
multi-scale feature maps that are upsampled and concatenated for detection.

Architecture:
    Block 1: stride-2 conv → 3x CBR blocks   (1/2 resolution)
    Block 2: stride-2 conv → 5x CBR blocks   (1/4 resolution)
    Block 3: stride-2 conv → 5x CBR blocks   (1/8 resolution)

FPN Neck:
    Each block is upsampled back to 1/2 resolution and concatenated.
    Output: (B, C1+C2+C3, H/2, W/2)
"""

import torch
import torch.nn as nn
from typing import List, Tuple


def make_cbr(in_ch: int, out_ch: int, kernel: int, stride: int, padding: int) -> nn.Sequential:
    """Conv-BN-ReLU block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
        nn.ReLU(inplace=True),
    )


class DownsampleBlock(nn.Module):
    """
    Downsampling block: stride-2 conv followed by N CBR layers at same resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
    ):
        super().__init__()
        layers = [make_cbr(in_channels, out_channels, 3, stride=2, padding=1)]
        for _ in range(num_layers - 1):
            layers.append(make_cbr(out_channels, out_channels, 3, stride=1, padding=1))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock(nn.Module):
    """
    FPN upsample block: deconvolution to upsample features to target stride.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
    ):
        super().__init__()
        if stride == 1:
            self.deconv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            )
        else:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels,
                    kernel_size=stride, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


class BackboneNeck(nn.Module):
    """
    2D CNN backbone with FPN neck for PointPillars.

    Input:  (B, C_in, H, W) — BEV pseudo-image
    Output: (B, C_out, H/2, W/2) — concatenated multi-scale features

    Typical config:
        layer_nums = [3, 5, 5]
        layer_strides = [2, 2, 2]  (cumulative: x2, x4, x8)
        out_channels = [64, 128, 256]
        upsample_strides = [1, 2, 4]
        upsample_out_channels = [128, 128, 128]

    Final output channels = sum(upsample_out_channels) = 384
    """

    def __init__(
        self,
        in_channels: int = 64,
        layer_nums: List[int] = None,
        layer_strides: List[int] = None,
        out_channels: List[int] = None,
        upsample_strides: List[int] = None,
        upsample_out_channels: List[int] = None,
    ):
        super().__init__()

        if layer_nums is None:
            layer_nums = [3, 5, 5]
        if layer_strides is None:
            layer_strides = [2, 2, 2]
        if out_channels is None:
            out_channels = [64, 128, 256]
        if upsample_strides is None:
            upsample_strides = [1, 2, 4]
        if upsample_out_channels is None:
            upsample_out_channels = [128, 128, 128]

        assert len(layer_nums) == len(out_channels) == len(upsample_strides)

        # Build downsampling blocks
        down_blocks = []
        c = in_channels
        for n_layers, out_ch in zip(layer_nums, out_channels):
            down_blocks.append(DownsampleBlock(c, out_ch, n_layers))
            c = out_ch
        self.down_blocks = nn.ModuleList(down_blocks)

        # Build FPN upsample blocks
        up_blocks = []
        for in_ch, out_ch, stride in zip(out_channels, upsample_out_channels, upsample_strides):
            up_blocks.append(UpsampleBlock(in_ch, out_ch, stride))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.out_channels = sum(upsample_out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W)

        Returns:
            features: (B, C_out, H/2, W/2)
        """
        multi_scale_features = []
        for down_block in self.down_blocks:
            x = down_block(x)
            multi_scale_features.append(x)

        upsampled = []
        for feat, up_block in zip(multi_scale_features, self.up_blocks):
            upsampled.append(up_block(feat))

        # Concatenate along channel dimension
        out = torch.cat(upsampled, dim=1)  # (B, C_out, H', W')
        return out
