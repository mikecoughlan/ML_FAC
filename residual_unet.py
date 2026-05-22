"""
Modular Residual U-Net in PyTorch
==================================
Supports:
  - Configurable number of residual blocks per encoder/decoder level
  - Configurable number of conv layers within each residual block
  - Different input and output spatial sizes (via bilinear upsampling)
  - Configurable channel progression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    A residual block made of `num_layers` sequential Conv2d → BN → ReLU stacks,
    with a skip connection from input to output.

    If the number of channels changes between input and output, a 1×1 conv is
    used on the skip path to match dimensions.
    """

    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 2):
        super().__init__()
        assert num_layers >= 1, "A residual block must have at least one layer."

        layers = []
        for i in range(num_layers):
            ch_in  = in_channels  if i == 0 else out_channels
            ch_out = out_channels
            layers += [
                nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            ]
        # Drop the final ReLU so we can apply it *after* the residual addition
        layers = layers[:-1]
        self.block = nn.Sequential(*layers)

        # Skip projection if channel counts differ
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + self.skip(x))


class EncoderBlock(nn.Module):
    """
    One encoder level: one or more residual blocks followed by 2×2 max-pool.
    Returns both the pooled output (to the next level) and the pre-pool
    feature map (for the skip connection).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 num_res_blocks: int = 1, layers_per_block: int = 2):
        super().__init__()
        blocks = [ResidualBlock(in_channels, out_channels, layers_per_block)]
        for _ in range(num_res_blocks - 1):
            blocks.append(ResidualBlock(out_channels, out_channels, layers_per_block))
        self.res_blocks = nn.Sequential(*blocks)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.res_blocks(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    """
    One decoder level: bilinear upsample → concat skip → one or more residual blocks.
    Bilinear upsampling means the decoder handles arbitrary (non-power-of-two) sizes.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 num_res_blocks: int = 1, layers_per_block: int = 2):
        super().__init__()
        # After concatenation the channel count is in_channels + skip_channels
        merged = in_channels + skip_channels
        blocks = [ResidualBlock(merged, out_channels, layers_per_block)]
        for _ in range(num_res_blocks - 1):
            blocks.append(ResidualBlock(out_channels, out_channels, layers_per_block))
        self.res_blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample to the spatial size of the skip tensor
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.res_blocks(x)


# ---------------------------------------------------------------------------
# Full U-Net
# ---------------------------------------------------------------------------

class ResidualUNet(nn.Module):
    """
    Modular Residual U-Net.

    Parameters
    ----------
    in_channels       : Number of input image channels (e.g. 3 for RGB).
    out_channels      : Number of output channels / classes.
    base_channels     : Channel count at the first encoder level. Subsequent
                        levels double the channels (configurable via channel_mult).
    depth             : Number of encoder (and decoder) levels, excluding bottleneck.
    num_res_blocks    : Residual blocks per encoder/decoder level.
    layers_per_block  : Conv layers inside each residual block.
    channel_mult      : Multiplier applied to channels at each deeper level.
    output_size       : Optional (H, W) to resize the final output. If None the
                        output spatial size matches the last decoder feature map.
    """

    def __init__(
        self,
        in_channels:    int        = 1,
        out_channels:   int        = 1,
        base_channels:  int        = 64,
        depth:          int        = 4,
        num_res_blocks: int        = 1,
        layers_per_block: int      = 2,
        channel_mult:   float      = 2.0,
        output_size:    tuple      = None,
    ):
        super().__init__()
        self.output_size = output_size

        # --- Channel progression ----------------------------------------
        enc_channels: List[int] = [
            int(base_channels * (channel_mult ** i)) for i in range(depth)
        ]
        bot_channels: int = int(enc_channels[-1] * channel_mult)

        # --- Stem (maps input channels → base_channels) -----------------
        self.stem = ResidualBlock(in_channels, enc_channels[0], layers_per_block)

        # --- Encoder ----------------------------------------------------
        self.encoders = nn.ModuleList()
        for i in range(depth):
            ch_in  = enc_channels[i]
            ch_out = enc_channels[i]   # stem already set first ch_out
            if i > 0:
                ch_in = enc_channels[i - 1]
                ch_out = enc_channels[i]
                block = EncoderBlock(ch_in, ch_out, num_res_blocks, layers_per_block)
            else:
                # First encoder level: stem already processed, just pool + res
                block = EncoderBlock(enc_channels[0], enc_channels[0], num_res_blocks, layers_per_block)
            self.encoders.append(block)

        # --- Bottleneck -------------------------------------------------
        self.bottleneck = nn.Sequential(
            *[ResidualBlock(
                enc_channels[-1] if j == 0 else bot_channels,
                bot_channels, layers_per_block
              ) for j in range(num_res_blocks)]
        )

        # --- Decoder ----------------------------------------------------
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):
            dec_in   = bot_channels if i == depth - 1 else int(enc_channels[i + 1])
            skip_ch  = enc_channels[i]
            dec_out  = enc_channels[i]
            self.decoders.append(
                DecoderBlock(dec_in, skip_ch, dec_out, num_res_blocks, layers_per_block)
            )

        # --- Output head ------------------------------------------------
        self.head = nn.Conv2d(enc_channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.stem(x)

        # Encoder — collect skip connections
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder — consume skips in reverse
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        # Output projection
        x = self.head(x)

        # Optional resize to target output size
        if self.output_size is not None:
            x = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)

        return x


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = ResidualUNet(
        in_channels=3,
        out_channels=1,
        base_channels=32,
        depth=4,
        num_res_blocks=2,       # 2 residual blocks per encoder/decoder level
        layers_per_block=3,     # 3 conv layers per residual block
        channel_mult=2.0,
        output_size=(256, 256), # output a fixed 256×256 regardless of input size
    )

    dummy = torch.randn(2, 3, 572, 572)   # e.g. original U-Net input size
    out   = model(dummy)

    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Param count  : {sum(p.numel() for p in model.parameters()):,}")
