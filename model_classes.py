import argparse
# Importing the libraries
import datetime
import gc
import glob
import json
import math
import os
import pickle
import subprocess
import time
from typing import List, Optional, Tuple, Union

import matplotlib
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torchvision.transforms as transforms
# import torchvision
# import torchvision.transforms as transforms
import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

# from torchsummary import summary
# from torchvision.models.feature_extraction import (create_feature_extractor,
#                                                    get_graph_node_names)
import utils


class BK_model(nn.Module):
	def __init__(self, input_size, output_size, num_channels, num_residual_blocks, crps=False):
		super(BK_model, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.num_channels = num_channels
		self.num_residual_blocks = num_residual_blocks
		self.num_nodes = (128)*32
		self.dropout = 0.2
		self.crps = crps

		# Initial convolution layer
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_channels, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(num_channels)

		# Residual blocks
		self.residual_block_1 = nn.Sequential(
				nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
				nn.BatchNorm2d(num_channels),
				nn.ReLU(),
				nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
				nn.BatchNorm2d(num_channels),
				nn.ReLU(),
				nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
				nn.BatchNorm2d(num_channels)
			)
		self.residual_block_2 = nn.Sequential(
				nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
				nn.BatchNorm2d(num_channels),
				nn.ReLU(),
				nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
				nn.BatchNorm2d(num_channels),
				nn.ReLU(),
				nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
				nn.BatchNorm2d(num_channels)
			)
		self.residual_block_3 = nn.Sequential(
				nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
				nn.BatchNorm2d(num_channels),
				nn.ReLU(),
				nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
				nn.BatchNorm2d(num_channels),
				nn.ReLU(),
				nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
				nn.BatchNorm2d(num_channels)
			)

		# Calculate the size after the convolutional layers to define the first fully connected layer
		residual_output_size = (num_channels, input_size[0], input_size[1])  # Assuming padding keeps the spatial dimensions the same

		# Final layers
		self.linear_block = nn.Sequential(
				nn.Linear(residual_output_size[0]*residual_output_size[1]*residual_output_size[2], output_size[0]),
				nn.ReLU(),
				nn.Dropout(self.dropout),
				nn.Linear(output_size[0], output_size[0])
			)
	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))

		# for ___ in range(self.num_residual_blocks):
		# 	residual = x
		# 	x = self.residual_block(x)
		# 	x += residual
		# 	x = F.relu(x)
		residual = x
		x = self.residual_block_1(x)
		x = x + residual
		x = F.relu(x)

		residual = x
		x = self.residual_block_2(x)
		x = x + residual
		x = F.relu(x)

		residual = x
		x = self.residual_block_2(x)
		x = x + residual
		x = F.relu(x)

		x = x.view(x.size(0), -1)
		x = self.linear_block(x)

		if self.crps:
			x = x.view(x.shape[0], int(x.shape[1]/2), 2)
		# x = x.view(x.size(0), self.output_size[0], self.output_size[1], self.output_size[2])
		return x


"""
ACORN-Net  —  Attention Convolutional Residual Network
=======================================================
A Residual U-Net with two attention mechanisms:

Architecture
------------
  Stem → [Encoder × depth] → [Bottleneck blocks] → [Decoder × depth] → Head

  Each encoder level  : num_res_blocks ResidualBlock(+CBAM) → MaxPool2d(2)
  Bottleneck          : num_res_blocks ResidualBlock(+CBAM) + Dropout2d
  Each decoder level  : AttentionGate → bilinear upsample → concat → num_res_blocks ResidualBlock(+CBAM)
"""

# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention (half of CBAM)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(self.avg_pool(x))
        mx  = self.mlp(self.max_pool(x))
        scale = self.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class SpatialAttention(nn.Module):
    """Spatial attention map (second half of CBAM)."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * scale


class CBAM(nn.Module):
    """Full Convolutional Block Attention Module (channel → spatial)."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class AttentionGate(nn.Module):
    """
    Additive attention gate for skip connections (Oktay et al., 2018).

    g  : gating signal from the decoder path (coarser resolution).
    x  : skip feature map from the encoder path.

    Returns x re-weighted by a spatial attention coefficient.
    """

    def __init__(self, skip_channels: int, gate_channels: int):
        super().__init__()
        inter = max(1, skip_channels // 2)
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter),
        )
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # Upsample g to match x's spatial size if needed
        g_up = F.interpolate(g, size=x.shape[2:],
                             mode="bilinear", align_corners=False)
        alpha = self.psi(self.relu(self.W_x(x) + self.W_g(g_up)))
        return x * alpha


# ---------------------------------------------------------------------------
# Core residual block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    num_layers × (Conv2d → BN → ReLU) with a residual skip connection.
    Optional Dropout2d before each BN and optional CBAM after the block.

    Parameters
    ----------
    use_cbam    : Append a CBAM module after the residual add + ReLU.
    dropout_rate: If > 0, Dropout2d(p) is inserted before each BN.
    cbam_reduction: Channel reduction ratio for CBAM.
    """

    def __init__(
        self,
        in_channels:    int,
        out_channels:   int,
        num_layers:     int   = 2,
        use_cbam:       bool  = True,
        dropout_rate:   float = 0.0,
        cbam_reduction: int   = 16,
        kernel_size:    int   = 3,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(num_layers):
            ch_in = in_channels if i == 0 else out_channels
            if dropout_rate > 0:
                layers.append(nn.Dropout2d(p=dropout_rate))
            layers += [
                nn.Conv2d(ch_in, out_channels, kernel_size=kernel_size,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        layers = layers[:-1]   # drop final ReLU; applied after residual add
        self.block = nn.Sequential(*layers)

        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_channels, cbam_reduction) if use_cbam else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.block(x) + self.skip(x))
        if self.cbam is not None:
            out = self.cbam(out)
        return out


# ---------------------------------------------------------------------------
# Encoder / Decoder blocks
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """
    num_res_blocks ResidualBlocks → MaxPool2d(2).
    Returns (pooled, skip) where skip feeds the corresponding decoder level.
    """

    def __init__(
        self,
        in_channels:    int,
        out_channels:   int,
        num_res_blocks: int   = 1,
        layers_per_block: int = 2,
        use_cbam:       bool  = True,
        dropout_rate:   float = 0.0,
        cbam_reduction: int   = 16,
    ):
        super().__init__()
        blocks: List[nn.Module] = [
            ResidualBlock(in_channels, out_channels, layers_per_block,
                          use_cbam, dropout_rate, cbam_reduction)
        ]
        for _ in range(num_res_blocks - 1):
            blocks.append(
                ResidualBlock(out_channels, out_channels, layers_per_block,
                              use_cbam, dropout_rate, cbam_reduction)
            )
        self.res_blocks = nn.Sequential(*blocks)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = self.res_blocks(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    """
    Optional AttentionGate on skip → bilinear upsample → concat →
    num_res_blocks ResidualBlocks.
    """

    def __init__(
        self,
        in_channels:         int,
        skip_channels:       int,
        out_channels:        int,
        num_res_blocks:      int   = 1,
        layers_per_block:    int   = 2,
        use_cbam:            bool  = True,
        use_attention_gates: bool  = True,
        dropout_rate:        float = 0.0,
        cbam_reduction:      int   = 16,
        kernel_size:         int   = 2,
    ):
        super().__init__()
        self.attention_gate = (
            AttentionGate(skip_channels, in_channels)
            if use_attention_gates else None
        )
        merged = in_channels + skip_channels
        blocks: List[nn.Module] = [
            ResidualBlock(merged, out_channels, layers_per_block,
                          use_cbam, dropout_rate, cbam_reduction, kernel_size)
        ]
        for _ in range(num_res_blocks - 1):
            blocks.append(
                ResidualBlock(out_channels, out_channels, layers_per_block,
                              use_cbam, dropout_rate, cbam_reduction, kernel_size)
            )
        self.res_blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.attention_gate is not None:
            skip = self.attention_gate(skip, x)
        x = F.interpolate(x, size=skip.shape[2:],
                          mode="bilinear", align_corners=False)
        return self.res_blocks(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# ACORN-Net
# ---------------------------------------------------------------------------

class ACORN(nn.Module):
    """
    ACORN — Attention COnvolutional Residual Network.

    A Residual U-Net with independently toggleable CBAM and Attention Gates.

    Parameters
    ----------
    in_channels          : Input image channels.
    out_channels         : Output channels / segmentation classes.
    base_channels        : Channels at the first encoder level.
    depth                : Number of encoder/decoder levels (excl. bottleneck).
    num_res_blocks       : Residual blocks per encoder/decoder/bottleneck level.
    layers_per_block     : Conv layers inside each residual block (>= 1).
    channel_mult         : Per-level channel multiplier (e.g. 2.0 doubles each level).
    cbam_reduction       : Channel reduction ratio inside CBAM (default 16).
    use_cbam             : If True, CBAM is added inside every residual block.
    use_attention_gates  : If True, AttentionGates are applied on all skip connections.
    dropout_rate         : Dropout2d probability in bottleneck blocks (0 = disabled).
    dropout_depth        : Apply dropout to this many deepest encoder/decoder levels
                           in addition to the bottleneck (0 = bottleneck only).
    output_size          : Optional (H, W) to resize the final output.
    input_size           : Optional (H, W) checked at construction time.
    debug                : If True, print channel summary on construction.
    """

    def __init__(
        self,
        in_channels:          int                       = 1,
        out_channels:         int                       = 1,
        base_channels:        int                       = 64,
        depth:                int                       = 4,
        num_res_blocks:       int                       = 1,
        layers_per_block:     int                       = 2,
        channel_mult:         float                     = 2.0,
        cbam_reduction:       int                       = 16,
        use_cbam:             bool                      = True,
        use_attention_gates:  bool                      = True,
        dropout_rate:         float                     = 0.0,
        dropout_depth:        int                       = 0,
        output_size:          Optional[Tuple[int, int]] = None,
        input_size:           Optional[Tuple[int, int]] = None,
        debug:                bool                      = True,
    ):
        super().__init__()

        # ── Validation ───────────────────────────────────────────────────────
        if depth < 1:
            raise ValueError(f"depth={depth} must be >= 1.")
        if num_res_blocks < 1:
            raise ValueError(f"num_res_blocks={num_res_blocks} must be >= 1.")
        if layers_per_block < 1:
            raise ValueError(f"layers_per_block={layers_per_block} must be >= 1.")
        if base_channels < 1:
            raise ValueError(f"base_channels={base_channels} must be >= 1.")
        if channel_mult <= 0:
            raise ValueError(f"channel_mult={channel_mult} must be > 0.")
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"dropout_rate={dropout_rate} must be in [0, 1).")
        if dropout_depth < 0 or dropout_depth > depth:
            raise ValueError(
                f"dropout_depth={dropout_depth} must be in [0, depth={depth}]."
            )

        self.output_size         = output_size
        self.depth               = depth
        self.use_cbam            = use_cbam
        self.use_attention_gates = use_attention_gates

        # ── Channel progression ──────────────────────────────────────────────
        enc_channels: List[int] = [
            max(1, round(base_channels * (channel_mult ** i)))
            for i in range(depth)
        ]
        bot_channels: int = max(1, round(enc_channels[-1] * channel_mult))

        # ── Spatial size check ───────────────────────────────────────────────
        self._min_spatial = 2 ** depth
        if input_size is not None:
            h, w = input_size
            if h < self._min_spatial or w < self._min_spatial:
                raise ValueError(
                    f"input_size=({h}, {w}) too small for depth={depth}. "
                    f"Both H and W must be >= 2^depth = {self._min_spatial}."
                )

        # Helper: should level lvl (0=shallowest) receive dropout?
        def _do(lvl: int) -> float:
            return dropout_rate if lvl >= depth - dropout_depth else 0.0

        # ── Stem ─────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, enc_channels[0],
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(enc_channels[0]),
            nn.ReLU(inplace=True),
        )

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        for i in range(depth):
            ch_in  = enc_channels[i - 1] if i > 0 else enc_channels[0]
            ch_out = enc_channels[i]
            self.encoders.append(EncoderBlock(
                ch_in, ch_out,
                num_res_blocks   = num_res_blocks,
                layers_per_block = layers_per_block,
                use_cbam         = use_cbam,
                dropout_rate     = _do(i),
                cbam_reduction   = cbam_reduction,
            ))

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            *[ResidualBlock(
                enc_channels[-1] if j == 0 else bot_channels,
                bot_channels,
                num_layers     = layers_per_block,
                use_cbam       = use_cbam,
                dropout_rate   = dropout_rate,   # always gets dropout
                cbam_reduction = cbam_reduction,
              ) for j in range(num_res_blocks)]
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoders = nn.ModuleList()
        prev_channels = bot_channels
        for i in reversed(range(depth)):
            skip_ch = enc_channels[i]
            dec_out = enc_channels[i]
            self.decoders.append(DecoderBlock(
                prev_channels, skip_ch, dec_out,
                num_res_blocks       = num_res_blocks,
                layers_per_block     = layers_per_block,
                use_cbam             = use_cbam,
                use_attention_gates  = use_attention_gates,
                dropout_rate         = _do(i),
                cbam_reduction       = cbam_reduction,
            ))
            prev_channels = dec_out

        # ── Output head ──────────────────────────────────────────────────────
        self.head = nn.Conv2d(enc_channels[0], out_channels, kernel_size=1)

        if debug:
            print(self.summary())

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.dim() != 4:
            raise ValueError(
                f"Expected 4-D input (batch, C, H, W), got shape {tuple(x.shape)}."
            )
        _, in_ch, h, w = x.shape
        expected = self.stem[0].in_channels
        if in_ch != expected:
            raise ValueError(
                f"Input has {in_ch} channel(s); model expects {expected}."
            )
        if h < self._min_spatial or w < self._min_spatial:
            raise ValueError(
                f"Input ({h}×{w}) too small for depth={self.depth}. "
                f"Min size: {self._min_spatial}×{self._min_spatial}."
            )

        x = self.stem(x)

        skips: List[torch.Tensor] = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        x = self.head(x)

        if self.output_size is not None:
            x = F.interpolate(x, size=self.output_size,
                              mode="bilinear", align_corners=False)
        return x

    # ─────────────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable config and channel progression."""
        enc_ch = [enc.res_blocks[0].block[0 if not isinstance(enc.res_blocks[0].block[0], nn.Dropout2d) else 1].out_channels
                  for enc in self.encoders]
        bot_ch = self.bottleneck[-1].block[-2].out_channels  # last BN out

        attn_str = []
        if self.use_cbam:
            attn_str.append("CBAM")
        if self.use_attention_gates:
            attn_str.append("AttentionGates")
        attn_label = " + ".join(attn_str) if attn_str else "none"

        lines = [
            "─" * 52,
            f"  ACORN-Net  |  attention: {attn_label}",
            "─" * 52,
            (f"  in → stem({enc_ch[0]}) → "
             + " → ".join(f"enc{i+1}({c})" for i, c in enumerate(enc_ch))
             + f" → bot({bot_ch}) → "
             + " → ".join(f"dec{i+1}({c})" for i, c in enumerate(reversed(enc_ch)))
             + f" → head({self.head.out_channels})"),
            f"  params: {sum(p.numel() for p in self.parameters()):,}",
            "─" * 52,
        ]
        return "\n".join(lines)



