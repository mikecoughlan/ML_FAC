"""
model_classes.py
=================
ACORN -- Attention COnvolutional Residual Network. This is the single,
consolidated model definition file: the attention and residual building
blocks (CBAM, AttentionGate, ResidualBlock, EncoderBlock, DecoderBlock)
and the spatial refinement head all live here, assembled by one ACORN
class.

model_training.py calls ACORN(**model_config), and model_config comes
straight from config.json, so what is in that block is what gets built.
Attention is switchable via use_cbam / use_attention_gates; the
refinement head is always built and its width, depth and dropout are
adjustable through conv_head_hidden_channels / conv_head_num_layers /
conv_head_dropout.

Unrecognized kwargs
-------------------
A kwarg this class does not recognize is accepted and ignored with a
printed warning rather than raising TypeError -- a safety net against a
config carrying a stray or future key, not a substitute for wiring up a
kwarg you intend to have an effect. Keys belonging to features that no
longer exist raise instead, since silently ignoring those would build
something other than what the config asked for.

MLT wrap-padding convention
---------------------------
The training pipeline pads AMPERE targets with Y[:, -1:] (column 23) at
the front, giving [23, 0, 1, ..., 23, 0]. THis is removed in the model
evaluation stage so the final results is teh normal 50x24 array
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════════
# Attention modules
# ═══════════════════════════════════════════════════════════════════════════

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
        mx = self.mlp(self.max_pool(x))
        scale = self.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class SpatialAttention(nn.Module):
    """Spatial attention map (second half of CBAM)."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * scale


class CBAM(nn.Module):
    """Full Convolutional Block Attention Module (channel -> spatial)."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class AttentionGate(nn.Module):
    """Additive attention gate for skip connections."""

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
        g_up = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        alpha = self.psi(self.relu(self.W_x(x) + self.W_g(g_up)))
        return x * alpha


# ═══════════════════════════════════════════════════════════════════════════
# Core residual block
# ═══════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        use_cbam: bool = True,
        dropout_rate: float = 0.0,
        cbam_reduction: int = 16,
        kernel_size: int = 3,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(num_layers):
            ch_in = in_channels if i == 0 else out_channels
            if dropout_rate > 0:
                layers.append(nn.Dropout2d(p=dropout_rate))
            layers += [
                nn.Conv2d(ch_in, out_channels, kernel_size=kernel_size, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        layers = layers[:-1]
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


# ═══════════════════════════════════════════════════════════════════════════
# Encoder / Decoder blocks
# ═══════════════════════════════════════════════════════════════════════════

class EncoderBlock(nn.Module):
    '''
    Block made up of residual sections with skip connections. Reduces
    the dimensions of it's output by using a pooling layer at the end,
    distinguishing it from teh decoder block which expands the dimensions.
    Also generates the array used for the U-Net skip connections.

    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int = 1,
        layers_per_block: int = 2,
        use_cbam: bool = True,
        dropout_rate: float = 0.0,
        cbam_reduction: int = 16,
    ):
        super().__init__()
        blocks: List[nn.Module] = [
            ResidualBlock(in_channels, out_channels, layers_per_block, use_cbam, dropout_rate, cbam_reduction)
        ]
        for _ in range(num_res_blocks - 1):
            blocks.append(
                ResidualBlock(out_channels, out_channels, layers_per_block, use_cbam, dropout_rate, cbam_reduction)
            )
        self.res_blocks = nn.Sequential(*blocks)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Call for the encoder block

        Args:
            x (torch.Tensor): output of the previous layer

        Returns:
            torch.Tensor: pooled output of the encoder block
            torch.Tensor: non-pooled output used to make U-Net skip connection
        '''
        skip = self.res_blocks(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    '''
    Block made up of residual sections with skip connections. Expands
    the dimensions of its output using a bi-linear interpolation.
    '''
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_res_blocks: int = 1,
        layers_per_block: int = 2,
        use_cbam: bool = True,
        use_attention_gates: bool = True,
        dropout_rate: float = 0.0,
        cbam_reduction: int = 16,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.attention_gate = AttentionGate(skip_channels, in_channels) if use_attention_gates else None
        merged = in_channels + skip_channels
        blocks: List[nn.Module] = [
            ResidualBlock(merged, out_channels, layers_per_block, use_cbam, dropout_rate, cbam_reduction, kernel_size)
        ]
        for _ in range(num_res_blocks - 1):
            blocks.append(
                ResidualBlock(out_channels, out_channels, layers_per_block, use_cbam, dropout_rate, cbam_reduction, kernel_size)
            )
        self.res_blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        '''
        Call for the decoder block

        Args:
            x (torch.Tensor): output of the previous layer
            skip (torch.Tensor): output of the conjugate encoder block

        Returns:
            torch.Tensor: expanded output of the decoder block, combined with encoder skip output
        '''
        if self.attention_gate is not None:
            skip = self.attention_gate(skip, x)
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.res_blocks(torch.cat([x, skip], dim=1))


class ConvRefinementHead(nn.Module):
    """
    Small spatial refinement block: `num_layers` kernel_size=3 conv layers
    (BatchNorm + ReLU, optional dropout), then a final 1x1 projection to
    out_channels.

    `out_channels` is exposed as a plain attribute (not inferred from
    nn.Sequential, which has none) so ACORN.summary() keeps working.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: int = 32, num_layers: int = 2, dropout_rate: float = 0.0):
        super().__init__()
        self.out_channels = out_channels

        layers = []
        ch = in_channels
        for _ in range(num_layers):
            layers += [
                nn.Conv2d(ch, hidden_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
            ]
            if dropout_rate > 0:
                layers.append(nn.Dropout2d(dropout_rate))
            ch = hidden_channels

        layers.append(nn.Conv2d(ch, out_channels, kernel_size=1))   # final per-pixel projection
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ═══════════════════════════════════════════════════════════════════════════
# ACORN
# ═══════════════════════════════════════════════════════════════════════════

class ACORN(nn.Module):
    """
    ACORN -- Attention COnvolutional Residual Network.

    A Residual U-Net with independently toggleable CBAM and Attention
    Gates, followed by a spatial refinement head.

    Note the encoder/decoder axes are time-history x input-feature, not
    space. The output is resized to the MLAT x MLT grid only at the final
    interpolation step, which is the first and only point at which the
    spatial layout is meaningful -- and therefore where the refinement
    head operates.

    Core parameters
    ---------------
    in_channels, out_channels, base_channels, depth, num_res_blocks,
    layers_per_block, channel_mult, cbam_reduction, use_cbam,
    use_attention_gates, dropout_rate, dropout_depth, output_size,
    input_size, debug -- see the individual block classes above.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        num_res_blocks: int = 1,
        layers_per_block: int = 2,
        channel_mult: float = 2.0,
        cbam_reduction: int = 16,
        use_cbam: bool = True,
        use_attention_gates: bool = True,
        dropout_rate: float = 0.0,
        dropout_depth: int = 0,
        output_size: Optional[Tuple[int, int]] = None,
        input_size: Optional[Tuple[int, int]] = None,
        debug: bool = True,
        # ── conv refinement head ─────────────────────────────────────────────
        conv_head_hidden_channels: int = 32,
        conv_head_num_layers: int = 2,
        conv_head_dropout: float = 0.0,
        # ── safety net -- absorbs unrecognized kwargs instead of raising ────
        **extra_kwargs,
    ):
        super().__init__()

        # Keys from architecture options that no longer exist. Absorbing
        # these into extra_kwargs would print a soft warning and then
        # build the default anyway, so a config still requesting one
        # would silently get something else. Raise instead.
        removed_keys = sorted(k for k in extra_kwargs if k in REMOVED_KWARGS)
        if removed_keys:
            raise TypeError(
                f"{removed_keys} are no longer parameters of ACORN. Remove "
                "them from model_config. A checkpoint trained with one of "
                "these needs retraining or an older revision of this file."
            )

        if extra_kwargs:
            print(
                f"WARNING: ACORN received {len(extra_kwargs)} unrecognized kwarg(s), "
                f"ignored: {sorted(extra_kwargs.keys())}. If any of these were meant "
                "to do something, wire them into __init__ explicitly -- this catch-all "
                "is a safety net against crashes, not a substitute for actually using them."
            )

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
            raise ValueError(f"dropout_depth={dropout_depth} must be in [0, depth={depth}].")

        self.output_size = output_size
        self.depth = depth
        self.use_cbam = use_cbam
        self.use_attention_gates = use_attention_gates

        # ── Channel progression ──────────────────────────────────────────────
        enc_channels: List[int] = [max(1, round(base_channels * (channel_mult ** i))) for i in range(depth)]
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

        def _do(lvl: int) -> float:
            return dropout_rate if lvl >= depth - dropout_depth else 0.0

        # ── Stem ─────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, enc_channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(enc_channels[0]),
            nn.ReLU(inplace=True),
        )

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        for i in range(depth):
            ch_in = enc_channels[i - 1] if i > 0 else enc_channels[0]
            ch_out = enc_channels[i]
            self.encoders.append(EncoderBlock(
                ch_in, ch_out,
                num_res_blocks=num_res_blocks, layers_per_block=layers_per_block,
                use_cbam=use_cbam, dropout_rate=_do(i), cbam_reduction=cbam_reduction,
            ))

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(*[
            ResidualBlock(
                enc_channels[-1] if j == 0 else bot_channels, bot_channels,
                num_layers=layers_per_block, use_cbam=use_cbam,
                dropout_rate=dropout_rate, cbam_reduction=cbam_reduction,
            ) for j in range(num_res_blocks)
        ])

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoders = nn.ModuleList()
        prev_channels = bot_channels
        for i in reversed(range(depth)):
            skip_ch = enc_channels[i]
            dec_out = enc_channels[i]
            self.decoders.append(DecoderBlock(
                prev_channels, skip_ch, dec_out,
                num_res_blocks=num_res_blocks, layers_per_block=layers_per_block,
                use_cbam=use_cbam, use_attention_gates=use_attention_gates,
                dropout_rate=_do(i), cbam_reduction=cbam_reduction,
            ))
            prev_channels = dec_out

        # ── Refinement head ────────────────────────────────────────────────
        head_in_channels = enc_channels[0]

        self.head = ConvRefinementHead(
            head_in_channels, out_channels,
            hidden_channels=conv_head_hidden_channels,
            num_layers=conv_head_num_layers,
            dropout_rate=conv_head_dropout,
        )
        self.conv_head_hidden_channels = conv_head_hidden_channels
        self.conv_head_num_layers = conv_head_num_layers

        if debug:
            print(self.summary())

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4-D input (batch, C, H, W), got shape {tuple(x.shape)}.")
        _, in_ch, h, w = x.shape
        expected = self.stem[0].in_channels
        if in_ch != expected:
            raise ValueError(f"Input has {in_ch} channel(s); model expects {expected}.")
        if h < self._min_spatial or w < self._min_spatial:
            raise ValueError(
                f"Input ({h}x{w}) too small for depth={self.depth}. "
                f"Min size: {self._min_spatial}x{self._min_spatial}."
            )

        x = self.stem(x)

        skips: List[torch.Tensor] = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        if self.output_size is not None:
            x = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)

        x = self.head(x)
        return x

    # ─────────────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        enc_ch = [
            enc.res_blocks[0].block[0 if not isinstance(enc.res_blocks[0].block[0], nn.Dropout2d) else 1].out_channels
            for enc in self.encoders
        ]
        bot_ch = self.bottleneck[-1].block[-2].out_channels

        attn_str = []
        if self.use_cbam:
            attn_str.append("CBAM")
        if self.use_attention_gates:
            attn_str.append("AttentionGates")
        attn_label = " + ".join(attn_str) if attn_str else "none"

        head_label = (f"ConvRefinementHead({self.conv_head_num_layers}L, "
                      f"{self.conv_head_hidden_channels}ch)")

        lines = [
            "-" * 52,
            f"  ACORN-Net  |  attention: {attn_label}  |  head: {head_label}",
            "-" * 52,
            (f"  in -> stem({enc_ch[0]}) -> "
             + " -> ".join(f"enc{i+1}({c})" for i, c in enumerate(enc_ch))
             + f" -> bot({bot_ch}) -> "
             + " -> ".join(f"dec{i+1}({c})" for i, c in enumerate(reversed(enc_ch)))
             + f" -> head({self.head.out_channels})"),
            f"  params: {sum(p.numel() for p in self.parameters()):,}",
            "-" * 52,
        ]
        return "\n".join(lines)
