"""
model_classes.py
=================
ACORN -- Attention COnvolutional Residual Network. This is the single,
consolidated model definition file: architecture (CBAM, AttentionGate,
ResidualBlock, EncoderBlock, DecoderBlock), RFF/sham position encoding, and
the spatial conv refinement head all live here as one ACORN class with
everything gated behind optional kwargs.

Nothing about model_training.py needs to change to use any of this --
model_training.py calls ACORN(**model_config), and model_config comes
straight from the config file. What's in that config's model_config block
controls what gets built:

    "use_rff_position": true, "rff_freq_scale_lat": 8.0, ...   -> RFF
    "use_sham_position": true, ...                              -> sham control
    "use_conv_head": false                                      -> plain 1x1 head (old default)
    (omit use_conv_head)                                        -> conv head (current default, see below)

DEFAULT CHANGED: use_conv_head now defaults to True -- the conv head
consistently outperformed the plain 1x1 head in testing, so it's the
default rather than something to opt into. Loading a checkpoint saved
BEFORE this default flipped requires passing use_conv_head=False
explicitly, or load_state_dict will fail on a head-shape mismatch.

Any kwarg this class doesn't recognize is accepted and ignored with a
printed warning rather than raising TypeError -- a safety net against a
config carrying a stray/future key, not a substitute for actually wiring up
a kwarg you intend to have an effect.

Why the conv head exists: the plain head is `nn.Conv2d(..., kernel_size=1)`
-- zero spatial receptive field, a fixed per-pixel linear combination of
channels with no access to neighboring pixels. Pixel-level CRPS delta maps
from the RFF-vs-sham ablation showed radial, sign-flipping structure right
at the R0/R1 boundary, consistent with the model trying to shift or sharpen
a boundary -- inherently a neighborhood operation a 1x1 conv can't do.
The conv head inserts 1-2 real (kernel_size=3) conv layers between the
RFF/sham concatenation and the final per-pixel projection, giving the
network that mechanism. (Result: the conv head alone, with NO position
information, outperformed every RFF/sham variant tested with the plain
head -- see project notes. RFF/sham on top of the conv head so far
haven't beaten the conv head alone.)

IMPORTANT -- MLT wrap-padding convention:
  The training pipeline pads AMPERE targets with Y[:,-1:] (column 23) at
  the front -- [23, 0, 1, ..., 23, 0]. RFFPositionEncoding's default
  `wrap_front_source_idx=-1` matches this. Only pass `wrap_front_source_idx=-2`
  if deliberately working with an old checkpoint that used the since-fixed
  buggy Y[:,-2:-1] (column 22) convention.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Attention modules (unchanged from model_classes.py)
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
    """Additive attention gate for skip connections (Oktay et al., 2018)."""

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
# Core residual block (unchanged from model_classes.py)
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
# Encoder / Decoder blocks (unchanged from model_classes.py)
# ═══════════════════════════════════════════════════════════════════════════

class EncoderBlock(nn.Module):
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
        skip = self.res_blocks(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
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
        if self.attention_gate is not None:
            skip = self.attention_gate(skip, x)
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.res_blocks(torch.cat([x, skip], dim=1))


# ═══════════════════════════════════════════════════════════════════════════
# RFF position encoding
# ═══════════════════════════════════════════════════════════════════════════

def _build_wrapped_mlt_hours(n_mlt: int, wrap_front_source_idx: int) -> torch.Tensor:
    """Real MLT hours at each of the (n_mlt + 2) wrap-padded column positions,
    e.g. wrap_front_source_idx=-1 -> [23, 0, 1, ..., 23, 0] (the originally-
    intended scheme, matching the FIXED pipeline). Use -2 only if working
    with the old checkpoint's buggy [22, 0, ..., 23, 0] convention.
    """
    hours = torch.arange(n_mlt, dtype=torch.float32)
    idx = wrap_front_source_idx % n_mlt
    front = hours[idx:idx + 1]
    back = hours[0:1]
    return torch.cat([front, hours, back])


class RFFPositionEncoding(nn.Module):
    """
    Random Fourier Feature position encoding for a static (grid_rows,
    grid_cols) output grid, with independent frequency scales for MLAT and
    MLT.

    Each of the m random frequency vectors nu[i] is 3-dimensional -- one
    component per coordinate axis: [MLAT_norm, sin(MLT), cos(MLT)]. The
    MLAT component is drawn from N(0, freq_scale_lat^2); the two MLT
    components are drawn from N(0, freq_scale_mlt^2) -- every frequency
    still mixes both axes (MLAT-MLT-coupled structure is representable),
    but the typical magnitude along each axis is controlled independently.

    Handles both the natural (n_mlt) grid and the wrap-padded (n_mlt + 2)
    grid automatically, based on grid_cols vs. n_mlt.

    Optionally supplements this with a small deterministic harmonic block
    for MLT: cos(k*theta), sin(k*theta) for k=1..max_mlt_harmonic, the true
    Fourier harmonics of a periodic signal -- independent of, and appended
    alongside, the random features.
    """

    def __init__(
        self,
        grid_rows: int,
        grid_cols: int,
        num_frequencies: int = 32,
        freq_scale_lat: float = 5.0,
        freq_scale_mlt: float = 5.0,
        mlat_span_deg: float = 50.0,
        n_mlt: int = 24,
        wrap_front_source_idx: int = -1,   # matches the FIXED pipeline; see module docstring
        seed: int = 0,
        include_mlt_harmonics: bool = False,
        max_mlt_harmonic: int = 4,
    ):
        super().__init__()

        rows = torch.arange(grid_rows, dtype=torch.float32) + 0.5
        mlat_norm = rows / mlat_span_deg

        if grid_cols == n_mlt:
            mlt_hours = torch.arange(n_mlt, dtype=torch.float32) + 0.5
        elif grid_cols == n_mlt + 2:
            mlt_hours = _build_wrapped_mlt_hours(n_mlt, wrap_front_source_idx) + 0.5
        else:
            raise ValueError(
                f"grid_cols={grid_cols} matches neither the natural MLT grid "
                f"(n_mlt={n_mlt}) nor the wrap-padded grid (n_mlt+2={n_mlt+2}). "
                "Pass the correct n_mlt, or add a case for this grid shape."
            )
        mlt_frac = mlt_hours / n_mlt

        MLAT, MLT_FRAC = torch.meshgrid(mlat_norm, mlt_frac, indexing="ij")   # (H, W)
        mlt_theta = 2 * math.pi * MLT_FRAC

        coords = torch.stack([MLAT, torch.sin(mlt_theta), torch.cos(mlt_theta)], dim=0)   # (3, H, W)
        coords_flat = coords.reshape(3, -1)

        g = torch.Generator().manual_seed(seed)
        nu_lat = torch.randn(num_frequencies, 1, generator=g) * freq_scale_lat
        nu_mlt = torch.randn(num_frequencies, 2, generator=g) * freq_scale_mlt
        nu = torch.cat([nu_lat, nu_mlt], dim=1)   # (m, 3)

        proj = 2 * math.pi * (nu @ coords_flat)                       # (m, H*W)
        gamma = torch.cat([torch.cos(proj), torch.sin(proj)], dim=0)   # (2m, H*W)
        out_channels = 2 * num_frequencies

        if include_mlt_harmonics:
            theta_flat = mlt_theta.reshape(-1)
            k = torch.arange(1, max_mlt_harmonic + 1, dtype=torch.float32)
            phase = k[:, None] * theta_flat[None, :]
            harmonics = torch.cat([torch.cos(phase), torch.sin(phase)], dim=0)
            gamma = torch.cat([gamma, harmonics], dim=0)
            out_channels += 2 * max_mlt_harmonic

        gamma = gamma.reshape(1, out_channels, grid_rows, grid_cols)

        self.register_buffer("gamma", gamma)
        self.out_channels = out_channels

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.gamma.expand(batch_size, -1, -1, -1)


class NoisePositionEncoding(nn.Module):
    """
    Fixed random-noise channels, same interface as RFFPositionEncoding
    (out_channels attribute, forward(batch_size) -> fixed buffer expanded
    over batch). Exists purely as a capacity-matched SHAM CONTROL: if
    concatenating this instead of real RFF features produces a similar CRPS
    change, the RFF result isn't really about position information -- it's
    about the head having more input channels and training slightly
    differently as a result. Use the same out_channels count as whatever RFF
    config you're comparing against for a clean, isolated comparison.
    """

    def __init__(self, grid_rows: int, grid_cols: int, out_channels: int, seed: int = 999):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        noise = torch.randn(1, out_channels, grid_rows, grid_cols, generator=g)
        self.register_buffer("noise", noise)
        self.out_channels = out_channels

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.noise.expand(batch_size, -1, -1, -1)


# ═══════════════════════════════════════════════════════════════════════════
# Spatial conv refinement head
# ═══════════════════════════════════════════════════════════════════════════

class ConvRefinementHead(nn.Module):
    """
    Small spatial refinement block: `num_layers` kernel_size=3 conv layers
    (BatchNorm + ReLU, optional dropout), then a final 1x1 projection to
    out_channels -- structurally the same final projection the plain head
    does, just preceded by layers that can actually see neighboring pixels
    first (a 1x1 conv can't -- zero spatial receptive field).

    `out_channels` is exposed as a plain attribute (not inferred from
    nn.Sequential, which has none) so ACORN.summary() keeps working
    regardless of which head type is active.
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
# ACORN-Net
# ═══════════════════════════════════════════════════════════════════════════

class ACORN(nn.Module):
    """
    ACORN -- Attention COnvolutional Residual Network.

    A Residual U-Net with independently toggleable CBAM and Attention Gates,
    plus optional RFF position encoding concatenated onto the output right
    after the final resize (the only point where MLAT/MLT is actually known
    in this architecture -- the encoder/decoder axes are time-history x
    input-feature, not space).

    Core parameters (unchanged from model_classes.py)
    --------------------------------------------------
    in_channels, out_channels, base_channels, depth, num_res_blocks,
    layers_per_block, channel_mult, cbam_reduction, use_cbam,
    use_attention_gates, dropout_rate, dropout_depth, output_size,
    input_size, debug -- see original docstrings; behavior unchanged.

    RFF parameters (new)
    ---------------------
    use_rff_position       : If True, concatenate RFF position features
                              onto the interpolated output before head.
    rff_num_frequencies    : Number of random frequency pairs (m). Output
                              channels from the random block = 2m.
    rff_freq_scale_lat     : Frequency-scale (bandwidth) for the MLAT axis.
    rff_freq_scale_mlt     : Frequency-scale (bandwidth) for the MLT axis.
    rff_mlat_span_deg      : Normalization span for MLAT coordinates.
    rff_n_mlt              : Real number of MLT hours (24) -- used to detect
                              whether output_size's column count is the
                              natural grid or the wrap-padded grid.
    rff_wrap_front_source_idx : Which column gets duplicated at the front of
                              the wrap-padded grid. -1 matches the current
                              pipeline (originally-intended scheme); -2 only
                              for loading an old checkpoint that used the
                              since-fixed buggy convention. See module docstring.
    rff_seed                : Random seed for the fixed frequency draw.
    rff_include_mlt_harmonics : If True, append a deterministic MLT harmonic
                              block (true Fourier harmonics) alongside the
                              random features.
    rff_max_mlt_harmonic     : Highest harmonic order if the above is True.

    Sham-control parameters (new)
    -------------------------------
    use_sham_position       : If True, concatenate fixed random NOISE
                              channels instead of RFF position features --
                              mutually exclusive with use_rff_position.
                              Capacity-matched control for isolating whether
                              an RFF effect is really about position.
    sham_num_channels        : Channel count for the noise block. If None,
                              defaults to 2 * rff_num_frequencies so it's
                              easy to keep exactly capacity-matched to a
                              particular RFF config by reusing that config's
                              rff_num_frequencies value.
    sham_seed                : Random seed for the fixed noise draw.

    Conv-head parameters (new)
    ----------------------------
    use_conv_head            : Default True -- the conv head consistently
                              outperformed the plain 1x1 head in testing
                              (see project notes), so it's now the default
                              rather than something to opt into.
                              IMPORTANT: any OLD checkpoint saved before this
                              default flipped was trained with a plain 1x1
                              head. Loading one now requires passing
                              use_conv_head=False explicitly, or
                              load_state_dict will fail on a head-shape
                              mismatch.
    conv_head_hidden_channels : Channel width of the refinement layers.
    conv_head_num_layers      : How many kernel_size=3 layers before the
                              final 1x1 projection.
    conv_head_dropout         : Dropout rate inside the refinement block
                              (0.0 = none).

    Any other keyword arguments are accepted and ignored, with a printed
    warning -- a safety net so a new experimental kwarg doesn't raise
    TypeError, but NOT a substitute for actually wiring up a kwarg you
    intend to have an effect. If the warning fires unexpectedly, that's a
    bug to fix here, not something to silence.
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
        # ── RFF position encoding ──────────────────────────────────────────
        use_rff_position: bool = False,
        rff_num_frequencies: int = 32,
        rff_freq_scale_lat: float = 5.0,
        rff_freq_scale_mlt: float = 5.0,
        rff_mlat_span_deg: float = 50.0,
        rff_n_mlt: int = 24,
        rff_wrap_front_source_idx: int = -1,
        rff_seed: int = 0,
        rff_include_mlt_harmonics: bool = False,
        rff_max_mlt_harmonic: int = 4,
        # ── sham/noise control ──────────────────────────────────────────────
        use_sham_position: bool = False,
        sham_num_channels: Optional[int] = None,
        sham_seed: int = 999,
        # ── conv refinement head ─────────────────────────────────────────────
        use_conv_head: bool = True,
        conv_head_hidden_channels: int = 32,
        conv_head_num_layers: int = 2,
        conv_head_dropout: float = 0.0,
        # ── safety net -- absorbs unrecognized kwargs instead of raising ────
        **extra_kwargs,
    ):
        super().__init__()

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
        if use_rff_position and output_size is None:
            raise ValueError("use_rff_position=True requires output_size to be set.")
        if use_sham_position and output_size is None:
            raise ValueError("use_sham_position=True requires output_size to be set.")
        if use_rff_position and use_sham_position:
            raise ValueError(
                "use_rff_position and use_sham_position are mutually exclusive -- "
                "the sham control is meant to replace RFF, not combine with it."
            )

        self.output_size = output_size
        self.depth = depth
        self.use_cbam = use_cbam
        self.use_attention_gates = use_attention_gates
        self.use_rff_position = use_rff_position
        self.use_sham_position = use_sham_position
        self.use_conv_head = use_conv_head

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

        # ── RFF position encoding / sham control + head ─────────────────────
        if use_rff_position:
            self.rff_position = RFFPositionEncoding(
                grid_rows=output_size[0], grid_cols=output_size[1],
                num_frequencies=rff_num_frequencies,
                freq_scale_lat=rff_freq_scale_lat, freq_scale_mlt=rff_freq_scale_mlt,
                mlat_span_deg=rff_mlat_span_deg, n_mlt=rff_n_mlt,
                wrap_front_source_idx=rff_wrap_front_source_idx, seed=rff_seed,
                include_mlt_harmonics=rff_include_mlt_harmonics,
                max_mlt_harmonic=rff_max_mlt_harmonic,
            )
            self.sham_position = None
            head_in_channels = enc_channels[0] + self.rff_position.out_channels
        elif use_sham_position:
            self.rff_position = None
            n_sham = sham_num_channels if sham_num_channels is not None else 2 * rff_num_frequencies
            self.sham_position = NoisePositionEncoding(
                grid_rows=output_size[0], grid_cols=output_size[1],
                out_channels=n_sham, seed=sham_seed,
            )
            head_in_channels = enc_channels[0] + self.sham_position.out_channels
        else:
            self.rff_position = None
            self.sham_position = None
            head_in_channels = enc_channels[0]

        self.head = (
            ConvRefinementHead(
                head_in_channels, out_channels,
                hidden_channels=conv_head_hidden_channels, num_layers=conv_head_num_layers,
                dropout_rate=conv_head_dropout,
            )
            if use_conv_head else
            nn.Conv2d(head_in_channels, out_channels, kernel_size=1)
        )
        self._conv_head_hidden_channels = conv_head_hidden_channels
        self._conv_head_num_layers = conv_head_num_layers

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

        if self.use_rff_position:
            pos = self.rff_position(x.shape[0])
            x = torch.cat([x, pos], dim=1)
        elif self.use_sham_position:
            pos = self.sham_position(x.shape[0])
            x = torch.cat([x, pos], dim=1)

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

        rff_label = f"RFF({self.rff_position.out_channels}ch)" if self.use_rff_position else \
                    f"SHAM({self.sham_position.out_channels}ch)" if self.use_sham_position else "none"

        head_label = (f"ConvRefinementHead({self._conv_head_num_layers}L, "
                      f"{self._conv_head_hidden_channels}ch)" if self.use_conv_head else "1x1")

        lines = [
            "-" * 52,
            f"  ACORN-Net  |  attention: {attn_label}  |  position: {rff_label}  |  head: {head_label}",
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
