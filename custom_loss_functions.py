"""
custom_loss_functions.py
========================

Loss functions for ACORN training.

ACORN produces a two-channel output -- a posterior mean and a posterior
standard deviation -- for every cell of the 50 x 24 (MLAT x MLT) grid.
The losses here fall into two families:

  * distributional (CRPS), which score both output channels
  * deterministic (MSE), which score the mean channel only

Both families have a plain and a frequency-weighted variant, so runs can
be compared while changing only the scoring rule.

Contents
--------
gaussian_crps               Closed-form CRPS for a Gaussian forecast.
split_mean_std              Splits a two-channel prediction tensor.
CRPS                        Plain Gaussian CRPS.
BinWeightedLoss             Shared bin-weighting machinery.
WeightedCRPS                Frequency-weighted CRPS.
WeightedMeanSquaredError    Frequency-weighted MSE.
create_bin_weights          Builds (weights, hist, bin_edges).
build_loss                  Factory keyed on the config "loss" string.

Selecting a loss
----------------
The `"loss"` key in config.json selects the
objective via build_loss(). Recognised values are listed in LOSS_REGISTRY
below.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

pd.options.mode.chained_assignment = None

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# Lower bound applied to the standard deviation inside gaussian_crps.
# Clamping at a small positive value does two jobs at once: it keeps the
# division finite when the model predicts near-zero spread, and it
# rejects the negative values the output head can otherwise produce.
STD_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# Shared CRPS kernel
# ---------------------------------------------------------------------------

def gaussian_crps(epsilon: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
    """
    Closed-form CRPS for a Gaussian forecast, evaluated elementwise.

    For a forecast N(mu, sig) and observation y, with the absolute error
    epsilon = |y - mu| and the standardised error z = epsilon / sig:

        CRPS = sig * [ z * erf(z / sqrt(2))
                       + sqrt(2/pi) * exp(-z^2 / 2)
                       - 1 / sqrt(pi) ]

    This is the standard result specialised to epsilon >= 0, where the
    usual `2*Phi(z) - 1` term reduces to `erf(z / sqrt(2))`. Lower is
    better, and the score carries the same physical units as the target
    (here uA/m^2).

    Standard deviation positivity
    -----------------------------
    sig is clamped to a minimum of STD_FLOOR before use. ACORN's output
    head ends in a linear projection to out_channels with no positivity
    activation -- true of the ConvRefinementHead used in the final model
    as much as of the plain head -- so the second output channel is
    unbounded and can go negative. A negative sig makes this expression
    return a negative value, which is not a valid score and cannot be
    compared against the positive scores of later epochs. Historically
    that let epoch 1 post an unbeatable "best" loss and silently
    disabled early stopping.

    Clamping is done here, inside the score, rather than by transforming
    the output channel. That matters: clamping leaves every already-valid
    sig untouched, so the std channel means the same thing during
    training and at inference and nothing downstream needs a matching
    transform. An activation such as softplus would instead make the
    channel a pre-activation, and every consumer of it -- inference.py,
    the evaluation notebook, uncertainty propagation -- would have to
    apply the same function to recover a true standard deviation.

    Note the gradient is zero for clamped elements, so a unit that has
    driven its sig negative receives no signal through this term.

    Args:
        epsilon: Absolute error |y - mu|, any shape.
        sig:     Predicted standard deviation, broadcastable to epsilon.

    Returns:
        Elementwise CRPS, same shape as epsilon.
    """
    sig = torch.clamp(sig, min=STD_FLOOR)

    return torch.mul(
        sig,
        torch.add(
            torch.mul(
                torch.div(epsilon, sig),
                torch.erf(torch.div(epsilon, torch.mul(np.sqrt(2), sig))),
            ),
            torch.sub(
                torch.mul(
                    torch.sqrt(torch.div(2, np.pi)),
                    torch.exp(
                        torch.div(
                            torch.mul(-1, torch.pow(epsilon, 2)),
                            torch.mul(2, torch.pow(sig, 2)),
                        )
                    ),
                ),
                torch.div(1, torch.sqrt(torch.tensor(np.pi))),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Channel splitting
# ---------------------------------------------------------------------------

def find_channel_dim(y_pred: torch.Tensor) -> int:
    """
    Locate the axis holding the (mean, std) pair.

    The channel axis sits at a different position depending on the model
    head -- axis 1 for a 2D conv output (batch, 2, mlat, mlt), axis 2 for
    a flat output (batch, features, 2) -- so it is found by searching for
    the axis of length 2 rather than hardcoded.

    Axis 0 is never considered: it is the batch axis, and a trailing
    partial batch of exactly 2 samples would otherwise be mistaken for
    the channel axis.

    Args:
        y_pred: Prediction tensor with a length-2 channel axis.

    Returns:
        int: index of the channel axis.

    Raises:
        ValueError: if no non-batch axis has length 2 (usually means
            model_config sets out_channels != 2), or if more than one
            does, leaving the split ambiguous.
    """
    candidates = [d for d in range(1, y_pred.dim()) if y_pred.shape[d] == 2]

    if not candidates:
        raise ValueError(
            f'No non-batch axis of length 2 in prediction of shape '
            f'{tuple(y_pred.shape)}. Check that model_config sets '
            'out_channels=2.'
        )
    if len(candidates) > 1:
        raise ValueError(
            f'Ambiguous channel axis: prediction of shape '
            f'{tuple(y_pred.shape)} has length-2 axes at {candidates}. '
            'Cannot tell which holds (mean, std).'
        )

    return candidates[0]


def split_mean_std(y_pred: torch.Tensor, channel_dim: int):
    """
    Split a two-channel prediction into (mean, std) along channel_dim.

    Args:
        y_pred:      Prediction tensor.
        channel_dim: Axis to unbind, from find_channel_dim.

    Returns:
        (mean, std) tensors, each with channel_dim removed.
    """
    return torch.unbind(y_pred, dim=channel_dim)


class ProbabilisticLossMixin:
    """
    Caches the channel axis across calls for the distributional losses.

    Every batch in a run shares a layout, so the axis is resolved from
    the first batch seen and reused thereafter rather than re-derived on
    every forward pass. `channel_dim` starts as None and is filled in on
    first use.

    If a run legitimately changes output rank partway through, reset
    `channel_dim` to None to force re-detection.
    """

    channel_dim = None

    def resolve_channel_dim(self, y_pred: torch.Tensor) -> int:
        if self.channel_dim is None:
            self.channel_dim = find_channel_dim(y_pred)
        return self.channel_dim


# ---------------------------------------------------------------------------
# Unweighted CRPS
# ---------------------------------------------------------------------------

class CRPS(nn.Module, ProbabilisticLossMixin):
    """
    Gaussian CRPS, averaged uniformly over every element of the batch.

    Every grid cell contributes equally, so the objective reflects the
    natural distribution of FAC magnitudes in the training set. Compare
    against WeightedCRPS to isolate the effect of the frequency
    weighting.
    """

    # Emits a distribution (mean, std) rather than a point estimate, so the
    # model's output carries a channel per parameter. Read by evaluation code
    # that needs to know how to interpret the channel axis.
    is_probabilistic = True

    def __init__(self):
        super(CRPS, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Two-channel prediction with a length-2 channel axis.
            y_true: Observed FAC, broadcastable to the mean channel.

        Returns:
            Scalar mean CRPS over the batch.
        """
        mean, std = split_mean_std(y_pred, self.resolve_channel_dim(y_pred))

        # Trailing singleton axis keeps mean/std/y_true mutually
        # broadcastable regardless of the incoming rank.
        mean = mean.unsqueeze(-1)
        std = std.unsqueeze(-1)
        y_true = y_true.unsqueeze(-1)

        return torch.mean(gaussian_crps(torch.abs(y_true - mean), std))


# ---------------------------------------------------------------------------
# Frequency weighting
# ---------------------------------------------------------------------------

def create_bin_weights(y_train, num_bins=50, range_min=None, range_max=None):
    """
    Build inverse-frequency weights over |target| magnitude bins.

    FAC magnitude is heavily right-skewed: most cells at most times sit
    near zero, and the large-amplitude events we most want to forecast
    are rare. Weighting each sample by the reciprocal of its bin's
    relative frequency raises the training signal from those rare cells.

    Binning uses |y_train|, so one bin covers both signs (upward and
    downward FAC of equal magnitude share a weight).

    Empty bins would give an infinite weight, so zero-count bins are set
    to NaN and linearly interpolated from their neighbours before the
    reciprocal is taken.

    Args:
        y_train:   Training targets, any shape (flattened internally).
        num_bins:  Bin count, or an explicit sequence of bin edges.
        range_min: Lower histogram edge. Defaults to min(|y_train|).
        range_max: Upper histogram edge. Defaults to max(|y_train|).

    Returns:
        weights:   (num_bins,) inverse-frequency weight per bin.
        hist:      (num_bins,) normalised relative frequency per bin.
        bin_edges: (num_bins + 1,) bin edges.

    Note:
        Pass `bin_edges` and `weights` straight through to the weighted
        losses -- their constructors expect the edges array, not the
        bin count.
    """
    if range_min is None:
        range_min = min(np.abs(y_train.flatten()))
    if range_max is None:
        range_max = max(np.abs(y_train.flatten()))

    hist, bin_edges = np.histogram(
        np.abs(y_train), bins=num_bins, range=(range_min, range_max), density=False
    )
    hist = hist / np.sum(hist)   # relative, not absolute, frequency

    print(f'Hist results: {hist}')

    # Zero-count bins -> NaN -> linearly interpolated, so 1/x stays finite.
    inverse_weights = (
        pd.Series(np.where(hist == 0, np.nan, hist))
        .interpolate(method='linear')
        .to_numpy()
    )
    weights = 1 / inverse_weights

    return weights, hist, bin_edges


class BinWeightedLoss(nn.Module):
    """
    Base class for losses weighted by target-magnitude frequency.

    Holds the bin edges and per-bin weights from create_bin_weights and
    exposes `weights_for(y_true)`, mapping a flat target tensor to its
    per-element weights.

    Edges and weights are registered as buffers rather than plain
    attributes so they follow the module across `.to(device)` and are
    captured in `state_dict()` -- meaning a reloaded checkpoint reuses
    the exact weighting scheme it was trained with.

    Args:
        bin_edges:   (num_bins + 1,) edges from create_bin_weights.
        bin_weights: (num_bins,) weights from create_bin_weights.
    """

    def __init__(self, bin_edges, bin_weights):
        super().__init__()

        # bin_edges[:-1] holds the LEFT edge of each bin, the form
        # torch.searchsorted needs in weights_for below.
        # as_tensor avoids the copy-construct warning when a caller passes
        # tensors rather than numpy arrays, and copies only when needed.
        self.register_buffer(
            "bin_edges",
            torch.as_tensor(bin_edges[:-1], dtype=torch.float32).clone()
        )
        self.register_buffer(
            "bin_weights",
            torch.as_tensor(bin_weights, dtype=torch.float32).clone()
        )

        self.num_bins = len(bin_weights)
        self.range_min = bin_edges[0]
        self.range_max = bin_edges[-1]
        self.bin_width = (self.range_max - self.range_min) / self.num_bins

    def weights_for(self, y_true_flat: torch.Tensor) -> torch.Tensor:
        """
        Look up the frequency weight for every element of a flat target.

        searchsorted on the left edges returns the insertion index, so
        subtracting 1 gives the containing bin. Targets outside the
        histogram range are clamped into the end bins rather than
        dropped, so out-of-range magnitudes still contribute.

        Uses |y_true| to match the magnitude binning in
        create_bin_weights.
        """
        bindices = torch.searchsorted(
            self.bin_edges, torch.abs(y_true_flat), right=False
        ) - 1
        bindices = torch.clamp(bindices, 0, len(self.bin_weights) - 1)
        return self.bin_weights[bindices].to(DEVICE)


class WeightedCRPS(BinWeightedLoss, ProbabilisticLossMixin):
    """
    Inverse-frequency-weighted Gaussian CRPS.

    The training objective for the published ACORN Sci and Op models.
    Scores the full predicted distribution while upweighting rare
    large-amplitude cells via create_bin_weights.

    Args:
        bin_edges:   (num_bins + 1,) edges from create_bin_weights.
        bin_weights: (num_bins,) weights from create_bin_weights.
    """

    # Emits a distribution (mean, std) rather than a point estimate, so the
    # model's output carries a channel per parameter. Read by evaluation code
    # that needs to know how to interpret the channel axis.
    is_probabilistic = True

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Two-channel prediction with a length-2 channel axis.
            y_true: Observed FAC, broadcastable to the mean channel.

        Returns:
            Scalar weighted-mean CRPS over the batch.
        """
        mean, std = split_mean_std(y_pred, self.resolve_channel_dim(y_pred))

        # Flatten to 1D: weighting is per grid cell, so the spatial
        # layout carries no information the loss needs.
        y_true = y_true.flatten()
        mean = mean.flatten()
        std = std.flatten()

        weights = self.weights_for(y_true)
        crps = gaussian_crps(torch.abs(y_true - mean), std)

        return torch.mean(torch.mul(weights, crps))


class WeightedMeanSquaredError(BinWeightedLoss):
    """
    Inverse-frequency-weighted mean squared error.

    Scores the mean channel only, sharing its weighting scheme with
    WeightedCRPS so runs differ solely in the scoring rule.

    Expects a SINGLE-channel y_pred -- passing a two-channel ACORN output
    here treats the std channel as extra predictions and yields a
    meaningless number.

    Args:
        bin_edges:   (num_bins + 1,) edges from create_bin_weights.
        bin_weights: (num_bins,) weights from create_bin_weights.
    """

    # Point estimate: a single output channel.
    is_probabilistic = False

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Single-channel prediction.
            y_true: Observed FAC, same shape as y_pred.

        Returns:
            Scalar weighted-mean squared error over the batch.
        """
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)

        weights = self.weights_for(y_true)
        mean_squared = torch.pow(torch.sub(y_pred, y_true), 2)

        return torch.mean(torch.mul(weights, mean_squared))


# ---------------------------------------------------------------------------
# Config-driven selection
# ---------------------------------------------------------------------------

# Maps the config "loss" string to its class. Keys are lowercased and
# stripped before lookup, so "Weighted_CRPS" and "weighted_crps" match.
LOSS_REGISTRY = {
    'crps':          CRPS,
    'weighted_crps': WeightedCRPS,
    'mse':           WeightedMeanSquaredError,
    'weighted_mse':  WeightedMeanSquaredError,
}

# Losses that need bin_edges/bin_weights handed to their constructor.
WEIGHTED_LOSSES = {'weighted_crps', 'mse', 'weighted_mse'}


def build_loss(name, bin_edges=None, bin_weights=None):
    """
    Construct the loss named by the config "loss" key.

    Args:
        name:        Loss name; see LOSS_REGISTRY for valid values.
        bin_edges:   Required for the frequency-weighted losses.
        bin_weights: Required for the frequency-weighted losses.

    Returns:
        nn.Module: the configured loss.

    Raises:
        ValueError: on an unknown name, or if a weighted loss is
            requested without bin edges and weights.
    """
    key = str(name).strip().lower()

    if key not in LOSS_REGISTRY:
        raise ValueError(
            f'Unknown loss {name!r}. Valid options: '
            f'{sorted(LOSS_REGISTRY)}.'
        )

    kwargs = {}

    if key in WEIGHTED_LOSSES:
        if bin_edges is None or bin_weights is None:
            raise ValueError(
                f'Loss {name!r} is frequency-weighted and needs both '
                'bin_edges and bin_weights (see create_bin_weights).'
            )
        kwargs['bin_edges'] = bin_edges
        kwargs['bin_weights'] = bin_weights

    return LOSS_REGISTRY[key](**kwargs)
