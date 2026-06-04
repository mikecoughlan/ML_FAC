# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import gc
import json
import os
import pickle
import shutil
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm

import shap
from data_prep import PreparingData
from model_classes import *

pd.options.mode.chained_assignment = None

os.environ["CDF_LIB"] = "~/CDF/lib"

working_dir = os.path.dirname(os.path.abspath(__file__))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
# DEVICE = torch.device('cpu')
# print(f'Device: {DEVICE}')

# Loading CONFIG json file
with open('sci_config.json', 'r') as f:
	CONFIG = json.load(f)

os.makedirs(CONFIG["model_dir"], exist_ok=True)

model_file = f'{CONFIG["model_dir"]}{CONFIG["model"]}_{CONFIG["version"]}_{CONFIG["eras"]}.pt'

# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------
# The MLAT axis has 50 bins running from the pole (index 0, MLAT_MAX) to the
# equatorward edge (index 49, MLAT_MIN).  Index 0 = pole is consistent with
# origin='upper' / colatitude-at-top used in all spatial plots in this project.
# Adjust MLAT_MIN / MLAT_MAX if your grid covers a different range.
MLAT_MIN = 50.0   # degrees MLAT at the equatorward edge (index N_MLAT - 1)
MLAT_MAX = 90.0   # degrees MLAT at the pole             (index 0)
N_MLAT   = 50
N_MLT    = 24

# ---------------------------------------------------------------------------
# Region definitions
# ---------------------------------------------------------------------------
# Each entry is a dict with keys:
#   mlat_low   (float): equatorward MLAT boundary in degrees
#   mlat_high  (float): poleward MLAT boundary in degrees
#   mlt_start  (float): start of MLT sector in hours (0–23)
#   mlt_end    (float): end of MLT sector in hours (0–23); may be < mlt_start
#                       for sectors that cross midnight (e.g. 22 → 2)
#
# Add, remove, or reorder entries freely — the loop in main() processes each
# one independently and saves a separate pickle + region plot per entry.
REGIONS = [
	{'mlat_low': 60.0, 'mlat_high': 75.0, 'mlt_start':  6, 'mlt_end': 17},   # dayside auroral
	{'mlat_low': 60.0, 'mlat_high': 75.0, 'mlt_start': 22, 'mlt_end':  2},   # nightside, wraps midnight
	{'mlat_low': 75.0, 'mlat_high': 90.0, 'mlt_start':  0, 'mlt_end': 23},   # polar cap, full circle
]


# ---------------------------------------------------------------------------
# Spatial index helpers
# ---------------------------------------------------------------------------

def mlat_to_indices(mlat_low: float, mlat_high: float) -> list:
	'''
	Convert a closed MLAT interval [mlat_low, mlat_high] (degrees magnetic
	latitude) to the corresponding bin indices along the colatitude axis.

	The colatitude axis runs pole-first:
	  index 0            → MLAT_MAX (pole, ~90° MLAT)
	  index N_MLAT - 1   → MLAT_MIN (equatorward edge, ~50° MLAT)

	Both endpoints are included (closed interval).

	Args:
		mlat_low  (float): equatorward boundary in degrees MLAT.
		mlat_high (float): poleward boundary in degrees MLAT.

	Returns:
		list[int]: bin indices covering [mlat_low, mlat_high], pole-first order.

	Example:
		mlat_to_indices(60, 75)  →  indices for the 60°–75° MLAT band
	'''
	if not (MLAT_MIN <= mlat_low <= mlat_high <= MLAT_MAX):
		raise ValueError(
			f'Expected {MLAT_MIN} ≤ mlat_low ≤ mlat_high ≤ {MLAT_MAX}, '
			f'got [{mlat_low}, {mlat_high}]'
		)
	bin_width = (MLAT_MAX - MLAT_MIN) / N_MLAT
	# High MLAT (poleward) maps to a lower index; low MLAT maps to a higher index
	idx_pole  = int((MLAT_MAX - mlat_high) / bin_width)
	idx_eq    = min(int((MLAT_MAX - mlat_low) / bin_width), N_MLAT - 1)
	indices   = list(range(idx_pole, idx_eq + 1))
	print(f'MLAT [{mlat_low}°, {mlat_high}°]  →  indices [{idx_pole}, {idx_eq}]  '
		  f'({len(indices)} bins)')
	return indices


def mlt_to_indices(mlt_start: float, mlt_end: float) -> list:
	'''
	Convert an MLT sector to bin indices, correctly handling sectors that
	cross midnight (i.e. mlt_start > mlt_end).

	MLT bins are numbered 0–23, where bin k covers [k, k+1) hours.
	Both endpoints are included at integer-hour bin resolution.

	Args:
		mlt_start (float): start of the sector in MLT hours (0–23).
		mlt_end   (float): end   of the sector in MLT hours (0–23).

	Returns:
		list[int]: bin indices in traversal order (wrapping through midnight
		           when mlt_start > mlt_end).

	Examples:
		mlt_to_indices(9,  15)  →  [9, 10, 11, 12, 13, 14, 15]   (dayside)
		mlt_to_indices(22,  2)  →  [22, 23, 0, 1, 2]              (nightside, wraps)
		mlt_to_indices(0,  23)  →  [0, 1, ..., 23]                (full circle)
	'''
	start = int(mlt_start) % N_MLT
	end   = int(mlt_end)   % N_MLT

	if start <= end:
		indices = list(range(start, end + 1))
	else:
		# Sector crosses midnight: tail of the 0–23 cycle, then the head
		indices = list(range(start, N_MLT)) + list(range(0, end + 1))

	print(f'MLT [{mlt_start}, {mlt_end}]  →  indices {indices}  ({len(indices)} bins)')
	return indices


def region_tag(mlat_low: float, mlat_high: float, mlt_start: float, mlt_end: float) -> str:
	'''
	Build a compact, filename-safe string describing a MLAT × MLT region.

	Args:
		mlat_low  (float): equatorward MLAT boundary in degrees.
		mlat_high (float): poleward    MLAT boundary in degrees.
		mlt_start (float): start of the MLT sector in hours.
		mlt_end   (float): end   of the MLT sector in hours.

	Returns:
		str: e.g. "mlat60-75_mlt22-02"
	'''
	return (
		f'mlat{int(mlat_low)}-{int(mlat_high)}'
		f'_mlt{int(mlt_start):02d}-{int(mlt_end):02d}'
	)


# ---------------------------------------------------------------------------
# Region visualisation
# ---------------------------------------------------------------------------

def plot_shap_region(mlat_indices: list, mlt_indices: list, save_path: str = None):
	'''
	Draws a blank polar (MLT × MLAT) map with a highlighted bounding box
	showing the spatial region selected for SHAP attribution.

	The plot matches the project's standard polar orientation:
	  - Midnight (00 MLT) at the bottom  [set_theta_zero_location('S')]
	  - Counter-clockwise MLT progression
	  - Colatitude on the radial axis, pole at the centre
	  - MLAT degree labels on the radial rings

	Midnight-crossing MLT sectors (e.g. mlt_indices=[22, 23, 0, 1, 2]) are
	detected automatically by looking for a gap larger than 1 in the sorted
	index list, and the arc is drawn correctly by offsetting the angular span
	from the true start rather than using raw modular arithmetic on the indices.

	Args:
		mlat_indices (list[int]): MLAT bin indices defining the region
		                          (as returned by mlat_to_indices).
		mlt_indices  (list[int]): MLT  bin indices defining the region
		                          (as returned by mlt_to_indices; may wrap midnight).
		save_path    (str | None): path to save the figure (PNG/PDF/etc).
		                           If None, plt.show() is called instead.
	'''
	bin_width_mlat = (MLAT_MAX - MLAT_MIN) / N_MLAT   # degrees per MLAT bin
	colat_max      = 90.0 - MLAT_MIN                   # outermost ring in colatitude

	# ---- Derive degree bounds from bin indices --------------------------------
	# MLAT axis: index 0 = pole (MLAT_MAX), so poleward edge of the region is
	# determined by the smallest (most-poleward) index.
	idx_pole = min(mlat_indices)
	idx_eq   = max(mlat_indices)
	mlat_high_deg = MLAT_MAX - idx_pole * bin_width_mlat          # poleward edge
	mlat_low_deg  = MLAT_MAX - (idx_eq + 1) * bin_width_mlat      # equatorward edge
	colat_inner   = 90.0 - mlat_high_deg   # poleward edge in colatitude (smaller r)
	colat_outer   = 90.0 - mlat_low_deg    # equatorward edge in colatitude (larger r)

	# ---- Derive angular bounds from MLT bin indices --------------------------
	# Detect midnight-crossing: a gap of more than 1 between consecutive sorted
	# indices means the sector wraps through index 0.
	sorted_mlt = sorted(mlt_indices)
	gaps       = [sorted_mlt[i + 1] - sorted_mlt[i] for i in range(len(sorted_mlt) - 1)]

	if gaps and max(gaps) > 1:
		# Midnight-crossing: the real start is the bin immediately after the gap.
		gap_pos      = gaps.index(max(gaps))
		mlt_start_hr = sorted_mlt[gap_pos + 1]          # first bin on the post-midnight side
		mlt_end_hr   = sorted_mlt[gap_pos] + 1          # trailing edge of last pre-midnight bin
	else:
		mlt_start_hr = sorted_mlt[0]
		mlt_end_hr   = sorted_mlt[-1] + 1

	# Angular span in hours — always positive (0, 24]
	span_hours = (mlt_end_hr - mlt_start_hr) % 24
	if span_hours == 0:
		span_hours = 24

	# ---- Build closed sector polygon -----------------------------------------
	# theta = MLT_hours * 2π/24.  Allowing theta > 2π is fine in matplotlib's
	# polar axes; no explicit modular wrapping needed.
	start_rad = mlt_start_hr * (2.0 * np.pi / 24.0)
	span_rad  = span_hours   * (2.0 * np.pi / 24.0)
	n_pts     = 300

	arc = np.linspace(start_rad, start_rad + span_rad, n_pts)

	# Polygon traces the inner arc (poleward), then the outer arc in reverse,
	# closing back at the start — giving a solid filled shape.
	theta_poly = np.concatenate([arc, arc[::-1]])
	r_poly     = np.concatenate([
		np.full(n_pts, colat_inner),
		np.full(n_pts, colat_outer),
	])

	# ---- Draw -----------------------------------------------------------------
	fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})

	ax.set_theta_zero_location('S')    # midnight (00 MLT) at bottom
	ax.set_theta_direction(1)          # counter-clockwise: dawn (06) left, dusk (18) right

	# Radial axis: colatitude 0 → colat_max, labelled in MLAT degrees
	ax.set_ylim(0, colat_max)
	colat_ticks = np.arange(0, colat_max + 1, 10)
	ax.set_yticks(colat_ticks)
	ax.set_yticklabels(
		[f'{int(90 - c)}°' for c in colat_ticks],
		fontsize=8, color='dimgrey',
	)

	# Angular axis: MLT labels every 3 hours
	mlt_tick_hrs = np.arange(0, 24, 3)
	ax.set_xticks(mlt_tick_hrs * 2.0 * np.pi / 24.0)
	ax.set_xticklabels([f'{h:02d}' for h in mlt_tick_hrs], fontsize=9)

	ax.grid(color='lightgrey', linestyle='--', linewidth=0.6, zorder=0)

	# Filled region + solid outline.
	# ax.fill closes automatically; ax.plot does not, so the first point is
	# appended explicitly to avoid a gap in the bounding box outline.
	theta_closed = np.append(theta_poly, theta_poly[0])
	r_closed     = np.append(r_poly,     r_poly[0])

	ax.fill(theta_poly,   r_poly,   color='crimson', alpha=0.20, zorder=2, label='SHAP region')
	ax.plot(theta_closed, r_closed, color='crimson', linewidth=2.0, zorder=3)

	# Title with human-readable region description
	mlt_end_label = int(mlt_end_hr) % 24
	ax.set_title(
		f'SHAP Attribution Region\n'
		f'MLAT {mlat_low_deg:.0f}°–{mlat_high_deg:.0f}°   '
		f'MLT {mlt_start_hr:02d}–{mlt_end_label:02d}',
		fontsize=11, pad=20,
	)

	plt.tight_layout()

	if save_path:
		plt.savefig(save_path, dpi=150, bbox_inches='tight')
		print(f'Region plot saved → {save_path}')
	else:
		plt.show()

	plt.close(fig)


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def _free_gpu():
	'''Flush GPU memory after freeing a tensor.'''
	gc.collect()
	if DEVICE.type == 'cuda':
		torch.cuda.empty_cache()


def _make_predict_fn(model, input_shape):
	'''
	Wraps a PyTorch model into a numpy-in / numpy-out callable for KernelExplainer.

	KernelExplainer communicates entirely in flat 2D numpy arrays (n_samples, n_features).
	This wrapper:
	  1. Reshapes the flat array back to (n_samples, *input_shape)
	  2. Moves the batch to GPU only for the forward pass
	  3. Pulls the result back to CPU as numpy immediately after

	At any moment only one perturbation batch occupies GPU memory.

	Args:
		model (torch.nn.Module): trained model (or RegionalMeanWrapper) in eval mode.
		input_shape (tuple): per-sample shape before flattening, e.g. (C, H, W).

	Returns:
		callable: prediction function suitable for shap.KernelExplainer.
	'''
	def predict_fn(x_flat):
		x = torch.tensor(x_flat, dtype=torch.float).reshape(-1, *input_shape).to(DEVICE)
		with torch.no_grad():
			out = model(x)
		result = out.cpu().numpy().reshape(x.shape[0], -1)
		del x, out
		_free_gpu()
		return result

	return predict_fn


# ---------------------------------------------------------------------------
# Regional mean wrapper
# ---------------------------------------------------------------------------

class RegionalMeanWrapper(torch.nn.Module):
	'''
	Wraps a model so that its spatial output is masked to a specific
	MLAT × MLT region before SHAP attribution.

	The underlying model outputs shape (N, 2, 50, 24):
	  - dim 1, channel 0: posterior mean FAC
	  - dim 1, channel 1: posterior std FAC
	  - dim 2: 50 MLAT bins (colatitude, pole at index 0)
	  - dim 3: 24 MLT bins  (index 0 = midnight, 0 MLT)

	This wrapper selects only the pixels at the intersection of
	``mlat_indices`` and ``mlt_indices``, using fancy (non-contiguous) indexing
	so that MLT sectors crossing midnight (e.g. [22, 23, 0, 1, 2]) are handled
	correctly without any special-casing.  The selected pixels are reduced to a
	per-channel mean, yielding shape (N, C).

	SHAP values therefore represent each input feature's contribution to the
	mean FAC (and/or mean std) within the chosen spatial region.

	Pass ``channels=(0,)`` to restrict attribution to the posterior mean only;
	this halves the SHAP output dimensionality and is usually sufficient.

	Args:
		model        (torch.nn.Module): trained ACORN model in eval mode.
		mlat_indices (array-like):      MLAT axis (dim 2) indices to include.
		mlt_indices  (array-like):      MLT  axis (dim 3) indices to include.
		channels     (tuple[int]):      output channels to keep. Default (0, 1).
	'''

	def __init__(self, model, mlat_indices, mlt_indices, channels=(0, 1)):
		super().__init__()
		self.model    = model
		self.channels = list(channels)
		# Register index tensors as buffers so .to(device) carries them
		# automatically — the mask operation then stays on the same device
		# as the model activations with no manual .to() calls needed.
		self.register_buffer('mlat_idx', torch.tensor(mlat_indices, dtype=torch.long))
		self.register_buffer('mlt_idx',  torch.tensor(mlt_indices,  dtype=torch.long))

	def forward(self, x):
		out = self.model(x)                    # (N, 2, 50, 24) or (N, 2, 50, 26)

		# The midnight-looping model variant outputs 26 MLT columns: one padding
		# column is prepended and one appended around the standard 24-column grid
		# to ensure continuity at midnight.  Truncate to the inner 24 before any
		# MLT indexing so that mlt_idx values 0–23 map to the correct hours.
		if out.shape[-1] == 26:
			out = out[:, :, :, 1:-1]           # (N, 2, 50, 24)

		out = out[:, self.channels, :, :]      # (N, C, 50, 24)  — select channels
		out = out[:, :, self.mlat_idx, :]      # (N, C, |mlat|, 24)
		out = out[:, :, :, self.mlt_idx]       # (N, C, |mlat|, |mlt|)
		return out.mean(dim=(-2, -1))          # (N, C)


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def get_shap_values(model, model_name, training_data, testing_data,
					background_examples=1000, delimiter=100, explainer_type='deep',
					mlat_indices=None, mlt_indices=None, channels=(0, 1),
					checkpoint_dir=None):
	'''
	Calculates SHAP values for the given model and test data.

	GPU memory strategy: everything is built and stored on CPU. Data is moved
	to GPU only at the moment it is consumed, then freed immediately. At peak,
	only the model weights + one batch (or the background for GradientExplainer
	initialisation) occupy GPU memory at any one time.

	Supports two explainer backends:

	  'deep'   -- shap.GradientExplainer. Fast, gradient-based, PyTorch-native.
	             Background is moved to GPU for init then freed before the
	             test loop begins. Each test batch is moved to GPU, explained,
	             then freed before the next batch.

	  'kernel' -- shap.KernelExplainer. Model-agnostic, perturbation-based.
	             Fully CPU/numpy-based. predict_fn moves each internal
	             perturbation batch to GPU for inference only.
	             Use smaller background_examples (e.g. 100) and delimiter (e.g. 5).

	When ``mlat_indices`` and/or ``mlt_indices`` are provided, the model output
	is first reduced to the mean FAC over the specified MLAT × MLT region via
	RegionalMeanWrapper before any SHAP attribution is computed.  MLT sectors
	that cross midnight (e.g. mlt_indices=[22, 23, 0, 1, 2]) are handled
	correctly by the wrapper's fancy indexing.

	Checkpointing: when ``checkpoint_dir`` is provided, each completed batch is
	saved immediately as ``batch_NNNNN.pkl`` inside that directory.  The
	background tensor is saved as ``background.pkl`` on the first run and
	reloaded on subsequent runs, ensuring all batches share the same explainer
	baseline even across interruptions.  Already-saved batches are skipped on
	resume.  Once the caller has consolidated results into the final output
	pickle, the checkpoint directory can be deleted safely.

	Args:
		model              (torch.nn.Module): trained neural network model.
		model_name         (str):             version string, used for labelling.
		training_data      (list[Tensor] | np.ndarray | Tensor): background samples.
		testing_data       (list[Tensor] | np.ndarray | Tensor): data to explain.
		background_examples (int):  number of background samples.  Default 1000.
		delimiter          (int):   batch size for SHAP forward passes. Default 100.
		explainer_type     (str):   'deep' or 'kernel'. Default 'deep'.
		mlat_indices       (list[int] | None): MLAT bin indices to include in the
		                    regional mean.  None uses the full 50-bin axis.
		                    Use mlat_to_indices() to convert from degree MLAT.
		mlt_indices        (list[int] | None): MLT bin indices to include in the
		                    regional mean.  None uses the full 24-bin axis.
		                    Use mlt_to_indices() to build sectors that cross midnight.
		channels           (tuple[int]): output channels passed to
		                    RegionalMeanWrapper.  Default (0, 1) keeps both
		                    posterior mean and std.  Pass (0,) for mean only.
		checkpoint_dir     (str | None): directory for per-batch checkpoint files.
		                    None disables checkpointing (original behaviour).

	Returns:
		list: SHAP value batches across the test set.
		str:  placeholder for expected_value (not yet wired up).
	'''

	if explainer_type not in ('deep', 'kernel'):
		raise ValueError(f"explainer_type must be 'deep' or 'kernel', got '{explainer_type}'.")

	if checkpoint_dir is not None:
		os.makedirs(checkpoint_dir, exist_ok=True)

	# ------------------------------------------------------------------
	# 0. Optionally restrict SHAP attribution to a spatial sub-region.
	#    RegionalMeanWrapper reduces model output from (N, 2, 50, 24) to
	#    (N, C) — the mean FAC over the requested MLAT × MLT patch.
	#    Non-contiguous MLT index lists (midnight-crossing sectors) are
	#    handled correctly by the wrapper's buffer-based fancy indexing.
	# ------------------------------------------------------------------
	if mlat_indices is not None or mlt_indices is not None:
		_mlat = mlat_indices if mlat_indices is not None else list(range(N_MLAT))
		_mlt  = mlt_indices  if mlt_indices  is not None else list(range(N_MLT))
		print(
			f'Restricting SHAP to MLAT indices {_mlat} × MLT indices {_mlt}  '
			f'({len(_mlat) * len(_mlt)} pixels, channels {list(channels)})'
		)
		effective_model = RegionalMeanWrapper(model, _mlat, _mlt, channels).to(DEVICE)
	else:
		print('No spatial mask applied — SHAP computed over full 50×24 output.')
		effective_model = model

	effective_model.eval()

	# ------------------------------------------------------------------
	# 1. Build background and testing tensors on CPU only.
	#
	#    Background: if a checkpoint exists from a previous interrupted run,
	#    reload the exact same background tensor so all batches share the
	#    same explainer baseline.  Otherwise sample fresh and save it.
	#
	#    Testing data is always rebuilt from scratch; individual batch
	#    results that were already saved will simply be skipped below.
	# ------------------------------------------------------------------
	if isinstance(training_data, list):
		print(f'Training data is a list of {len(training_data)} tensors, shape: {training_data[0].shape}')
		input_shape = training_data[0].shape

		bg_ckpt = os.path.join(checkpoint_dir, 'background.pkl') if checkpoint_dir else None
		if bg_ckpt and os.path.exists(bg_ckpt):
			print(f'Reloading background from checkpoint: {bg_ckpt}')
			with open(bg_ckpt, 'rb') as f:
				background_cpu = pickle.load(f)
		else:
			random_indices = np.random.choice(len(training_data), background_examples, replace=False)
			background_cpu = torch.stack(
				[training_data[i] for i in random_indices], dim=0
			).to('cpu', dtype=torch.float)
			if bg_ckpt:
				with open(bg_ckpt, 'wb') as f:
					pickle.dump(background_cpu, f)
				print(f'Background saved to checkpoint: {bg_ckpt}')

		testing_cpu = torch.stack(testing_data, dim=0).to('cpu', dtype=torch.float)

	elif isinstance(training_data, (np.ndarray, torch.Tensor)):
		print('Training data is a numpy array / tensor....')
		input_shape = training_data[0].shape

		bg_ckpt = os.path.join(checkpoint_dir, 'background.pkl') if checkpoint_dir else None
		if bg_ckpt and os.path.exists(bg_ckpt):
			print(f'Reloading background from checkpoint: {bg_ckpt}')
			with open(bg_ckpt, 'rb') as f:
				background_cpu = pickle.load(f)
		else:
			random_indices = np.random.choice(len(training_data), background_examples, replace=False)
			if isinstance(training_data, np.ndarray):
				background_cpu = torch.tensor(training_data[random_indices], dtype=torch.float)
			else:
				background_cpu = training_data[random_indices].to('cpu', dtype=torch.float)
			if bg_ckpt:
				with open(bg_ckpt, 'wb') as f:
					pickle.dump(background_cpu, f)
				print(f'Background saved to checkpoint: {bg_ckpt}')

		if isinstance(testing_data, np.ndarray):
			testing_cpu = torch.tensor(testing_data, dtype=torch.float)
		else:
			testing_cpu = testing_data.to('cpu', dtype=torch.float)

	else:
		raise ValueError('training_data must be a list of Tensors, a numpy array, or a Tensor.')

	print(f'Background shape: {background_cpu.shape}')
	print(f'Testing data shape: {testing_cpu.shape}')

	del training_data
	gc.collect()

	n_samples  = testing_cpu.shape[0]
	n_batches  = (n_samples + delimiter - 1) // delimiter

	# ------------------------------------------------------------------
	# 2. Build explainer.
	#    eval() is required regardless of explainer type — disables
	#    dropout and fixes batchnorm statistics for consistent explanations.
	# ------------------------------------------------------------------
	if explainer_type == 'deep':
		background_gpu = background_cpu.to(DEVICE)
		del background_cpu
		gc.collect()

		explainer = shap.GradientExplainer(model=effective_model, data=background_gpu)

		del background_gpu
		_free_gpu()

		# ------------------------------------------------------------------
		# 3a. Calculate SHAP values — GradientExplainer
		#
		#     Each batch checks for an existing checkpoint file first.
		#     Completed batches are loaded from disk and skipped; new batches
		#     are computed, saved, then freed before the next batch.
		# ------------------------------------------------------------------
		print('Calculating SHAP values (GradientExplainer)....')
		shap_values = []
		n_skipped   = 0
		for batch_idx, batch_start in enumerate(
			tqdm.tqdm(range(0, n_samples, delimiter), total=n_batches, desc='shap batches')
		):
			batch_ckpt = (
				os.path.join(checkpoint_dir, f'batch_{batch_idx:05d}.pkl')
				if checkpoint_dir else None
			)

			if batch_ckpt and os.path.exists(batch_ckpt):
				with open(batch_ckpt, 'rb') as f:
					shap_values.append(pickle.load(f))
				n_skipped += 1
				continue

			batch_end = min(batch_start + delimiter, n_samples)
			batch_gpu = testing_cpu[batch_start:batch_end].to(DEVICE)
			result    = explainer.shap_values(batch_gpu)
			shap_values.append(result)

			if batch_ckpt:
				with open(batch_ckpt, 'wb') as f:
					pickle.dump(result, f)

			del batch_gpu
			_free_gpu()

		if n_skipped:
			print(f'  Resumed: {n_skipped}/{n_batches} batches loaded from checkpoint, '
				  f'{n_batches - n_skipped} computed fresh.')

	else:
		# ------------------------------------------------------------------
		# 3b. Calculate SHAP values — KernelExplainer
		#
		# KernelExplainer is fully CPU/numpy-based. Background and test data
		# are flattened to (n_samples, n_features) numpy arrays here.
		# predict_fn (see _make_predict_fn) moves each internal perturbation
		# batch to GPU only for the forward pass, then pulls back to CPU.
		# Peak GPU = model + one KernelExplainer perturbation batch.
		#
		# Note: KernelExplainer.shap_values() does not accept check_additivity.
		# It is significantly slower than GradientExplainer — keep delimiter small.
		# ------------------------------------------------------------------
		predict_fn = _make_predict_fn(effective_model, input_shape)

		background_np = background_cpu.numpy().reshape(background_cpu.shape[0], -1)
		del background_cpu
		gc.collect()

		explainer = shap.KernelExplainer(predict_fn, background_np)
		del background_np
		gc.collect()

		testing_np = testing_cpu.numpy().reshape(n_samples, -1)
		del testing_cpu
		gc.collect()

		print('Calculating SHAP values (KernelExplainer)....')
		shap_values = []
		n_skipped   = 0
		for batch_idx, batch_start in enumerate(
			tqdm.tqdm(range(0, n_samples, delimiter), total=n_batches, desc='shap batches')
		):
			batch_ckpt = (
				os.path.join(checkpoint_dir, f'batch_{batch_idx:05d}.pkl')
				if checkpoint_dir else None
			)

			if batch_ckpt and os.path.exists(batch_ckpt):
				with open(batch_ckpt, 'rb') as f:
					shap_values.append(pickle.load(f))
				n_skipped += 1
				continue

			batch_end = min(batch_start + delimiter, n_samples)
			result    = explainer.shap_values(testing_np[batch_start:batch_end])
			shap_values.append(result)

			if batch_ckpt:
				with open(batch_ckpt, 'wb') as f:
					pickle.dump(result, f)

		if n_skipped:
			print(f'  Resumed: {n_skipped}/{n_batches} batches loaded from checkpoint, '
				  f'{n_batches - n_skipped} computed fresh.')

	return shap_values, '____'  # explainer.expected_value not yet wired up


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def converting_shap_to_percentages(shap_values, features):
	'''
	Converts raw SHAP values to percentage contributions per feature.

	Args:
		shap_values (list[np.ndarray] | np.ndarray): SHAP values to convert.
		features (list[str]): feature names for DataFrame columns.

	Returns:
		list[pd.DataFrame] | pd.DataFrame: percentage contributions.
	'''

	def _to_percentage(arr):
		# Sum across spatial/time axis, reshape to (samples, features), then normalise
		summed = np.sum(arr, axis=1).reshape(arr.shape[0], -1)
		df = pd.DataFrame(summed, columns=features)
		return df.div(df.abs().sum(axis=1), axis=0) * 100

	if len(shap_values) > 1:
		return [_to_percentage(sv) for sv in shap_values]
	else:
		return _to_percentage(shap_values[0])


# ---------------------------------------------------------------------------
# Checkpoint diagnostic
# ---------------------------------------------------------------------------

def compare_state_dicts(checkpoint_path, model):
	'''Print all shape mismatches between a checkpoint and a live model.'''
	ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

	# Unwrap if nested
	if isinstance(ckpt, dict) and 'model' in ckpt:
		ckpt = ckpt['model']

	model_sd = model.state_dict()

	ckpt_keys   = set(ckpt.keys())
	model_keys  = set(model_sd.keys())

	missing    = model_keys - ckpt_keys
	unexpected = ckpt_keys  - model_keys
	mismatched = {
		k for k in ckpt_keys & model_keys
		if ckpt[k].shape != model_sd[k].shape
	}

	print(f'  Missing in checkpoint   : {len(missing)}')
	print(f'  Unexpected in checkpoint: {len(unexpected)}')
	print(f'  Shape mismatches        : {len(mismatched)}')

	if mismatched:
		print(f"\n  {'Key':<55} {'Checkpoint':>20} {'Model':>20}")
		print(f"  {'─'*55} {'─'*20} {'─'*20}")
		for k in sorted(mismatched):
			print(f"  {k:<55} {str(tuple(ckpt[k].shape)):>20} {str(tuple(model_sd[k].shape)):>20}")

	if missing:
		print(f"\n  Missing keys:\n  " + "\n  ".join(sorted(missing)))
	if unexpected:
		print(f"\n  Unexpected keys:\n  " + "\n  ".join(sorted(unexpected)))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
	'''
	Loads data and the model once, then iterates over every entry in REGIONS,
	running SHAP attribution and saving a pickle + region plot per slice.
	'''
	os.makedirs(working_dir + '/outputs', exist_ok=True)
	os.makedirs(working_dir + '/models',  exist_ok=True)
	os.makedirs(working_dir + '/shap',    exist_ok=True)

	# ------------------------------------------------------------------
	# 1. Load data — done once; all region loops reuse the same tensors.
	# ------------------------------------------------------------------
	print('Loading data...')
	PD = PreparingData()
	train_dict, val_dict, test_dict = PD()

	dates = [date for date in test_dict.keys()]

	# Build tensor lists on CPU — unsqueeze(0) adds the channel dim required by the conv layers
	train_x = [torch.tensor(train_dict[key]['input']).unsqueeze(0) for key in train_dict.keys()]
	test_x  = [torch.tensor(test_dict[key]['input']).unsqueeze(0)  for key in test_dict.keys()]

	del train_dict, val_dict, test_dict
	gc.collect()

	# ------------------------------------------------------------------
	# 2. Build and load model — done once; RegionalMeanWrapper is a thin
	#    stateless shim so the base model is never modified between regions.
	# ------------------------------------------------------------------
	print('Creating model....')
	torch.manual_seed(CONFIG['random_seed'])
	torch.cuda.manual_seed(CONFIG['random_seed'])

	CONFIG['model_config']['output_size'] = (50, 26)

	model = ACORN(**CONFIG['model_config'])
	compare_state_dicts(model_file, model)
	print(model)

	model.to(DEVICE)
	checkpoint = torch.load(model_file, map_location=DEVICE)
	model.load_state_dict(checkpoint['model'])

	# Set explainer_type to 'deep' or 'kernel'
	# 'deep'   -- faster, gradient-based, recommended for differentiable models
	# 'kernel' -- slower, model-agnostic; use smaller background_examples + delimiter
	EXPLAINER_TYPE = 'deep'

	# ------------------------------------------------------------------
	# 3. Region loop — each iteration is fully independent.
	#    Data and model are never reloaded; only the wrapper and explainer
	#    are rebuilt per region, which is cheap.
	# ------------------------------------------------------------------
	n_regions = len(REGIONS)
	for i, region in enumerate(REGIONS):
		mlat_low  = region['mlat_low']
		mlat_high = region['mlat_high']
		mlt_start = region['mlt_start']
		mlt_end   = region['mlt_end']

		tag = region_tag(mlat_low, mlat_high, mlt_start, mlt_end)
		print(f'\n{"="*60}')
		print(f'Region {i + 1}/{n_regions}: {tag}')
		print(f'{"="*60}')

		mlat_indices = mlat_to_indices(mlat_low, mlat_high)
		mlt_indices  = mlt_to_indices(mlt_start, mlt_end)

		# Sanity-check plot saved before the (potentially long) SHAP run
		plot_shap_region(
			mlat_indices=mlat_indices,
			mlt_indices=mlt_indices,
			save_path=working_dir + f'/shap/{CONFIG["version"]}_{tag}_region.png',
		)

		print(f'Getting SHAP values using {EXPLAINER_TYPE} explainer....')
		ckpt_dir = working_dir + f'/shap/checkpoints/{CONFIG["version"]}_{EXPLAINER_TYPE}_{tag}'
		shap_values, expectation_values = get_shap_values(
			model=model,
			model_name=CONFIG['version'],
			training_data=train_x,
			testing_data=test_x,
			background_examples=1000,
			delimiter=1,
			explainer_type=EXPLAINER_TYPE,
			mlat_indices=mlat_indices,
			mlt_indices=mlt_indices,
			channels=(0,),   # posterior mean only; use (0, 1) to include std channel
			checkpoint_dir=ckpt_dir,
		)

		out_path = working_dir + f'/shap/{CONFIG["version"]}_{EXPLAINER_TYPE}_{tag}.pkl'
		print(f'Saving results → {out_path}')
		with open(out_path, 'wb') as f:
			pickle.dump(
				{
					'shap_values':        shap_values,
					'expectation_values': expectation_values,
					'dates':              dates,
					'mlat_low':           mlat_low,
					'mlat_high':          mlat_high,
					'mlt_start':          mlt_start,
					'mlt_end':            mlt_end,
					'mlat_indices':       mlat_indices,
					'mlt_indices':        mlt_indices,
				},
				f,
			)

		# Remove checkpoint directory now that results are safely consolidated.
		# Comment this block out if you want to keep the per-batch files.
		if os.path.isdir(ckpt_dir):
			shutil.rmtree(ckpt_dir)
			print(f'Checkpoint directory removed: {ckpt_dir}')

		gc.collect()

	print('\nAll regions complete.')


if __name__ == '__main__':

	main()

	print('It ran. Good job!')
