"""
spatial_correlation.py
======================
Computes per-pixel spatial correlation maps for ACORN FAC model outputs and
AMPERE ground-truth data over a complete test set.

For each of the 1200 pixels in a 50×24 (colatitude × MLT) FAC grid, this
module produces a 50×24 correlation map showing how strongly every other
pixel co-varies with that reference pixel across all test-set samples.  The
computation is carried out for three quantities extracted from the results
pickle:

    • "predicted"[0]  – ACORN posterior mean  (50×24 float, index 0 of the
                         (2, 50, 24) array stored under the "predicted" key)
    • "predicted"[1]  – ACORN posterior std   (50×24 float, index 1 of the
                         same array)
    • "ampere"        – AMPERE ground truth   (50×24 float)

The "predicted" value in each per-timestamp sub-dictionary must therefore
have shape (2, 50, 24).  The two planes are split automatically; no separate
"std" key is expected or used.

The full per-pixel correlation tensor has shape (1200, 50, 24), which is
equivalent to the (1200×1200) Pearson correlation matrix reshaped for spatial
visualisation.

Output files (written to --output-dir, default: current directory)
-------------------------------------------------------------------
    spatial_corr_predicted_mean.npy – float32 array (1200, 50, 24)
    spatial_corr_predicted_std.npy  – float32 array (1200, 50, 24)
    spatial_corr_ampere.npy         – float32 array (1200, 50, 24)
    spatial_corr_residual.npy       – float32 array (1200, 50, 24)
    corr_matrix_predicted_mean.npy  – float32 array (1200, 1200)   [optional]
    corr_matrix_predicted_std.npy   – float32 array (1200, 1200)   [optional]
    corr_matrix_ampere.npy          – float32 array (1200, 1200)   [optional]
    corr_matrix_residual.npy        – float32 array (1200, 1200)   [optional]
    spatial_correlation_meta.json   – run metadata (timestamps, shapes, …)

Usage
-----
    python spatial_correlation.py results.pkl
    python spatial_correlation.py results.pkl --output-dir ./corr_results
    python spatial_correlation.py results.pkl --save-matrices --output-dir ./corr_results

Dependencies
------------
    numpy, pickle (stdlib), json (stdlib), argparse (stdlib), tqdm (optional)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional pretty progress bar – gracefully degrade if tqdm is not installed
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm as _tqdm

    def progress(iterable, **kwargs):
        return _tqdm(iterable, **kwargs)

except ImportError:
    def progress(iterable, desc="", **kwargs):  # type: ignore[misc]
        if desc:
            print(f"{desc} …")
        return iterable


# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------

GRID_ROWS: int = 50   # colatitude bins
GRID_COLS: int = 24   # MLT bins
N_PIXELS: int = GRID_ROWS * GRID_COLS  # 1200


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(pickle_path: str | Path) -> Dict:
    """Load the test-set results dictionary from a pickle file.

    The pickle is expected to contain a flat dictionary keyed by timestamp
    (any hashable type – datetime objects, strings, or integers are all
    accepted).  Each value must itself be a dictionary with at least the
    following keys:

        "predicted"  : numpy array of shape (2, 50, 24)
                       [0, :, :] → ACORN posterior mean
                       [1, :, :] → ACORN posterior std
        "ampere"     : numpy array of shape (50, 24) – AMPERE ground truth

    Missing or malformed entries for a given timestamp are skipped with a
    warning.

    Parameters
    ----------
    pickle_path : str or Path
        Path to the results pickle file.

    Returns
    -------
    dict
        The raw results dictionary as stored in the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the loaded object is not a dictionary.
    """
    pickle_path = Path(pickle_path)
    if not pickle_path.exists():
        raise FileNotFoundError(f"Results file not found: {pickle_path}")

    with open(pickle_path, "rb") as fh:
        data = pickle.load(fh)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a dict at the top level of the pickle, "
            f"got {type(data).__name__!r}."
        )

    return data


def extract_predicted_arrays(
    results: Dict,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Extract and stack ACORN posterior mean and std from the ``"predicted"`` key.

    Each per-timestamp entry is expected to store a ``(2, 50, 24)`` array
    under the key ``"predicted"``, where:

        ``entry["predicted"][0]``  → posterior mean  (50×24)
        ``entry["predicted"][1]``  → posterior std   (50×24)

    Parameters
    ----------
    results : dict
        Top-level results dictionary (timestamp → sub-dict).

    Returns
    -------
    means : np.ndarray, shape (N, 50, 24)
        Float64 array of posterior mean maps for all valid timestamps.
    stds : np.ndarray, shape (N, 50, 24)
        Float64 array of posterior std maps for all valid timestamps.
    valid_timestamps : list
        Ordered list of timestamps that were successfully extracted.

    Warns
    -----
    UserWarning
        For timestamps where ``"predicted"`` is missing or has an unexpected
        shape.  Those timestamps are skipped for both outputs.
    """
    expected_shape = (2, GRID_ROWS, GRID_COLS)
    mean_arrays = []
    std_arrays  = []
    valid_timestamps = []

    for ts, entry in results.items():
        if not isinstance(entry, dict):
            warnings.warn(
                f"Timestamp {ts!r}: expected a dict, got "
                f"{type(entry).__name__!r} – skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue

        if "predicted" not in entry:
            warnings.warn(
                f"Timestamp {ts!r}: key 'predicted' not found – skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue

        arr = np.asarray(entry["predicted"], dtype=np.float64)

        if arr.shape != expected_shape:
            warnings.warn(
                f"Timestamp {ts!r}, key 'predicted': expected shape "
                f"{expected_shape}, got {arr.shape} – skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue

        mean_arrays.append(arr[0])  # (50, 24) – posterior mean
        std_arrays.append(arr[1])   # (50, 24) – posterior std
        valid_timestamps.append(ts)

    if not mean_arrays:
        raise ValueError(
            "No valid 'predicted' arrays found.  "
            "Check that each timestamp entry contains a (2, 50, 24) array "
            "under the 'predicted' key."
        )

    means = np.stack(mean_arrays, axis=0)  # (N, 50, 24)
    stds  = np.stack(std_arrays,  axis=0)  # (N, 50, 24)
    return means, stds, valid_timestamps


def extract_ampere_arrays(
    results: Dict,
) -> Tuple[np.ndarray, list]:
    """Extract and stack AMPERE ground-truth maps from the ``"ampere"`` key.

    Each per-timestamp entry is expected to store a ``(50, 24)`` array under
    the key ``"ampere"``.

    Parameters
    ----------
    results : dict
        Top-level results dictionary (timestamp → sub-dict).

    Returns
    -------
    stacked : np.ndarray, shape (N, 50, 24)
        Float64 array of AMPERE maps for all valid timestamps.
    valid_timestamps : list
        Ordered list of timestamps that were successfully extracted.

    Warns
    -----
    UserWarning
        For timestamps where ``"ampere"`` is missing or has an unexpected
        shape.  Those timestamps are skipped.
    """
    expected_shape = (GRID_ROWS, GRID_COLS)
    arrays = []
    valid_timestamps = []

    for ts, entry in results.items():
        if not isinstance(entry, dict):
            warnings.warn(
                f"Timestamp {ts!r}: expected a dict, got "
                f"{type(entry).__name__!r} – skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue

        if "ampere" not in entry:
            warnings.warn(
                f"Timestamp {ts!r}: key 'ampere' not found – skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue

        arr = np.asarray(entry["ampere"], dtype=np.float64)

        if arr.shape != expected_shape:
            warnings.warn(
                f"Timestamp {ts!r}, key 'ampere': expected shape "
                f"{expected_shape}, got {arr.shape} – skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue

        arrays.append(arr)
        valid_timestamps.append(ts)

    if not arrays:
        raise ValueError(
            "No valid 'ampere' arrays found.  "
            "Check that each timestamp entry contains a (50, 24) array "
            "under the 'ampere' key."
        )

    stacked = np.stack(arrays, axis=0)  # (N, 50, 24)
    return stacked, valid_timestamps


# ---------------------------------------------------------------------------
# Core correlation computation
# ---------------------------------------------------------------------------

def compute_correlation_matrix(
    maps: np.ndarray,
    nan_policy: str = "propagate",
) -> np.ndarray:
    """Compute the full (N_pixels × N_pixels) Pearson correlation matrix.

    For each pair of pixels (i, j) the Pearson correlation is computed across
    all *N* test-set samples.

    The computation uses a vectorised matrix multiply:

        C = (X_centered.T  @  X_centered) / N
        corr[i, j] = C[i, j] / (std[i] * std[j])

    where ``X_centered`` has shape ``(N, 1200)`` and each column is the
    zero-meaned time series for one pixel.

    Parameters
    ----------
    maps : np.ndarray, shape (N, 50, 24)
        Test-set maps for a single quantity.
    nan_policy : {"propagate", "omit"}
        How to handle NaN values in the input.

        * ``"propagate"`` (default): any pixel whose time series contains a
          NaN will have NaN correlation with all other pixels.
        * ``"omit"``: NaN values are replaced with the per-pixel mean before
          computing, effectively ignoring those samples for that pixel.

    Returns
    -------
    corr_matrix : np.ndarray, shape (1200, 1200), dtype float64
        Pearson correlation matrix.  ``corr_matrix[i, j]`` is the correlation
        between the flattened pixel index *i* and pixel index *j*.
        Values lie in [-1, 1]; NaN indicates a pixel with zero variance
        (constant across all samples, e.g. always-zero boundary rows).

    Notes
    -----
    Memory: a float64 1200×1200 array occupies ~11 MB.
    """
    N, H, W = maps.shape
    flat = maps.reshape(N, H * W)  # (N, 1200)

    if nan_policy == "omit":
        # Replace each pixel's NaNs with that pixel's nanmean
        pixel_nanmean = np.nanmean(flat, axis=0, keepdims=True)
        nan_mask = np.isnan(flat)
        flat = np.where(nan_mask, pixel_nanmean, flat)

    # Per-pixel mean and centred data
    pixel_mean = flat.mean(axis=0)           # (1200,)
    flat_c = flat - pixel_mean               # (N, 1200)

    # Per-pixel standard deviation
    pixel_std = flat_c.std(axis=0)           # (1200,)

    # Guard against zero-variance pixels
    zero_var = pixel_std == 0
    if zero_var.any():
        warnings.warn(
            f"{zero_var.sum()} pixel(s) have zero variance and will produce "
            "NaN correlations.",
            UserWarning,
            stacklevel=2,
        )
    pixel_std_safe = np.where(zero_var, np.nan, pixel_std)

    # Unnormalised covariance matrix  (1200, 1200)
    cov = (flat_c.T @ flat_c) / N

    # Normalise
    denom = np.outer(pixel_std_safe, pixel_std_safe)   # (1200, 1200)
    with np.errstate(invalid="ignore"):
        corr_matrix = cov / denom

    return corr_matrix


def corr_matrix_to_maps(
    corr_matrix: np.ndarray,
    grid_shape: Tuple[int, int] = (GRID_ROWS, GRID_COLS),
) -> np.ndarray:
    """Reshape a flat correlation matrix into per-pixel spatial maps.

    Parameters
    ----------
    corr_matrix : np.ndarray, shape (1200, 1200)
        Pearson correlation matrix as returned by
        :func:`compute_correlation_matrix`.
    grid_shape : tuple of int, optional
        Spatial grid dimensions ``(rows, cols)``.  Defaults to ``(50, 24)``.

    Returns
    -------
    corr_maps : np.ndarray, shape (1200, 50, 24), dtype float32
        ``corr_maps[k]`` is the 50×24 map showing the correlation of every
        spatial pixel with pixel *k* (in row-major flattened order).
        Stored as float32 to halve memory and file size versus float64.
    """
    n_pixels = corr_matrix.shape[0]
    H, W = grid_shape
    corr_maps = corr_matrix.reshape(n_pixels, H, W).astype(np.float32)
    return corr_maps


# ---------------------------------------------------------------------------
# Saving utilities
# ---------------------------------------------------------------------------

def save_arrays(
    output_dir: Path,
    corr_maps: np.ndarray,
    corr_matrix: np.ndarray,
    quantity: str,
    save_matrix: bool = False,
) -> Dict[str, str]:
    """Save correlation maps (and optionally the full matrix) to disk.

    Parameters
    ----------
    output_dir : Path
        Directory in which to write output files.
    corr_maps : np.ndarray, shape (1200, 50, 24)
        Per-pixel correlation maps.
    corr_matrix : np.ndarray, shape (1200, 1200)
        Full correlation matrix.
    quantity : str
        Short identifier used in filenames (e.g. ``"predicted"``, ``"std"``,
        ``"ampere"``).
    save_matrix : bool, optional
        If True, also save the full (1200, 1200) matrix.  Defaults to False.

    Returns
    -------
    paths : dict
        Mapping from logical name to saved file path (as strings).
    """
    paths = {}

    maps_path = output_dir / f"spatial_corr_{quantity}.npy"
    np.save(maps_path, corr_maps)
    paths["corr_maps"] = str(maps_path)
    print(f"  Saved corr maps   → {maps_path}")

    if save_matrix:
        mat_path = output_dir / f"corr_matrix_{quantity}.npy"
        np.save(mat_path, corr_matrix.astype(np.float32))
        paths["corr_matrix"] = str(mat_path)
        print(f"  Saved corr matrix → {mat_path}")

    return paths


def save_metadata(output_dir: Path, meta: Dict) -> Path:
    """Persist run metadata to a JSON file for reproducibility.

    Parameters
    ----------
    output_dir : Path
        Directory in which to write the metadata file.
    meta : dict
        Arbitrary JSON-serialisable metadata.

    Returns
    -------
    Path
        Path to the written JSON file.
    """
    json_path = output_dir / "spatial_correlation_meta.json"
    with open(json_path, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"  Saved metadata    → {json_path}")
    return json_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    pickle_path: str | Path,
    output_dir: str | Path = ".",
    save_matrices: bool = False,
    nan_policy: str = "propagate",
) -> Dict:
    """End-to-end spatial correlation pipeline.

    Loads the test-set results pickle, extracts ACORN posterior mean and std
    from the ``"predicted"`` key (shape ``(2, 50, 24)``) and AMPERE ground
    truth from the ``"ampere"`` key (shape ``(50, 24)``), computes the full
    Pearson spatial correlation matrix for each quantity — including the
    per-pixel residual (predicted mean − AMPERE) for timestamps common to
    both — and saves per-pixel correlation maps as ``.npy`` files.

    Parameters
    ----------
    pickle_path : str or Path
        Path to the results pickle file.
    output_dir : str or Path, optional
        Directory in which all outputs are written.  Created if it does not
        exist.  Defaults to the current working directory.
    save_matrices : bool, optional
        If True, also save the full (1200, 1200) correlation matrix for each
        quantity (as float32 .npy files, ~6 MB each).  Defaults to False.
    nan_policy : {"propagate", "omit"}, optional
        NaN handling strategy passed to :func:`compute_correlation_matrix`.

    Returns
    -------
    results_meta : dict
        Summary metadata that is also written to
        ``spatial_correlation_meta.json``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  AIMFAHR — Spatial Correlation Analysis")
    print(f"{'='*60}")
    print(f"  Input  : {pickle_path}")
    print(f"  Output : {output_dir.resolve()}")
    print(f"  NaN policy      : {nan_policy}")
    print(f"  Save matrices   : {save_matrices}")
    print()

    # ── 1. Load ─────────────────────────────────────────────────────────────
    print("[ 1 / 3 ]  Loading results …")
    results = load_results(pickle_path)
    print(f"  Found {len(results)} timestamps in pickle.")

    # ── 2. Compute & save ───────────────────────────────────────────────────
    meta_quantities = {}
    saved_paths_all = {}

    # -- 2a. ACORN predicted mean & std (both extracted from "predicted") ----
    print("\n[ 2.1 / 3 ]  Extracting 'predicted' arrays (mean + std) …")
    pred_means, pred_stds, pred_ts = extract_predicted_arrays(results)
    print(f"  Valid samples : {pred_means.shape[0]}")

    for label, maps in (("predicted_mean", pred_means), ("predicted_std", pred_stds)):
        print(f"\n  → Computing correlation for {label!r} …", end="", flush=True)
        corr_mat  = compute_correlation_matrix(maps, nan_policy=nan_policy)
        print(" done.")
        corr_maps = corr_matrix_to_maps(corr_mat)

        print("    Saving …")
        saved_paths = save_arrays(
            output_dir, corr_maps, corr_mat, label, save_matrix=save_matrices
        )
        saved_paths_all[label] = saved_paths

        meta_quantities[label] = {
            "source_key": "predicted",
            "source_index": 0 if label == "predicted_mean" else 1,
            "n_samples": int(pred_means.shape[0]),
            "n_zero_variance_pixels": int(np.isnan(corr_maps).any(axis=(1, 2)).sum()),
            "corr_range": [
                float(np.nanmin(corr_maps)),
                float(np.nanmax(corr_maps)),
            ],
            "saved_files": saved_paths,
        }

    # -- 2b. AMPERE ──────────────────────────────────────────────────────────
    print("\n[ 2.2 / 3 ]  Processing 'ampere' …")
    ampere_maps, ampere_ts = extract_ampere_arrays(results)
    print(f"  Valid samples : {ampere_maps.shape[0]}")

    print("  Computing correlation …", end="", flush=True)
    corr_mat  = compute_correlation_matrix(ampere_maps, nan_policy=nan_policy)
    print(" done.")
    corr_maps = corr_matrix_to_maps(corr_mat)

    print("  Saving …")
    saved_paths = save_arrays(
        output_dir, corr_maps, corr_mat, "ampere", save_matrix=save_matrices
    )
    saved_paths_all["ampere"] = saved_paths

    meta_quantities["ampere"] = {
        "source_key": "ampere",
        "source_index": None,
        "n_samples": int(ampere_maps.shape[0]),
        "n_zero_variance_pixels": int(np.isnan(corr_maps).any(axis=(1, 2)).sum()),
        "corr_range": [
            float(np.nanmin(corr_maps)),
            float(np.nanmax(corr_maps)),
        ],
        "saved_files": saved_paths,
    }

    # -- 2c. Residuals (predicted mean − AMPERE) ─────────────────────────────
    # Only use timestamps present in both pred_ts and ampere_ts.
    print("\n[ 2.3 / 3 ]  Computing residuals (predicted mean − AMPERE) …")
    pred_ts_set   = {ts: i for i, ts in enumerate(pred_ts)}
    ampere_ts_set = {ts: i for i, ts in enumerate(ampere_ts)}
    common_ts     = sorted(set(pred_ts_set) & set(ampere_ts_set), key=str)

    if not common_ts:
        warnings.warn(
            "No common timestamps between predicted and AMPERE arrays — "
            "residual correlation map will not be computed.",
            UserWarning,
        )
    else:
        residual_maps = np.stack(
            [
                pred_means[pred_ts_set[ts]] - ampere_maps[ampere_ts_set[ts]]
                for ts in common_ts
            ],
            axis=0,
        )  # (N_common, 50, 24)
        skipped = len(pred_ts) - len(common_ts)
        if skipped:
            warnings.warn(
                f"{skipped} predicted timestamp(s) had no matching AMPERE entry "
                "and were excluded from the residual.",
                UserWarning,
            )
        print(f"  Common samples : {len(common_ts)}")

        print("  Computing correlation …", end="", flush=True)
        corr_mat  = compute_correlation_matrix(residual_maps, nan_policy=nan_policy)
        print(" done.")
        corr_maps = corr_matrix_to_maps(corr_mat)

        print("  Saving …")
        saved_paths = save_arrays(
            output_dir, corr_maps, corr_mat, "residual", save_matrix=save_matrices
        )
        saved_paths_all["residual"] = saved_paths

        meta_quantities["residual"] = {
            "source_key": "predicted[0] - ampere",
            "source_index": None,
            "n_samples": len(common_ts),
            "n_skipped_timestamps": skipped,
            "n_zero_variance_pixels": int(np.isnan(corr_maps).any(axis=(1, 2)).sum()),
            "corr_range": [
                float(np.nanmin(corr_maps)),
                float(np.nanmax(corr_maps)),
            ],
            "saved_files": saved_paths,
        }

    # ── 3. Metadata ──────────────────────────────────────────────────────────
    print("\n[ 3 / 3 ]  Saving run metadata …")
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_pickle": str(Path(pickle_path).resolve()),
        "output_dir": str(output_dir.resolve()),
        "grid_shape": [GRID_ROWS, GRID_COLS],
        "n_pixels": N_PIXELS,
        "nan_policy": nan_policy,
        "save_matrices": save_matrices,
        "quantities": meta_quantities,
        "output_files": saved_paths_all,
    }
    save_metadata(output_dir, meta)

    print(f"\n{'='*60}")
    print("  All done.")
    print(f"{'='*60}\n")

    return meta


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spatial_correlation",
        description=(
            "Compute per-pixel spatial correlation maps for ACORN/AMPERE "
            "test-set results."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pickle",
        metavar="RESULTS_PICKLE",
        help="Path to the results pickle file.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        metavar="DIR",
        help="Directory in which to write output files (default: current dir).",
    )
    parser.add_argument(
        "--save-matrices",
        action="store_true",
        default=False,
        help=(
            "Also save the full (1200, 1200) correlation matrix for each "
            "quantity (~6 MB each as float32)."
        ),
    )
    parser.add_argument(
        "--nan-policy",
        choices=["propagate", "omit"],
        default="propagate",
        help=(
            "How to handle NaN values: 'propagate' (default) marks affected "
            "pixels as NaN; 'omit' replaces NaNs with the per-pixel mean."
        ),
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    try:
        run(
            pickle_path=args.pickle,
            output_dir=args.output_dir,
            save_matrices=args.save_matrices,
            nan_policy=args.nan_policy,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
