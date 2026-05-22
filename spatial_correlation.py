"""
spatial_correlation.py
======================
Computes per-pixel spatial correlation maps for ACORN FAC model outputs and
AMPERE ground-truth data over a complete test set.

For each of the 1200 pixels in a 50×24 (colatitude × MLT) FAC grid, this
module produces a 50×24 correlation map showing how strongly every other
pixel co-varies with that reference pixel across all test-set samples.  The
computation is carried out for three quantities stored in the results pickle:

    • "predicted"  – ACORN posterior mean  (50×24 float)
    • "std"        – ACORN posterior std   (50×24 float)
    • "ampere"     – AMPERE ground truth   (50×24 float)

The full per-pixel correlation tensor has shape (1200, 50, 24), which is
equivalent to the (1200×1200) Pearson correlation matrix reshaped for spatial
visualisation.

Output files (written to --output-dir, default: current directory)
-------------------------------------------------------------------
    spatial_corr_predicted.npy    – float32 array (1200, 50, 24)
    spatial_corr_std.npy          – float32 array (1200, 50, 24)
    spatial_corr_ampere.npy       – float32 array (1200, 50, 24)
    corr_matrix_predicted.npy     – float32 array (1200, 1200)   [optional]
    corr_matrix_std.npy           – float32 array (1200, 1200)   [optional]
    corr_matrix_ampere.npy        – float32 array (1200, 1200)   [optional]
    spatial_correlation_meta.json – run metadata (timestamps, shapes, …)

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

        "predicted"  : numpy array of shape (50, 24) – ACORN posterior mean
        "std"        : numpy array of shape (50, 24) – ACORN posterior std
        "ampere"     : numpy array of shape (50, 24) – AMPERE ground truth

    Missing keys for a given timestamp are handled gracefully: that timestamp
    is skipped for the affected quantity with a warning.

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


def extract_stacked_arrays(
    results: Dict,
    key: str,
    expected_shape: Tuple[int, int] = (GRID_ROWS, GRID_COLS),
) -> Tuple[np.ndarray, list]:
    """Extract and stack all arrays for a given quantity key.

    Iterates over every timestamp in *results*, reads ``results[ts][key]``,
    validates the shape, and stacks valid entries into a single array of shape
    ``(N, *expected_shape)``.

    Parameters
    ----------
    results : dict
        Top-level results dictionary (timestamp → sub-dict).
    key : str
        Sub-dictionary key to extract (e.g. ``"predicted"``, ``"std"``,
        ``"ampere"``).
    expected_shape : tuple of int, optional
        Expected spatial shape of each map.  Defaults to ``(50, 24)``.

    Returns
    -------
    stacked : np.ndarray, shape (N, rows, cols)
        Float64 array of all valid maps for this key.
    valid_timestamps : list
        Ordered list of timestamps whose maps were successfully extracted.

    Warns
    -----
    UserWarning
        For timestamps where the key is missing or the array has the wrong
        shape.  These timestamps are silently skipped.
    """
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

        if key not in entry:
            warnings.warn(
                f"Timestamp {ts!r}: key {key!r} not found – skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue

        arr = np.asarray(entry[key], dtype=np.float64)

        if arr.shape != expected_shape:
            warnings.warn(
                f"Timestamp {ts!r}, key {key!r}: expected shape "
                f"{expected_shape}, got {arr.shape} – skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue

        arrays.append(arr)
        valid_timestamps.append(ts)

    if not arrays:
        raise ValueError(
            f"No valid arrays found for key {key!r}.  "
            "Check that the pickle has the expected structure."
        )

    stacked = np.stack(arrays, axis=0)  # (N, rows, cols)
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

    Loads the test-set results pickle, computes the full Pearson spatial
    correlation matrix for ``"predicted"``, ``"std"``, and ``"ampere"``
    quantities, and saves per-pixel correlation maps as ``.npy`` files.

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
    quantities = ["predicted", "std", "ampere"]
    meta_quantities = {}
    saved_paths_all = {}

    for i, qty in enumerate(quantities, start=1):
        print(f"\n[ 2.{i} / 3 ]  Processing quantity: {qty!r} …")

        maps, valid_ts = extract_stacked_arrays(results, key=qty)
        N = maps.shape[0]
        print(f"  Valid samples : {N}")

        print("  Computing Pearson correlation matrix …", end="", flush=True)
        corr_mat = compute_correlation_matrix(maps, nan_policy=nan_policy)
        print(" done.")

        corr_maps = corr_matrix_to_maps(corr_mat)

        print("  Saving …")
        saved_paths = save_arrays(
            output_dir, corr_maps, corr_mat, qty, save_matrix=save_matrices
        )
        saved_paths_all[qty] = saved_paths

        meta_quantities[qty] = {
            "n_samples": N,
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
        "pickle",
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
