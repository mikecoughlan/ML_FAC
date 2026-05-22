"""
correlation_length.py
=====================
Computes a per-pixel Gaussian correlation length map from the spatial
correlation maps produced by ``spatial_correlation.py``.

For each of the 1200 reference pixels in the 50×24 (colatitude × MLT) FAC
grid the following steps are performed:

1.  Retrieve the 50×24 correlation map for that pixel from the pre-computed
    ``corr_maps`` array (shape ``(1200, 50, 24)``).
2.  Compute the angular distance (degrees) from the reference pixel to every
    other pixel using the spherical haversine formula.  MLT wrap-around
    (col 23 ↔ col 0) is handled automatically because the haversine operates
    on actual longitudes.
3.  Fit a Gaussian decay model  r(d) = exp(−d² / 2λ²)  to the scatter of
    (distance, correlation) pairs and extract λ (the correlation length in
    degrees).
4.  Store the result in a ``(50, 24)`` output map.

The haversine distance calculation deliberately ignores polar-geometry
complications (equal-area weighting, convergence of meridians at the pole)
for simplicity — this is a reasonable approximation for exploratory analysis.

Output files
------------
    corr_length_predicted_mean.npy   – float32 (50, 24), degrees
    corr_length_predicted_std.npy    – float32 (50, 24), degrees
    corr_length_ampere.npy           – float32 (50, 24), degrees
    corr_length_meta.json            – run metadata

Usage
-----
    python correlation_length.py
    python correlation_length.py --input-dir ./corr_results --output-dir ./corr_results

Dependencies
------------
    numpy, scipy, json (stdlib), argparse (stdlib), tqdm (optional)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):   # type: ignore[misc]
        desc = kwargs.get("desc", "")
        if desc:
            print(f"{desc} ...")
        return iterable


# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------

GRID_ROWS: int = 50
GRID_COLS: int = 24
N_PIXELS: int = GRID_ROWS * GRID_COLS


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------

def pixel_distances(
    ref_row: int,
    ref_col: int,
    grid_rows: int = GRID_ROWS,
    grid_cols: int = GRID_COLS,
) -> np.ndarray:
    """Compute the angular distance (degrees) from a reference pixel to every
    other pixel in the grid using the spherical haversine formula.

    Each pixel's position is taken as the centre of its bin:
        colatitude = row + 0.5  (degrees from pole)
        longitude  = (col + 0.5) / grid_cols * 360  (degrees, periodic)

    MLT wrap-around is handled automatically: because the haversine works with
    actual spherical coordinates, pixels at col 0 (00 MLT) and col 23 (23 MLT)
    are correctly found to be one bin-width apart with no special casing.

    Polar-geometry complications (non-equal-area bins, meridian convergence)
    are intentionally ignored.

    Parameters
    ----------
    ref_row : int
        Colatitude bin index of the reference pixel (0 = pole).
    ref_col : int
        MLT bin index of the reference pixel (0 = 00 MLT).
    grid_rows : int, optional
        Number of colatitude bins.  Defaults to 50.
    grid_cols : int, optional
        Number of MLT bins.  Defaults to 24.

    Returns
    -------
    dist_deg : np.ndarray, shape (grid_rows, grid_cols)
        Angular distance in degrees from the reference pixel to every pixel.
        The reference pixel itself has distance 0.
    """
    rows = np.arange(grid_rows)
    cols = np.arange(grid_cols)
    R, C = np.meshgrid(rows, cols, indexing='ij')   # (grid_rows, grid_cols)

    colat_ref = np.radians(ref_row + 0.5)
    lon_ref   = np.radians((ref_col + 0.5) / grid_cols * 360)

    colat = np.radians(R + 0.5)
    lon   = np.radians((C + 0.5) / grid_cols * 360)

    dlat = colat - colat_ref
    dlon = lon   - lon_ref

    a = (np.sin(dlat / 2) ** 2
         + np.sin(colat) * np.sin(colat_ref) * np.sin(dlon / 2) ** 2)
    dist_deg = np.degrees(2 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0)))
    return dist_deg


# ---------------------------------------------------------------------------
# Gaussian fit
# ---------------------------------------------------------------------------

def _gaussian(d: np.ndarray, lam: float) -> np.ndarray:
    """Gaussian decay model:  r(d) = exp(−d² / 2λ²)."""
    return np.exp(-(d ** 2) / (2 * lam ** 2))


def fit_gaussian_correlation_length(
    corr_values: np.ndarray,
    distances: np.ndarray,
    lam_init: float = 5.0,
    lam_bounds: tuple = (0.1, 180.0),
    min_points: int = 10,
) -> float:
    """Fit a Gaussian decay model to a scatter of (distance, correlation) pairs
    and return the correlation length λ in degrees.

    The model is::

        r(d) = exp(−d² / 2λ²)

    which gives r = 1 at d = 0 and r = exp(−0.5) ≈ 0.606 at d = λ.

    Only non-NaN points with positive correlation are used in the fit.  The
    self-correlation point (d = 0, r = 1) is excluded because it trivially
    satisfies the model and would bias λ downward.

    Parameters
    ----------
    corr_values : np.ndarray, shape (N,)
        Flattened correlation values for the reference pixel.
    distances : np.ndarray, shape (N,)
        Corresponding angular distances in degrees.
    lam_init : float, optional
        Initial guess for λ.  Defaults to 5.0°.
    lam_bounds : tuple of (float, float), optional
        Lower and upper bounds for λ.  Defaults to (0.1°, 180°).
    min_points : int, optional
        Minimum number of valid data points required to attempt a fit.
        Returns NaN if fewer points are available.  Defaults to 10.

    Returns
    -------
    lam : float
        Fitted Gaussian correlation length in degrees, or ``np.nan`` if the
        fit fails or there are insufficient valid points.
    """
    # Exclude self, NaNs, and non-positive correlations
    valid = (distances > 0) & np.isfinite(corr_values) & (corr_values > 0)
    d_fit = distances[valid]
    r_fit = corr_values[valid]

    if len(d_fit) < min_points:
        return np.nan

    try:
        popt, _ = curve_fit(
            _gaussian,
            d_fit,
            r_fit,
            p0=[lam_init],
            bounds=([lam_bounds[0]], [lam_bounds[1]]),
            maxfev=2000,
        )
        return float(popt[0])
    except (RuntimeError, ValueError):
        return np.nan


# ---------------------------------------------------------------------------
# Per-pixel correlation length map
# ---------------------------------------------------------------------------

def compute_correlation_length_map(
    corr_maps: np.ndarray,
    lam_init: float = 5.0,
    lam_bounds: tuple = (0.1, 180.0),
    min_points: int = 10,
) -> np.ndarray:
    """Compute a (50, 24) Gaussian correlation length map.

    For each of the 1200 reference pixels the haversine distance to every
    other pixel is computed, the Gaussian model is fitted to the resulting
    (distance, correlation) scatter, and λ is stored in the output map.

    Parameters
    ----------
    corr_maps : np.ndarray, shape (1200, 50, 24)
        Per-pixel spatial correlation maps as produced by
        ``spatial_correlation.py``.
    lam_init : float, optional
        Initial guess for λ in degrees.  Defaults to 5.0°.
    lam_bounds : tuple of (float, float), optional
        Bounds on λ in degrees.  Defaults to (0.1°, 180°).
    min_points : int, optional
        Minimum valid points required for a fit.  Defaults to 10.

    Returns
    -------
    cl_map : np.ndarray, shape (50, 24), dtype float32
        Gaussian correlation length (λ) in degrees for each reference pixel.
        Pixels where the fit fails are NaN.

    Notes
    -----
    Runtime: fitting 1200 pixels takes roughly 5–15 seconds on a modern
    laptop.  A ``tqdm`` progress bar is shown if the package is installed.
    """
    H, W = GRID_ROWS, GRID_COLS
    cl_map = np.full((H, W), np.nan, dtype=np.float64)

    for ref_row in tqdm(range(H), desc="Computing correlation lengths", unit="row"):
        for ref_col in range(W):
            k = ref_row * W + ref_col

            corr_flat = corr_maps[k].astype(np.float64).ravel()   # (1200,)
            dist_flat = pixel_distances(ref_row, ref_col).ravel()  # (1200,)

            cl_map[ref_row, ref_col] = fit_gaussian_correlation_length(
                corr_flat, dist_flat,
                lam_init=lam_init,
                lam_bounds=lam_bounds,
                min_points=min_points,
            )

    return cl_map.astype(np.float32)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    input_dir: str | Path = ".",
    output_dir: str | Path = ".",
    lam_init: float = 5.0,
    lam_bounds: tuple = (0.1, 180.0),
) -> dict:
    """End-to-end correlation length pipeline.

    Loads the three pre-computed correlation map files from ``input_dir``,
    fits a Gaussian correlation length for each reference pixel, and saves
    the resulting ``(50, 24)`` maps.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing ``spatial_corr_predicted_mean.npy``,
        ``spatial_corr_predicted_std.npy``, and ``spatial_corr_ampere.npy``.
    output_dir : str or Path
        Directory in which output files are written.
    lam_init : float, optional
        Initial guess for λ (degrees).  Defaults to 5.0°.
    lam_bounds : tuple of (float, float), optional
        (min, max) bounds for λ (degrees).  Defaults to (0.1°, 180°).

    Returns
    -------
    dict
        Run metadata also written to ``corr_length_meta.json``.
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  AIMFAHR — Gaussian Correlation Length")
    print(f"{'='*60}")
    print(f"  Input  : {input_dir.resolve()}")
    print(f"  Output : {output_dir.resolve()}")
    print(f"  λ init / bounds : {lam_init}° / {lam_bounds}")
    print()

    quantities = {
        "predicted_mean": input_dir / "spatial_corr_predicted_mean.npy",
        "predicted_std":  input_dir / "spatial_corr_predicted_std.npy",
        "ampere":         input_dir / "spatial_corr_ampere.npy",
    }

    meta_quantities = {}
    saved_paths = {}

    for qty, npy_path in quantities.items():
        print(f"\n── {qty} ──")
        if not npy_path.exists():
            warnings.warn(f"File not found, skipping: {npy_path}", UserWarning)
            continue

        corr_maps = np.load(npy_path)   # (1200, 50, 24)
        print(f"  Loaded {corr_maps.shape} from {npy_path.name}")

        cl_map = compute_correlation_length_map(
            corr_maps,
            lam_init=lam_init,
            lam_bounds=lam_bounds,
        )

        out_path = output_dir / f"corr_length_{qty}.npy"
        np.save(out_path, cl_map)
        print(f"  Saved  → {out_path}")

        saved_paths[qty] = str(out_path)
        valid = cl_map[np.isfinite(cl_map)]
        meta_quantities[qty] = {
            "n_valid_pixels": int(np.isfinite(cl_map).sum()),
            "n_failed_pixels": int(np.isnan(cl_map).sum()),
            "lambda_mean_deg": float(valid.mean()) if len(valid) else None,
            "lambda_std_deg":  float(valid.std())  if len(valid) else None,
            "lambda_min_deg":  float(valid.min())  if len(valid) else None,
            "lambda_max_deg":  float(valid.max())  if len(valid) else None,
            "saved_file": str(out_path),
        }
        print(f"  λ mean={meta_quantities[qty]['lambda_mean_deg']:.2f}°  "
              f"std={meta_quantities[qty]['lambda_std_deg']:.2f}°  "
              f"range=[{meta_quantities[qty]['lambda_min_deg']:.2f}°, "
              f"{meta_quantities[qty]['lambda_max_deg']:.2f}°]")

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_dir":  str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "model": "Gaussian: r(d) = exp(-d^2 / 2*lambda^2)",
        "distance_metric": "haversine (spherical, ignores polar geometry)",
        "lam_init_deg": lam_init,
        "lam_bounds_deg": list(lam_bounds),
        "grid_shape": [GRID_ROWS, GRID_COLS],
        "quantities": meta_quantities,
    }
    json_path = output_dir / "corr_length_meta.json"
    with open(json_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\n  Saved metadata → {json_path}")
    print(f"\n{'='*60}\n  Done.\n{'='*60}\n")
    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="correlation_length",
        description="Compute Gaussian correlation length maps for ACORN/AMPERE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input-dir",  "-i", default=".", metavar="DIR",
                   help="Directory containing spatial_corr_*.npy files.")
    p.add_argument("--output-dir", "-o", default=".", metavar="DIR",
                   help="Directory in which to write output files.")
    p.add_argument("--lam-init",   type=float, default=5.0,
                   help="Initial guess for λ in degrees (default: 5.0).")
    p.add_argument("--lam-max",    type=float, default=180.0,
                   help="Upper bound for λ in degrees (default: 180.0).")
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()
    try:
        run(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            lam_init=args.lam_init,
            lam_bounds=(0.1, args.lam_max),
        )
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
