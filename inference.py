####################################################################################
#
# inference.py
#
# Inference wrapper for the ACORN field-aligned current model.
# Model architecture (ACORN, including the refinement head) lives in
# model_classes.py, imported below -- this file needs model_classes.py
# importable alongside it. Everything else (data loading, preprocessing,
# scaling) is handled here.
#
# Three calling modes (all via FACInference.predict()):
#   Single timestamp  : predict(timestamp="2023-05-06 05:00:00")  → (H, W)
#   Date range        : predict(start="2023-05-06 00:00:00",
#                               end="2023-05-06 06:00:00")         → (N, H, W)
#   Full day          : predict(date="2023-05-06")                 → (N, H, W)
#   Current time      : predict()                                  → (H, W)
#
# Data sources:
#   realtime=False (default) : historical OMNI 1-min CDFs fetched directly
#                              from NASA SPDF and cached in ~/.cache/omni_cdfs/
#                              (requires: cdflib, requests)
#   realtime=True            : live NOAA SWPC feed (last 24 h only)
# Data is fetched fresh on every predict() call.
#
#       and _run_model once inference is stable.
#
####################################################################################

from __future__ import annotations

import datetime
import json
import pickle
import platform
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.request import urlopen

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _resolve_data_dir(config: dict) -> dict:
    """If data_dir is a per-platform dict (e.g. {"Linux": "...", "Darwin":
    "..."}), resolve it to the path for the current machine via
    platform.system(). Left as a plain string, it's used unchanged --
    backward compatible with a single shared path.
    """
    data_dir = config.get("data_dir")
    if isinstance(data_dir, dict):
        system = platform.system()
        if system not in data_dir:
            raise KeyError(
                f"data_dir has no entry for this platform ('{system}') -- "
                f"add one to config.json's data_dir block. Available: {list(data_dir.keys())}"
            )
        config["data_dir"] = data_dir[system]
    return config


# ══════════════════════════════════════════════════════════════════════════════
# Model architecture  (imported from model_classes.py -- single source of
# truth. ACORN builds itself from model_config alone, so no
# architecture-selection branching is needed here: the same
# ACORN(**model_config) call serves every checkpoint.)
# ══════════════════════════════════════════════════════════════════════════════
sys.path.append(".")
import utils
from model_classes import ACORN


# ══════════════════════════════════════════════════════════════════════════════
# Solar wind loading  (inference-only subset of data_prep.PreparingData)
# ══════════════════════════════════════════════════════════════════════════════


def _fetch_f107() -> float:
    """
    Fetch the most recent F10.7 solar flux value from the NOAA SWPC JSON feed.
    Falls back to 150.0 (solar-cycle mean) if the request fails.
    """
    try:
        url      = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"
        response = urlopen(url, timeout=10)
        df       = pd.DataFrame(json.loads(response.read().decode("utf-8")))

        possible_cols = ["f107", "flux", "f10_7", "f10.7", "observed_flux", "radio_flux"]
        col = next((c for c in possible_cols if c in df.columns), None)
        if col is None:
            return 150.0

        df["Epoch"] = pd.to_datetime(df["time_tag"])
        df[col]     = pd.to_numeric(df[col], errors="coerce")

        today     = datetime.datetime.utcnow().date()
        yesterday = today - datetime.timedelta(days=1)
        for day in (today, yesterday):
            subset = df[df["Epoch"].dt.date == day]
            val    = subset[col].dropna()
            if not val.empty:
                return float(val.iloc[-1])
        return float(df[col].dropna().iloc[-1])

    except Exception as e:
        print(f"Warning: could not fetch F10.7 ({e}). Using fallback value 150.0.")
        return 150.0


def _fetch_noaa_realtime() -> Optional[pd.DataFrame]:
    """
    Fetch the last 24 hours of real-time solar wind plasma and IMF data from
    NOAA SWPC. Returns a DatetimeIndex DataFrame with columns:
        density, speed, bx_gsm, by_gsm, bz_gsm
    Returns None if the request fails.
    """
    try:
        print("Fetching real-time solar wind data from NOAA SWPC...")

        plasma_url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json"
        mag_url    = "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json"

        plasma_json = json.loads(urlopen(plasma_url, timeout=10).read().decode("utf-8"))
        mag_json    = json.loads(urlopen(mag_url,    timeout=10).read().decode("utf-8"))

        plasma = pd.DataFrame(plasma_json[1:], columns=plasma_json[0])
        mag    = pd.DataFrame(mag_json[1:],    columns=mag_json[0])

        plasma["Epoch"] = pd.to_datetime(plasma["time_tag"])
        mag["Epoch"]    = pd.to_datetime(mag["time_tag"])

        for col in ["density", "speed", "temperature"]:
            plasma[col] = pd.to_numeric(plasma.get(col, np.nan), errors="coerce")
        for col in ["bx_gsm", "by_gsm", "bz_gsm", "bt", "lon_gsm", "lat_gsm"]:
            mag[col] = pd.to_numeric(mag.get(col, np.nan), errors="coerce")

        combined = (
            pd.merge(plasma, mag, on="Epoch", how="inner")
            .set_index("Epoch")
            .sort_index()
            [["density", "speed", "bx_gsm", "by_gsm", "bz_gsm"]]
        )

        print(f"Fetched {len(combined)} real-time data points "
              f"({combined.index.min()} -> {combined.index.max()})")
        return combined

    except Exception as e:
        print(f"Error fetching NOAA real-time data: {e}")
        return None


def _load_solarwind_realtime(config: dict, vars_to_keep: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Build the model input DataFrame from live NOAA SWPC feeds instead of
    Mirrors the column layout of _load_solarwind_omni() so the same
    scaler and sequence-building logic applies.

    Note: SuperMAG indices (SML, SMU, SYM_H, ASY_H, SME) are not available
    from NOAA in real time. They are filled with NaN and forward/back-filled
    from whatever recent values exist; predictions will be degraded when these
    indices are missing.
    """
    if vars_to_keep is None:
        vars_to_keep = config["input_params"]

    solarwind = _fetch_noaa_realtime()
    if solarwind is None:
        raise RuntimeError("Real-time NOAA fetch failed; cannot build input data.")

    if not isinstance(solarwind.index, pd.DatetimeIndex):
        solarwind.index = pd.to_datetime(solarwind.index, errors="coerce")

    # Resample to 1-min resolution to match training data cadence
    solarwind = solarwind.resample("1min").interpolate(method="linear", limit=10)

    # Cyclical month encoding
    months = solarwind.index.month
    solarwind["month"]     = months
    solarwind["sin_month"] = np.sin(months * 2 * np.pi / 12)
    solarwind["cos_month"] = np.cos(months * 2 * np.pi / 12)

    # F10.7 — single scalar broadcast across the full index
    solarwind["F107"] = _fetch_f107()

    # Rename NOAA columns to match training feature names
    solarwind = solarwind.rename(columns={
        "bx_gsm":  "BX_GSE",
        "by_gsm":  "BY_GSM",
        "bz_gsm":  "BZ_GSM",
        "speed":   "Vx",
        "density": "proton_density",
    })

    # SuperMAG indices are unavailable in real time — insert NaN columns so
    # the DataFrame has the right shape, then interpolate what we can
    supermag_cols = ["SML", "SMU", "SYM_H", "ASY_H", "SME"]
    for col in supermag_cols:
        if col not in solarwind.columns:
            solarwind[col] = np.nan

    solarwind = solarwind.interpolate(method="linear", limit=10).ffill().bfill()

    # Keep only the model input columns, drop storm param
    available = [c for c in vars_to_keep if c in solarwind.columns]
    missing   = [c for c in vars_to_keep if c not in solarwind.columns]
    if missing:
        print(f"Warning: real-time feed is missing columns {missing}. "
              "Predictions may be unreliable.")
    solarwind = solarwind[available]
    solarwind.dropna(inplace=True)

    return solarwind


# OMNI variable names -> training feature names
# AU_INDEX / AL_INDEX are the OMNI equivalents of SMU / SML
_OMNI_COL_MAP = {
    "Vx":             "Vx",
    "BX_GSE":         "BX_GSE",
    "BY_GSM":         "BY_GSM",
    "BZ_GSM":         "BZ_GSM",
    "proton_density": "proton_density",
    "SYM_H":          "SYM_H",
    "ASY_H":          "ASY_H",
    "AU_INDEX":       "SMU",
    "AL_INDEX":       "SML",
}

# Fill values used in OMNI CDFs for each variable (from omnitxtcdf.py metadata).
#
# Any variable absent from this table falls back to OMNI_FILLVAL_DEFAULT
# below. That default is NaN, meaning "no fill value is known for this
# variable" -- the comparison against NaN is always False, so nothing is
# masked and the raw values pass through untouched. This is deliberate:
# guessing a sentinel for an unknown variable risks silently deleting
# real measurements that happen to be large, which is a worse failure
# than leaving a fill value in place where it can still be spotted.
#
# Preferred fix for a missing entry is to add it here rather than to
# lean on the default. Note processing_omni.py takes the other approach
# for bulk processing, reading FILLVAL from each variable's own CDF
# attributes.
OMNI_FILLVALS = {
    "Vx":             99999.9,
    "BX_GSE":         9999.99,
    "BY_GSM":         9999.99,
    "BZ_GSM":         9999.99,
    "proton_density": 999.99,
    "SYM_H":          99999.0,
    "ASY_H":          99999.0,
    "AU_INDEX":       99999.0,
    "AL_INDEX":       99999.0,
}

# Sentinel meaning "unknown fill value"; see the note above.
OMNI_FILLVAL_DEFAULT = np.nan

_OMNI_CACHE_DIR = Path.home() / ".cache" / "omni_cdfs"
_OMNI_BASE_URL  = "https://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro_1min"


def _resolve_omni_cdf_filename(dt: datetime.datetime) -> str:
    """
    Resolve the exact CDF filename for a given month by scraping the SPDF
    directory listing. This handles version number changes (v01, v02, etc.)
    and avoids hardcoding a suffix that may be wrong.
    """
    import requests as _requests

    dir_url = f"{_OMNI_BASE_URL}/{dt.year}/"
    prefix  = f"omni_hro_1min_{dt.year}{dt.month:02d}"
    r = _requests.get(dir_url, timeout=30)
    r.raise_for_status()

    # Pull all .cdf filenames matching the year/month prefix from the listing
    matches = [
        tok for tok in r.text.split('"')
        if tok.startswith(prefix) and tok.endswith(".cdf")
    ]
    if not matches:
        month_str = dt.strftime("%Y-%m")
        raise FileNotFoundError(
            f"No OMNI CDF found for {month_str} at {dir_url}. "
            f"Directory listing snippet: {r.text[:500]}"
        )
    # Use the last match (highest version number)
    return sorted(matches)[-1]


def _fetch_omni_cdf(dt: datetime.datetime) -> Path:
    """
    Download the monthly 1-min OMNI CDF for the month containing dt if not
    already cached. Resolves the exact filename from the SPDF directory listing
    so version number changes are handled automatically.
    Returns the local path to the CDF file.
    """
    import requests as _requests

    _OMNI_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Check cache for any existing file matching this year/month
    prefix   = f"omni_hro_1min_{dt.year}{dt.month:02d}"
    existing = list(_OMNI_CACHE_DIR.glob(f"{prefix}*.cdf"))
    if existing:
        return existing[0]

    fn    = _resolve_omni_cdf_filename(dt)
    local = _OMNI_CACHE_DIR / fn
    url   = f"{_OMNI_BASE_URL}/{dt.year}/{fn}"

    print(f"Downloading OMNI CDF: {url}")
    r = _requests.get(url, timeout=120)
    r.raise_for_status()
    local.write_bytes(r.content)
    print(f"Saved to {local}")

    return local


def _read_omni_cdf(cdf_path: Path, startdt: datetime.datetime,
                   enddt: datetime.datetime) -> pd.DataFrame:
    """
    Read required variables from a single OMNI 1-min CDF file using cdflib,
    trim to [startdt, enddt], replace fill values with NaN, and return a
    DatetimeIndex DataFrame with training-aligned column names.
    """
    import cdflib

    cdf    = cdflib.CDF(str(cdf_path))
    epoch  = cdflib.cdfepoch.to_datetime(cdf["Epoch"])
    index  = pd.DatetimeIndex(epoch).tz_localize(None)

    mask = (index >= pd.Timestamp(startdt)) & (index <= pd.Timestamp(enddt))

    data = {"Epoch": index[mask]}
    for omni_name, col_name in _OMNI_COL_MAP.items():
        try:
            vals = np.array(cdf[omni_name], dtype=float)[mask]
        except Exception:
            vals = np.full(mask.sum(), np.nan)
        # Replace fill values with NaN. The 0.99 factor catches values
        # that sit just below the declared sentinel through rounding or
        # unit conversion. If no fill value is known the threshold is
        # NaN, the comparison is uniformly False, and nothing is masked.
        fill = OMNI_FILLVALS.get(omni_name, OMNI_FILLVAL_DEFAULT)
        vals[np.abs(vals) >= np.abs(fill) * 0.99] = np.nan
        data[col_name] = vals

    return pd.DataFrame(data).set_index("Epoch")


def _load_solarwind_omni(config: dict, startdt: datetime.datetime,
                         enddt: datetime.datetime,
                         vars_to_keep: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fetch OMNI 1-min solar wind data for [startdt, enddt] directly from NASA
    SPDF, with local CDF caching (~/.cache/omni_cdfs/). No third-party
    packages beyond cdflib (already required by processing_omni.py) are needed.

    OMNI variable name mapping to training feature names:
        AU_INDEX -> SMU
        AL_INDEX -> SML
        SME      -> derived as SMU - SML

    Returns a DatetimeIndex DataFrame with only the model input columns,
    interpolated up to 10 minutes, and NaNs dropped.
    """
    if vars_to_keep is None:
        vars_to_keep = config["input_params"]

    # Collect all months that span [startdt, enddt]
    months, cur = [], datetime.datetime(startdt.year, startdt.month, 1)
    while cur <= enddt:
        months.append(cur)
        cur = (cur + datetime.timedelta(days=32)).replace(day=1)

    # OMNI data is only available with a ~3-month processing lag.
    # If the requested window is too recent, fall back to the NOAA live feed.
    omni_cutoff = (datetime.datetime.utcnow()
                   - datetime.timedelta(days=90)).replace(day=1)
    if startdt >= omni_cutoff:
        print(
            f"Requested window {startdt.strftime('%Y-%m-%d')} is within the "
            f"~3-month OMNI processing lag — falling back to NOAA live feed."
        )
        return _load_solarwind_realtime(config)

    # Download (if needed) and read each monthly CDF
    frames = []
    errors = []
    for month_dt in months:
        try:
            cdf_path = _fetch_omni_cdf(month_dt)
            df       = _read_omni_cdf(cdf_path, startdt, enddt)
            frames.append(df)
        except Exception as e:
            errors.append(f"  {month_dt.strftime('%Y-%m')}: {e}")

    if not frames:
        error_detail = "\n".join(errors)
        raise RuntimeError(
            f"Failed to fetch any OMNI data for {startdt} -> {enddt}.\n"
            f"Errors:\n{error_detail}"
        )
    if errors:
        print("Warning: some months could not be fetched:\n" + "\n".join(errors))

    solarwind = pd.concat(frames).sort_index()
    solarwind = solarwind[~solarwind.index.duplicated(keep="first")]

    # Derive SME = SMU - SML
    solarwind["SME"] = solarwind["SMU"] - solarwind["SML"]

    # Cyclical month encoding
    months_col = solarwind.index.month
    solarwind["month"]     = months_col
    solarwind["sin_month"] = np.sin(months_col * 2 * np.pi / 12)
    solarwind["cos_month"] = np.cos(months_col * 2 * np.pi / 12)

    # F10.7 — fetch live scalar and broadcast
    solarwind["F107"] = _fetch_f107()

    # Interpolate gaps up to 10 minutes then drop remaining NaNs
    solarwind = solarwind.interpolate(method="linear", limit=10).ffill().bfill()

    # Keep only model input columns, drop storm param
    available = [c for c in vars_to_keep if c in solarwind.columns]
    missing   = [c for c in vars_to_keep if c not in solarwind.columns]
    if missing:
        print(f"Warning: OMNI fetch is missing columns {missing}.")
    solarwind = solarwind[available]
    solarwind.dropna(inplace=True)

    return solarwind



# ══════════════════════════════════════════════════════════════════════════════
# Inference wrapper
# ══════════════════════════════════════════════════════════════════════════════

class FACInference:
    """
    Inference wrapper for the trained ACORN field-aligned current model.

    Loads config, scaler, model weights, and solar wind data once at
    construction. Subsequent predict() calls are cheap.

    Parameters
    ----------
    config_path : str
        Path to the consolidated config.json (global params + per-target
        sci/op blocks).
    model_variant : str, optional
        Which model ("sci" or "op") to use. Defaults to whatever
        config.json's "active_model" says. Pass this explicitly when you
        need both sci and op in the same script/session (e.g. building
        acorn_results and op_results side by side) -- config.json's
        active_model is a single global default, not something you'd want
        to mutate per-call.
    model_path : str, optional
        Override the model checkpoint path. Defaults to the path derived
        from config (model_dir / MODEL_VERSION_ERAS.pt).
    lookback_limit : int, optional
        Maximum number of NaN rows that may be skipped while building a
        sequence window. If more than this many NaN rows are encountered
        before collecting `time_history` valid rows, the timestamp is
        skipped. Default is 10.
    realtime : bool, optional
        If True, fetch solar wind data from NOAA SWPC live feeds (last 24 h).
        If False (default), fetch historical data from NASA OMNI via
        NASA SPDF OMNI CDFs for the exact window requested (cached locally).
        Note that in realtime mode SuperMAG indices are unavailable and
        will be NaN-filled.

    Config keys of note
    --------------------
    model_config : dict
        Whatever's in this block flows straight into ACORN(**model_config).
        Attention is switchable via use_cbam / use_attention_gates; the
        refinement head is always built. No separate
        architecture-selection key needed -- ACORN figures out what to
        build from model_config alone.

    Examples
    --------
    # Uses config.json's active_model
    wrapper = FACInference("config.json")

    # Explicit model -- both usable in the same script without touching the file
    sci_wrapper = FACInference("config.json", model_variant="sci")
    op_wrapper = FACInference("config.json", model_variant="op")

    # Single timestamp -> arrays of shape (H, W)
    mean, std = wrapper.predict(timestamp="2023-05-06 05:00:00")

    # Date range -> arrays of shape (N, H, W)
    mean, std = wrapper.predict(start="2023-05-06 00:00:00",
                                end="2023-05-06 06:00:00")

    # Full day -> arrays of shape (N, H, W)
    mean, std = wrapper.predict(date="2023-05-06")

    # Conv-head model -- pass model_path explicitly, since experimental
    # checkpoint filenames don't follow this file's default naming derivation
    wrapper = FACInference("config.json", model_variant="sci",
                           model_path="models/acorn_sci.pt")
    """

    def __init__(
        self,
        config_path:     str           = "config.json",
        model_variant:   Optional[str] = None,   # override config.json's active_model -- "sci" or "op"
        model_path:      Optional[str] = None,
        lookback_limit:  int           = 10,
        realtime:        bool          = False,
    ):
        # utils.load_config performs the shared/per-model merge and
        # validates the model name, so inference and training resolve
        # configuration identically.
        self.config = _resolve_data_dir(utils.load_config(model_variant, config_path))
        self._model_variant = self.config["model_name"]

        self._time_history = self.config.get("time_history", 60)
        self._ampere_delay = self.config.get("ampere_delay", 0)
        self._here         = Path(config_path).resolve().parent
        self._lookback_limit = lookback_limit

        # ── Model path ────────────────────────────────────────────────────────
        if model_path is not None:
            self._model_path = Path(model_path)
        else:
            # Canonical name from utils, so training and inference cannot
            # disagree about where the weights live.
            self._model_path = self._here / utils.model_file(self.config)
            print(f'MODEL PATH: {self._model_path}')

        # ── Scaler ────────────────────────────────────────────────────────────
        scaler_path = self._here / utils.scaler_file(self.config)
        print(f'SCALER PATH: {scaler_path}')
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler not found at {scaler_path}. "
                "Run training first to generate a fitted scaler."
            )
        with open(scaler_path, "rb") as f:
            self._scaler: StandardScaler = pickle.load(f)
        print(f"Scaler loaded  : {scaler_path}")

        # ── Model ─────────────────────────────────────────────────────────────
        self._model = self._load_model()
        print(f"Model loaded   : {self._model_path}  (device: {DEVICE})")

        self._realtime = realtime

        # ── Solar wind ────────────────────────────────────────────────────────
        # Data is fetched per predict() call once we know the requested window.
        # Initialise empty structures here; _refresh_solarwind fills them.
        self._sw_values    = np.empty((0,))
        self._sw_index     = pd.DatetimeIndex([])
        self._index_to_pos = {}
        if self._realtime:
            print("Real-time mode — NOAA SWPC data will be fetched on each predict() call.")
        else:
            print("OMNI mode — data will be fetched per predict() call from NASA SPDF.")

    # ──────────────────────────────────────────────────────────────────────────

    def _refresh_solarwind(self,
                           startdt: Optional[datetime.datetime] = None,
                           enddt:   Optional[datetime.datetime] = None,
                           ) -> None:
        """
        Fetch solar wind data and rebuild the lookup structures.
        Called once at construction and again at the start of every predict()
        call when realtime=True.

        Data source:
          - realtime=True               -> NOAA SWPC live feed (last 24 h)
          - realtime=False              -> NASA SPDF OMNI 1-min CDF fetch
        """
        if self._realtime:
            solarwind = _load_solarwind_realtime(self.config)
        else:
            if startdt is None or enddt is None:
                raise ValueError(
                    "startdt and enddt must be provided for OMNI historical fetch."
                )
            print(f"Fetching historical OMNI data: {startdt} -> {enddt}")
            solarwind = _load_solarwind_omni(self.config, startdt, enddt)

        self._sw_values    = solarwind.to_numpy()
        self._sw_index     = solarwind.index
        self._index_to_pos = {ts: pos for pos, ts in enumerate(self._sw_index)}
        print(
            f"Solar wind ready: {self._sw_index[0]} -> {self._sw_index[-1]} "
            f"({len(self._sw_index):,} minutes, {len(solarwind.columns)} features)"
        )

    def predict(
        self,
        timestamp: Optional[str] = None,
        start:     Optional[str] = None,
        end:       Optional[str] = None,
        date:      Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, object]:
        """
        Run inference for one or more timestamps.

        Provide exactly one of:
            timestamp="YYYY-MM-DD HH:MM:SS"   -> single prediction, returns (H, W)
            date="YYYY-MM-DD"                  -> full day,   returns (N, H, W)
            start=..., end=...                 -> date range, returns (N, H, W)

        In realtime mode, solar wind data is re-fetched from NOAA SWPC on
        every call so predictions always use the latest available data.

        Returns
        -------
        mean : np.ndarray            Predicted mean FAC map(s).
        std  : np.ndarray            Predicted std  FAC map(s).
        time : pd.Timestamp or list  Timestamp(s) corresponding to predictions.
        """
        startdt, enddt = self._predict_window(timestamp, start, end, date)
        self._refresh_solarwind(startdt=startdt, enddt=enddt)

        timestamps = self._resolve_timestamps(timestamp, start, end, date)

        if len(timestamps) == 0:
            raise ValueError("No timestamps resolved from the given input.")

        sequences, valid_ts = self._build_sequences(timestamps)

        if len(sequences) == 0:
            raise ValueError(
                "No valid sequences could be built. Timestamps may fall outside "
                f"the solar wind data range or lack {self._time_history} steps of history."
            )

        mean, std = self._run_model(sequences)

        # Squeeze batch dim when only one timestep was produced -> (H, W)
        if mean.shape[0] == 1:
            mean = mean.squeeze(0)
            std  = std.squeeze(0)
            time = valid_ts[0]
        else:
            time = valid_ts

        return mean, std, time, sequences

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _predict_window(
        self,
        timestamp: Optional[str],
        start:     Optional[str],
        end:       Optional[str],
        date:      Optional[str],
    ) -> Tuple[datetime.datetime, datetime.datetime]:
        """
        Derive a (startdt, enddt) window that covers the requested timestamps
        plus enough look-back history for sequence building. Used to know what
        range to fetch from OMNI.
        """
        pad = datetime.timedelta(minutes=self._time_history + self._lookback_limit + 10)

        if timestamp is not None:
            dt = pd.to_datetime(timestamp).to_pydatetime()
            return dt - pad, dt

        if date is not None:
            day = pd.to_datetime(date).to_pydatetime()
            return day - pad, day + datetime.timedelta(days=1)

        if start is not None and end is not None:
            s = pd.to_datetime(start).to_pydatetime()
            e = pd.to_datetime(end).to_pydatetime()
            return s - pad, e

        # Default (no args) — use a 2-hour window ending now
        now = datetime.datetime.utcnow()
        return now - pad - datetime.timedelta(hours=1), now

    def _resolve_timestamps(
        self,
        timestamp: Optional[str],
        start:     Optional[str],
        end:       Optional[str],
        date:      Optional[str],
    ) -> pd.DatetimeIndex:
        """Convert user-facing input into a sorted DatetimeIndex at 1-min resolution.
        If all arguments are None, defaults to the current UTC time rounded to the
        nearest minute.
        """
        if (start is None) != (end is None):
            raise ValueError("Both start and end must be provided together.")

        # Default: most recent timestamp available in the loaded SW index.
        # Using wall-clock "now" risks requesting a timestamp slightly ahead of
        # the latest data point (NOAA feed latency / resampling alignment), so
        # we use the last index entry instead.
        if all(v is None for v in (timestamp, start, end, date)):
            latest = self._sw_index.tz_localize(None).max()
            print(f"No timestamp given — using latest available SW timestamp: {latest}")
            return pd.DatetimeIndex([latest])

        n_provided = sum([
            timestamp is not None,
            date is not None,
            start is not None or end is not None,
        ])
        if n_provided != 1:
            raise ValueError(
                "Provide exactly one of: timestamp, date, or (start + end)."
            )

        if timestamp is not None:
            return pd.DatetimeIndex([pd.to_datetime(timestamp)])

        if date is not None:
            day = pd.to_datetime(date)
            return pd.date_range(
                start=day,
                end=day + pd.Timedelta(days=1) - pd.Timedelta(minutes=1),
                freq="1min",
            )

        return pd.date_range(
            start=pd.to_datetime(start),
            end=pd.to_datetime(end),
            freq="1min",
        )

    def _build_sequences(
        self,
        timestamps: pd.DatetimeIndex,
    ) -> Tuple[np.ndarray, List[pd.Timestamp]]:
        """
        Slice a (time_history, n_features) window from the solar wind array for
        each requested timestamp, apply the stored scaler, and stack into a
        single (N, time_history, n_features) array.

        NaN handling: if any row in the nominal window is NaN (missing data),
        the window looks back further in time to collect the last `time_history`
        fully-valid rows available before the target timestamp. The search stops
        early if more than `lookback_limit` NaN rows are encountered, in which
        case the timestamp is skipped entirely.
        """
        windows:  List[np.ndarray]   = []
        valid_ts: List[pd.Timestamp] = []

        # ── Normalise index timezone ───────────────────────────────────────────
        # The SW index may be tz-aware (e.g. UTC from NOAA) while requested
        # timestamps are tz-naive, or vice-versa. Strip tz from both so that
        # lookups against _index_to_pos always match.
        if self._sw_index.tz is not None:
            print(f"Note: solar wind index is tz-aware ({self._sw_index.tz}), "
                  "converting to tz-naive UTC for lookup.")
            self._sw_index     = self._sw_index.tz_localize(None)
            self._index_to_pos = {ts: pos for pos, ts in enumerate(self._sw_index)}

        timestamps = pd.DatetimeIndex(timestamps).tz_localize(None)

        # ── Diagnostics ───────────────────────────────────────────────────────
        n_not_in_index   = 0
        n_bad_end_pos    = 0
        n_insufficient   = 0
        n_lookback_limit = 0

        for ts in timestamps:
            if ts not in self._index_to_pos:
                n_not_in_index += 1
                continue

            end_pos = self._index_to_pos[ts] - self._ampere_delay

            if end_pos < 0:
                n_bad_end_pos += 1
                continue

            # Collect valid (non-NaN) rows by scanning backwards from end_pos.
            # At most (time_history + lookback_limit) rows are examined — i.e.
            # up to lookback_limit NaN rows may be skipped over in total.
            collected: List[np.ndarray] = []
            pos       = end_pos
            skipped   = 0
            while pos >= 0 and len(collected) < self._time_history and skipped <= self._lookback_limit:
                row = self._sw_values[pos, :]
                if not np.isnan(row).any():
                    collected.append(row)
                else:
                    skipped += 1
                pos -= 1

            if len(collected) < self._time_history:
                if skipped > self._lookback_limit:
                    n_lookback_limit += 1
                else:
                    n_insufficient += 1
                continue  # Not enough valid history before this timestamp

            # collected is newest-first; reverse to get chronological order
            window = np.array(collected[::-1])   # (time_history, n_features)

            windows.append(window)
            valid_ts.append(ts)

        if not windows:
            print(
                f"No input sequences could be built from {len(timestamps)} timestamp(s). "
                "Breakdown of why each was skipped:\n"
                f"  Not found in SW index : {n_not_in_index}\n"
                f"  Negative end_pos      : {n_bad_end_pos}\n"
                f"  Hit lookback_limit    : {n_lookback_limit}\n"
                f"  Insufficient history  : {n_insufficient}\n"
                f"  SW index range        : {self._sw_index[0]} -> {self._sw_index[-1]}\n"
                f"  Requested range       : {timestamps[0]} -> {timestamps[-1]}"
            )
            return np.empty(0), []


        sequences = np.stack(windows, axis=0)           # (N, T, F)
        N, T, F   = sequences.shape
        scaled    = self._scaler.transform(sequences.reshape(N * T, F))
        sequences = scaled.reshape(N, T, F)

        return sequences, valid_ts

    def _run_model(
        self,
        sequences: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through ACORN.

        Input  tensor shape : (N, 1, T, F)  -- channel dim added for Conv2d
        Output tensor shape : (N, 2, H, W)  -- ch 0 = mean, ch 1 = std

        Returns
        -------
        mean : np.ndarray  shape (N, H, W)
        std  : np.ndarray  shape (N, H, W)
        """
        X = torch.tensor(sequences, dtype=torch.float32).unsqueeze(1)  # (N, 1, T, F)

        self._model.eval()
        self._model.to(DEVICE)

        with torch.no_grad():
            output = self._model(X.to(DEVICE))
            output = output.cpu().numpy()

        if output.ndim == 3:
            output = output[np.newaxis, ...]      # restore batch dim if squeezed

        if output.shape[3] > 24:
            mean = output[:, 0, :, 1:-1]
            std  = output[:, 1, :, 1:-1]
        else:
            mean = output[:, 0, :, :]
            std  = output[:, 1, :, :]


        return mean, std

    def load_ampere(self, timestamp: str) -> Optional[np.ndarray]:
        """
        Load the AMPERE observed current density for a given timestamp from
        the pre-computed pickle files in ampere_data/ (sibling of models/).

        Parameters
        ----------
        timestamp : str
            Timestamp string in "YYYY-MM-DD HH:MM:SS" format.
        Returns
        -------
        np.ndarray or None
            2D array of shape (lat, MLT) for the requested timestamp, or None
            if the timestamp is not found in the pickle files.
        """
        ampere_dir = self._here / "ampere_data"

        # Search all available pickle files regardless of era
        candidates = sorted(ampere_dir.glob("ampere*.pkl"))
        if not candidates:
            print(f"No AMPERE pickle files found in {ampere_dir}.")
            return None

        for pkl_path in candidates:
            with open(pkl_path, "rb") as f:
                ampere_dict = pickle.load(f)
            if timestamp in ampere_dict:
                data = ampere_dict[timestamp]

                if hasattr(data, "reindex"):
                    # Pivot DataFrame — reindex to the exact lat/MLT grid the
                    # polar plot uses so values land at the right coordinates
                    lat_grid = np.linspace(0, 50, 50, endpoint=False)
                    mlt_grid = np.linspace(0, 24, 24, endpoint=False)
                    data = (
                        data
                        .reindex(index=lat_grid,   method="nearest")
                        .reindex(columns=mlt_grid, method="nearest")
                    )
                    return data.to_numpy().astype(float)
                else:
                    # Already a numpy array (flattened from pivot_table row-major,
                    # i.e. lat-major order). Reshape directly to (50, 24).
                    arr = np.array(data, dtype=float).reshape(24, 50).T
                    print(f"[load_ampere] array reshaped to {arr.shape}, "
                          f"min={np.nanmin(arr):.3f}, max={np.nanmax(arr):.3f}")
                    return arr

        print(f"Timestamp '{timestamp}' not found in any AMPERE pickle in {ampere_dir}.")
        return None

    def _load_model(self) -> nn.Module:
        """Load ACORN weights from checkpoint using model_config from config.json.

        ACORN builds itself from model_config, so nothing here needs to
        branch on architecture -- the same ACORN(**model_config) call
        serves every checkpoint.
        """
        if not Path(self._model_path).exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self._model_path}"
            )

        checkpoint = torch.load(self._model_path, map_location=DEVICE)

        model_config = {**self.config["model_config"],
                        "output_size": tuple(self.config["output_size"])}

        state_dict = (
            checkpoint["model"]
            if isinstance(checkpoint, dict) and "model" in checkpoint
            else checkpoint
        )

        model = ACORN(**model_config)
        model.load_state_dict(state_dict)
        model.eval()
        return model



# ══════════════════════════════════════════════════════════════════════════════
# TF FAC Inference wrapper  (Keras/TensorFlow model)
# ══════════════════════════════════════════════════════════════════════════════

class TFACInference:
    """
    Inference wrapper for the TensorFlow/Keras FAC model (FAC_onlySW.hdf5).

    Handles its own data preparation pipeline, which differs from FACInference:
      - 7 input features:  Bx, By, Bz, Vx (negated speed), Np,
                           month_sine, month_cosine
      - Normalisation via input_mean_std.json (per-variable mean/std),
        not a sklearn StandardScaler
      - Output shape: (50, 25) — note 25 MLT columns vs ACORN's 24
      - Returns (mean, None) to match FACInference.predict() interface

    OMNI column mapping to model input names:
        BX_GSE        -> Bx
        BY_GSM        -> By
        BZ_GSM        -> Bz
        Vx            -> Vx  (negated: model expects -speed convention)
        proton_density -> Np

    Parameters
    ----------
    config_path : str
        Path to config.json (used for shared settings: time_history, etc.)
    model_path : str, optional
        Path to the .hdf5 model file. Defaults to models/FAC_onlySW.hdf5
        in the same directory as config.json.
    norm_path : str, optional
        Path to input_mean_std.json. Defaults to models/scalers/input_mean_std.json.
    lookback_limit : int, optional
        Max NaN rows to skip when building sequences. Default 10.
    realtime : bool, optional
        If True, fetch from NOAA SWPC live feed. If False, fetch from NASA
        SPDF OMNI CDFs. Default False.
    """

    def __init__(
        self,
        model_type:     str           = 'op',
        config_path:    str           = "config.json",
        model_path:     Optional[str] = None,
        norm_path:      Optional[str] = None,
        lookback_limit: int           = 10,
        realtime:       bool          = False,
    ):

        # Same merged config as the PyTorch wrapper; model_type selects
        # which block ('sci' or 'op'), rather than naming a separate file.
        self.config = utils.load_config(model_type, config_path)

        self._time_history  = self.config.get("time_history", 60)
        self._ampere_delay  = self.config.get("ampere_delay", 0)
        self._lookback_limit = lookback_limit
        self._realtime      = realtime
        self._here          = Path(config_path).resolve().parent

        # Feature order expected by the TF model
        self._INPUT_COLS = self.config["bk_input_params"]

        # OMNI column names -> TF model input names
        self._OMNI_RENAME = {
            "BZ_GSM":         "Bz",
            "BY_GSM":         "By",
            "BX_GSE":         "Bx",
            "Vx":             "Vx",
            "proton_density": "Np",
            "sin_month":      "month_sine",
            "cos_month":      "month_cosine",
        }

        # ── Model path ────────────────────────────────────────────────────────
        self._model_path = Path(model_path) if model_path is not None else (
            self._here / self.config.get("model_dir", "models/") / f"FAC_BK_{model_type}.hdf5"
        )

        # ── Normalisation JSON ────────────────────────────────────────────────
        self._norm_path = Path(norm_path) if norm_path is not None else (
            self._here
            / self.config.get("model_dir", "models/")
            / "scalers"
            / "input_mean_std.json"
        )
        if not self._norm_path.exists():
            raise FileNotFoundError(
                f"Normalisation file not found: {self._norm_path}"
            )
        with open(self._norm_path, "r") as f:
            self._norm = json.load(f)
        print(f"Norm loaded    : {self._norm_path}")

        # ── TF model ──────────────────────────────────────────────────────────
        self._model = self._load_tf_model()
        print(f"TF model loaded: {self._model_path}")

        # ── Solar wind (fetched per predict() call) ───────────────────────────
        self._sw_values    = np.empty((0,))
        self._sw_index     = pd.DatetimeIndex([])
        self._index_to_pos: dict = {}
        if self._realtime:
            print("TF real-time mode — NOAA SWPC data fetched on each predict() call.")
        else:
            print("TF OMNI mode — data fetched per predict() call from NASA SPDF.")

    # ──────────────────────────────────────────────────────────────────────────

    def predict(
        self,
        timestamp: Optional[str] = None,
        start:     Optional[str] = None,
        end:       Optional[str] = None,
        date:      Optional[str] = None,
    ) -> Tuple[np.ndarray, None, object]:
        """
        Run FAC inference for one or more timestamps.

        Provide exactly one of:
            timestamp="YYYY-MM-DD HH:MM:SS"   -> (50, 24)
            date="YYYY-MM-DD"                  -> (N, 50, 24)
            start=..., end=...                 -> (N, 50, 24)
            (no args)                          -> latest available timestamp

        Returns
        -------
        mean : np.ndarray            FAC prediction(s).
        None                         No uncertainty estimate for this model.
        time : pd.Timestamp or list  Timestamp(s) corresponding to predictions.
        """
        startdt, enddt = self._predict_window(timestamp, start, end, date)
        self._refresh_solarwind(startdt=startdt, enddt=enddt)

        timestamps = self._resolve_timestamps(timestamp, start, end, date)

        if len(timestamps) == 0:
            raise ValueError("No timestamps resolved from the given input.")

        sequences, valid_ts = self._build_sequences(timestamps)

        if len(sequences) == 0:
            raise ValueError(
                "No valid sequences could be built. Timestamps may fall outside "
                f"the solar wind data range or lack {self._time_history} steps of history."
            )

        mean = self._run_tf_model(sequences)

        if mean.shape[0] == 1:
            mean = mean.squeeze(0)
            time = valid_ts[0]
        else:
            time = valid_ts

        return mean, None, time

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _refresh_solarwind(self,
                           startdt: Optional[datetime.datetime] = None,
                           enddt:   Optional[datetime.datetime] = None,
                           ) -> None:
        """Fetch solar wind data and rebuild lookup structures."""
        # Request OMNI-named columns from the loader, then rename to TF names.
        # _OMNI_RENAME: {omni_name: tf_name}
        input_renamed = {v:k for k,v in self._OMNI_RENAME.items()}
        input_cols = [input_renamed.get(x,x) for x in self._INPUT_COLS]
        if self._realtime:
            solarwind = _load_solarwind_realtime(self.config, vars_to_keep=input_cols)
        else:
            if startdt is None or enddt is None:
                raise ValueError("startdt and enddt required for OMNI fetch.")
            print(f"Fetching OMNI data for TF model: {startdt} -> {enddt}")
            solarwind = _load_solarwind_omni(
                self.config, startdt, enddt, vars_to_keep=input_cols
            )

        # Rename OMNI column names to TF model input names
        solarwind = solarwind.rename(columns=self._OMNI_RENAME)
        # Negate Vx: OMNI Vx is negative by convention, TF model expects -speed
        if "Vx" in solarwind.columns:
            solarwind["Vx"] = -solarwind["Vx"]
        self._sw_values    = solarwind[self._INPUT_COLS].to_numpy()
        self._sw_index     = solarwind.index
        self._index_to_pos = {ts: pos for pos, ts in enumerate(self._sw_index)}
        print(
            f"SW ready (TF)  : {self._sw_index[0]} -> {self._sw_index[-1]} "
            f"({len(self._sw_index):,} minutes)"
        )

    def _predict_window(
        self,
        timestamp: Optional[str],
        start:     Optional[str],
        end:       Optional[str],
        date:      Optional[str],
    ) -> Tuple[datetime.datetime, datetime.datetime]:
        """Derive fetch window — identical logic to FACInference._predict_window."""
        pad = datetime.timedelta(
            minutes=self._time_history + self._lookback_limit + 10
        )
        if timestamp is not None:
            dt = pd.to_datetime(timestamp).to_pydatetime()
            return dt - pad, dt
        if date is not None:
            day = pd.to_datetime(date).to_pydatetime()
            return day - pad, day + datetime.timedelta(days=1)
        if start is not None and end is not None:
            s = pd.to_datetime(start).to_pydatetime()
            e = pd.to_datetime(end).to_pydatetime()
            return s - pad, e
        now = datetime.datetime.utcnow()
        return now - pad - datetime.timedelta(hours=1), now

    def _resolve_timestamps(
        self,
        timestamp: Optional[str],
        start:     Optional[str],
        end:       Optional[str],
        date:      Optional[str],
    ) -> pd.DatetimeIndex:
        """Identical timestamp resolution logic to FACInference."""
        if (start is None) != (end is None):
            raise ValueError("Both start and end must be provided together.")

        if all(v is None for v in (timestamp, start, end, date)):
            latest = self._sw_index.tz_localize(None).max()
            print(f"No timestamp given — using latest available SW timestamp: {latest}")
            return pd.DatetimeIndex([latest])

        n_provided = sum([
            timestamp is not None,
            date is not None,
            start is not None or end is not None,
        ])
        if n_provided != 1:
            raise ValueError(
                "Provide exactly one of: timestamp, date, or (start + end)."
            )

        if timestamp is not None:
            return pd.DatetimeIndex([pd.to_datetime(timestamp)])

        if date is not None:
            day = pd.to_datetime(date)
            return pd.date_range(
                start=day,
                end=day + pd.Timedelta(days=1) - pd.Timedelta(minutes=1),
                freq="1min",
            )

        return pd.date_range(
            start=pd.to_datetime(start),
            end=pd.to_datetime(end),
            freq="1min",
        )

    def _build_sequences(
        self,
        timestamps: pd.DatetimeIndex,
    ) -> Tuple[np.ndarray, List[pd.Timestamp]]:
        """
        Build (time_history, 7) windows, apply JSON-based normalisation,
        and stack into (N, time_history, 7). NaN skipping with lookback_limit
        is identical to FACInference._build_sequences.
        """
        windows:  List[np.ndarray]   = []
        valid_ts: List[pd.Timestamp] = []

        # Strip timezone for lookup
        if self._sw_index.tz is not None:
            self._sw_index     = self._sw_index.tz_localize(None)
            self._index_to_pos = {ts: pos for pos, ts in enumerate(self._sw_index)}
        timestamps = pd.DatetimeIndex(timestamps).tz_localize(None)

        n_not_in_index = n_bad_end_pos = n_lookback_limit = n_insufficient = 0

        for ts in tqdm(timestamps, desc='looping through timestamps'):
            if ts not in self._index_to_pos:
                n_not_in_index += 1
                continue

            end_pos = self._index_to_pos[ts] - self._ampere_delay
            if end_pos < 0:
                n_bad_end_pos += 1
                continue

            collected: List[np.ndarray] = []
            pos = end_pos
            skipped = 0
            while (pos >= 0
                   and len(collected) < self._time_history
                   and skipped <= self._lookback_limit):
                row = self._sw_values[pos, :]
                if not np.isnan(row).any():
                    collected.append(row)
                else:
                    skipped += 1
                pos -= 1

            if len(collected) < self._time_history:
                if skipped > self._lookback_limit:
                    n_lookback_limit += 1
                else:
                    n_insufficient += 1
                continue

            windows.append(np.array(collected[::-1]))
            valid_ts.append(ts)

        if not windows:
            print(
                f"[TF _build_sequences] No sequences built from {len(timestamps)} timestamp(s): "
                f"not_in_index={n_not_in_index}, bad_end_pos={n_bad_end_pos}, "
                f"lookback_limit={n_lookback_limit}, insufficient={n_insufficient}. "
                f"SW range: {self._sw_index[0]} -> {self._sw_index[-1]}. "
                f"Requested: {timestamps[0]} -> {timestamps[-1]}"
            )
            return np.empty(0), []

        sequences = np.stack(windows, axis=0)   # (N, T, 7)
        N, T, F   = sequences.shape

        # Apply JSON normalisation per feature
        for i, col in enumerate(self._INPUT_COLS):
            if col == 'month_sine' or col == 'month_cosine':
                continue
            mean_key = f"{col}_mean"
            std_key  = f"{col}_std"
            if mean_key in self._norm and std_key in self._norm:
                sequences[:, :, i] = (
                    (sequences[:, :, i] - self._norm[mean_key])
                    / self._norm[std_key]
                )
            else:
                # Per-sequence fallback if key missing
                raise KeyError(f"{col} scalers not avaialbe in file")
                # col_mean = sequences[:, :, i].mean()
                # col_std  = sequences[:, :, i].std() or 1.0
                # sequences[:, :, i] = (sequences[:, :, i] - col_mean) / col_std

        print(f"[TF _build_sequences] Built {len(windows)} sequence(s) from "
              f"{len(timestamps)} timestamp(s)")

        return sequences.astype(np.float32), valid_ts

    # def _run_tf_model(self, sequences: np.ndarray) -> np.ndarray:
    #     """
    #     Forward pass through the Keras model.

    #     Input  shape : (N, T, 7)
    #     Output shape : (N, 50, 25)  — reshaped and column-stacked per fac_SW()
    #     """
    #     try:
    #         import tensorflow as tf
    #     except ImportError as e:
    #         raise ImportError(
    #             "tensorflow is required for TFACInference. "
    #             "Install with: pip install tensorflow"
    #         ) from e

    #     results = []
    #     for i in tqdm(range(len(sequences)), desc='running model'):
    #         inp      = np.array(tf.expand_dims(sequences[i].astype(np.float32), axis=0))
    #         raw      = self._model.predict(inp, batch_size=1, verbose=0)
    #         fac      = np.reshape(raw, [50, 24])
    #         fac      = np.flipud(fac)                      # flip colatitude
    #         results.append(fac)

    #     return np.stack(results, axis=0)   # (N, 50, 25)

    def _run_tf_model(self, sequences: np.ndarray) -> np.ndarray:
        """
        Forward pass through the Keras model — batched for performance.
        A small warmup pass is run first to force TF graph compilation
        before the full dataset, avoiding a hang on the first real batch.

        Input  shape : (N, T, 7)
        Output shape : (N, 50, 24)
        """
        try:
            import tensorflow as tf
        except ImportError as e:
            raise ImportError(
                "tensorflow is required for TFACInference. "
                "Install with: pip install tensorflow"
            ) from e

        # ── Warmup: compile graph on a single sample ──────────────────────────
        print('Warming up TF graph...')
        _ = self._model(sequences[:1].astype(np.float32), training=False)
        print('Warmup done.')

        # ── Batched inference ─────────────────────────────────────────────────
        batch_size = 256
        n          = len(sequences)
        results    = []

        for i in tqdm(range(0, n, batch_size), desc='Running BK model'):
            batch = sequences[i:i + batch_size].astype(np.float32)
            # Use model() directly (no retracing) rather than model.predict()
            raw   = self._model(batch, training=False).numpy()
            if raw.ndim == 2:
                raw = raw.reshape(-1, 50, 24)
            fac = np.flip(raw, axis=1)
            results.append(fac)

        return np.concatenate(results, axis=0)   # (N, 50, 24)

    def _load_tf_model(self):
        """Load the Keras .hdf5 model without requiring custom loss functions."""
        try:
            import tensorflow as tf
        except ImportError as e:
            raise ImportError(
                "tensorflow is required for TFACInference. "
                "Install with: pip install tensorflow"
            ) from e

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"TF model not found: {self._model_path}"
            )

        return tf.keras.models.load_model(
            str(self._model_path),
            compile=False,
        )

# ══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════════════════

def testing_polar_plot(samples, time, labels):
    """Quick polar plot for sanity-checking model outputs side by side."""

    theta_ticks = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    rad_ticks   = np.linspace(0, 50, 5, endpoint=False)
    rad_labels  = ['', '80', '70', '60', '50']

    # Shared symmetric colour scale across all panels
    scale     = max(np.max(np.abs(sample)) for sample in samples)
    scale_map = mpl.colors.Normalize(vmin=-scale, vmax=scale)

    fig, axes = plt.subplots(
        ncols=len(samples), nrows=1,
        figsize=(6 * len(samples), 10),
        subplot_kw=dict(projection='polar'),
    )
    if len(samples) == 1:
        axes = [axes]

    ts_str = time.strftime("%Y-%m-%d %H:%M UT") if hasattr(time, "strftime") else str(time)
    plt.suptitle(ts_str, fontsize=20)

    r, th = np.meshgrid(
        np.linspace(0, 50, 50, endpoint=False),
        np.linspace(0, 2 * np.pi, 24, endpoint=False),
    )

    for ax, sample, label in zip(axes, samples, labels):
        ax.set_title(label)
        ax.set_theta_zero_location('S')
        if label == "STD":
            p = ax.pcolormesh(th, r, sample.T, cmap='Purples')
        else:
            c = ax.pcolormesh(th, r, sample.T, cmap='bwr', norm=scale_map)
        ax.set_xticks(theta_ticks)
        ax.set_xticklabels(['', '3', '', '9', '', '15', '', '21'])
        ax.set_yticks(rad_ticks)
        ax.set_yticklabels(rad_labels)
        ax.set_ylim(0, 35)

    fig.colorbar(c, ax=axes.ravel().tolist(), orientation='vertical')
    if "STD" in labels:
        fig.colorbar(p, ax=axes.ravel().tolist(), orientation='horizontal', pad=0.15)
    plt.show()


if __name__ == "__main__":

    # TS = "2023-05-05 00:00:00"
    TS = None
    realtime=True

    # ── ACORN (PyTorch) ───────────────────────────────────────────────────────
    acorn = FACInference("config.json", model_variant="op", realtime=realtime)
    acorn_mean, std, time = acorn.predict(timestamp=TS)
    print(f"ACORN Single mean: {acorn_mean.shape}  std: {std.shape}")

    # mean, std, time = acorn.predict(date="2023-05-06")
    # print(f"[ACORN Full day]   mean: {mean.shape}  std: {std.shape}")

    # mean, std, time = acorn.predict(start="2023-05-06 00:00:00", end="2023-05-06 06:00:00")
    # print(f"[ACORN Date range] mean: {mean.shape}  std: {std.shape}")

    # ── TF model (Keras) ──────────────────────────────────────────────────────
    tf_model = TFACInference("config.json", model_variant="op", realtime=realtime)
    bk_mean, _, time = tf_model.predict(timestamp=TS)
    print(f"TF Single mean: {bk_mean.shape}  std: None")

    # mean, _, time = tf_model.predict(date="2023-05-06")
    # print(f"[TF Full day]   mean: {mean.shape}  std: None")

    # ── AMPERE observed (optional) ────────────────────────────────────────────
    # if realtime:
    ampere=None
    # else:
    #     ampere = acorn.load_ampere(timestamp=TS)
    if ampere is not None:
        print(f"AMPERE shape: {ampere.shape}")
        samples = [acorn_mean, bk_mean, ampere]
        labels  = ["ACORN", "BK", "AMPERE"]
    else:
        print("AMPERE not available for this timestamp")
        samples = [acorn_mean, std, bk_mean]
        labels  = ["ACORN", "STD", "BK"]

    testing_polar_plot(samples=samples, time=time, labels=labels)
    print("Good job, it ran!")
