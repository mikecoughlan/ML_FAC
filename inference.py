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
#   SML / SMU / SME          : SuperMAG web service (requires supermag-api
#                              and a registered userid; set SUPERMAG_USERID).
#                              These match training. OMNI's AU_INDEX /
#                              AL_INDEX are a DIFFERENT index family and are
#                              only used when allow_ae_substitution=True.
#
#       and _run_model once inference is stable.
#
####################################################################################

from __future__ import annotations

import datetime
import json
import os
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


def _fill_and_validate(solarwind: pd.DataFrame,
                       vars_to_keep: List[str],
                       model_name: str = "",
                       gap_limit_minutes: int = 15) -> pd.DataFrame:
    """
    Fill short gaps in the model inputs, and refuse to proceed when an input
    is wholly absent.

    Distinguishes two cases, because they mean different things:

      * A required column that is entirely missing or entirely NaN means the
        data source cannot supply it at all -- e.g. SYM_H/ASY_H have no
        real-time source. Interpolating that is not a gap fill, it is
        invention, so this raises and points at the operational model, which
        is defined precisely to exclude these inputs.

      * A column that is present but patchy has real measurements either side
        of the gap, so gaps up to gap_limit_minutes are interpolated. Leading
        and trailing gaps are back/forward filled under the same limit, since
        there is only one side to interpolate from.

    The limit matters: an unbounded ffill/bfill would propagate a single
    value across hours and present it to the model as measurement.
    """
    present = [c for c in vars_to_keep if c in solarwind.columns]
    absent  = [c for c in vars_to_keep if c not in solarwind.columns]
    empty   = [c for c in present if solarwind[c].isna().all()]

    unavailable = absent + empty
    if unavailable:
        raise RuntimeError(
            f"Required model input(s) {unavailable} are entirely unavailable "
            f"from this data source"
            f"{f' for {model_name}' if model_name else ''}.\n"
            f"These cannot be interpolated -- there are no values to "
            f"interpolate between.\n"
            f"If this is a real-time window, use the operational ('op') model, "
            f"which excludes inputs that have no real-time source."
        )

    # Cadence-aware limit: the frames are 1-min, but derive it rather than
    # assume so a resampled frame does not get a silently wrong window.
    step = solarwind.index.to_series().diff().median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        limit = gap_limit_minutes
    else:
        limit = max(1, int(round(pd.Timedelta(minutes=gap_limit_minutes) / step)))

    before = {c: int(solarwind[c].isna().sum()) for c in present}

    solarwind[present] = (solarwind[present]
                          .interpolate(method="time", limit=limit,
                                       limit_area="inside")
                          .ffill(limit=limit)
                          .bfill(limit=limit))

    filled = {c: before[c] - int(solarwind[c].isna().sum())
              for c in present if before[c]}
    if filled:
        print(f"Gap-filled (<= {gap_limit_minutes} min): "
              + ", ".join(f"{c}: {n}" for c, n in filled.items() if n))

    remaining = {c: int(solarwind[c].isna().sum())
                 for c in present if solarwind[c].isna().any()}
    if remaining:
        print(f"Gaps longer than {gap_limit_minutes} min remain and those rows "
              f"will be dropped: "
              + ", ".join(f"{c}: {n}" for c, n in remaining.items()))

    return solarwind


def _enforce_vx_sign(solarwind: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """
    Force Vx to the OMNI sign convention: negative (anti-sunward flow).

    Training data comes from the OMNI feather, where Vx is already negative,
    and the scaler was fit on that. The NOAA RTSW feed reports proton_speed
    as a positive magnitude, so a sign flip there would place the input on
    the opposite side of the scaler's mean and silently corrupt every
    prediction. Rather than trusting each source's convention, the sign is
    asserted here in one place.
    """
    if "Vx" not in solarwind.columns:
        return solarwind

    vx = pd.to_numeric(solarwind["Vx"], errors="coerce")
    n_pos = int((vx > 0).sum())
    n_val = int(vx.notna().sum())
    if n_val == 0:
        return solarwind

    if n_pos:
        if n_pos == n_val:
            # Wholly positive: a speed magnitude, as the RTSW feed provides.
            solarwind["Vx"] = -vx.abs()
        else:
            # Mixed signs are not a convention difference -- something is
            # wrong upstream. Normalise, but say so rather than hiding it.
            print(f"Warning: Vx{f' ({source})' if source else ''} has "
                  f"{n_pos}/{n_val} positive values (mixed sign). Forcing all "
                  f"to negative to match the OMNI convention used in training, "
                  f"but the input source should be checked.")
            solarwind["Vx"] = -vx.abs()
        solarwind.attrs["vx_sign_corrected"] = True
    else:
        solarwind.attrs["vx_sign_corrected"] = False

    return solarwind


def _fetch_f107_series(startdt: datetime.datetime,
                       enddt: datetime.datetime) -> Optional[pd.Series]:
    """
    Daily F10.7 for [startdt, enddt] from the DRAO fluxtable -- the same
    source and column ('fluxadjflux', 1 AU adjusted) that data_prep.py uses
    for training, daily mean, timestamped at 20:00.

    Returns a date-indexed Series, or None if unavailable. Unlike
    _fetch_f107() this varies with time, so a historical window gets the
    F10.7 that actually applied rather than today's value.
    """
    import requests

    url = ("https://www.spaceweather.gc.ca/solar_flux_data/"
           "daily_flux_values/fluxtable.txt")
    try:
        txt = requests.get(url, timeout=60).text
    except Exception as e:
        print(f"Warning: DRAO fluxtable fetch failed ({e}).")
        return None

    lines = txt.splitlines()
    try:
        header = next(l for l in lines if "fluxdate" in l).split()
    except StopIteration:
        print("Warning: DRAO fluxtable has no recognisable header.")
        return None

    rows = [l.split() for l in lines
            if l.split() and l.split()[0].isdigit() and len(l.split()[0]) == 8]
    if not rows:
        print("Warning: DRAO fluxtable returned no data rows.")
        return None

    df = pd.DataFrame(rows, columns=header[:len(rows[0])])
    if "fluxadjflux" not in df.columns:
        print(f"Warning: DRAO fluxtable has no 'fluxadjflux' column. "
              f"Columns: {list(df.columns)}")
        return None

    df["F107"] = pd.to_numeric(df["fluxadjflux"], errors="coerce")
    daily = df.groupby("fluxdate")["F107"].mean()
    daily.index = (pd.to_datetime(daily.index, format="%Y%m%d")
                   + datetime.timedelta(hours=20))
    daily = daily.sort_index().dropna()

    lo = pd.Timestamp(startdt) - pd.Timedelta(days=2)
    hi = pd.Timestamp(enddt) + pd.Timedelta(days=2)
    window = daily[(daily.index >= lo) & (daily.index <= hi)]
    if window.empty:
        print(f"Warning: DRAO fluxtable has no F10.7 for "
              f"{startdt} -> {enddt}.")
        return None
    return window


def _apply_f107(solarwind: pd.DataFrame,
                startdt: Optional[datetime.datetime] = None,
                enddt: Optional[datetime.datetime] = None) -> pd.DataFrame:
    """
    Populate solarwind['F107'] with a time-varying series where possible.

    Falls back to the live NOAA scalar only when the historical table cannot
    cover the window, and records which was used in .attrs['f107_source'].
    A broadcast scalar over a historical window is wrong -- it feeds today's
    solar activity to a past event -- so the series path is preferred.
    """
    if startdt is None:
        startdt = solarwind.index.min().to_pydatetime()
    if enddt is None:
        enddt = solarwind.index.max().to_pydatetime()

    series = _fetch_f107_series(startdt, enddt)
    if series is not None:
        s = series.reindex(
            series.index.union(solarwind.index)
        ).interpolate("time").reindex(solarwind.index)
        s = s.ffill().bfill()
        if s.notna().all():
            solarwind["F107"] = s
            solarwind.attrs["f107_source"] = "drao_adjusted"
            return solarwind
        print("Warning: DRAO F10.7 did not cover the whole window.")

    scalar = _fetch_f107()
    print(f"Warning: broadcasting a single F10.7 value ({scalar}) across "
          f"{startdt} -> {enddt}. This is only appropriate for a near-real-"
          f"time window; for historical windows it applies present-day solar "
          f"activity to a past event.")
    solarwind["F107"] = scalar
    solarwind.attrs["f107_source"] = "noaa_scalar_broadcast"
    return solarwind


def _fetch_noaa_realtime() -> Optional[pd.DataFrame]:
    """
    Fetch the last 24 hours of real-time solar wind plasma and IMF data from
    NOAA SWPC. Returns a DatetimeIndex DataFrame with columns:
        density, speed, bx_gse, by_gsm, bz_gsm
    Returns None if the request fails.

    Uses the RTSW endpoints that replaced the retired
    products/solar-wind/*.json files. These serve one record per dict and
    interleave several spacecraft (ACE, DSCOVR, SOLAR1, IMAP), so rows must
    be filtered on the 'active' flag -- SWPC switches which spacecraft is
    authoritative, and mixing them would splice discontinuous series.
    """
    try:
        print("Fetching real-time solar wind data from NOAA SWPC...")

        wind_url = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"
        mag_url  = "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"

        wind = pd.DataFrame(json.loads(urlopen(wind_url, timeout=15).read().decode("utf-8")))
        mag  = pd.DataFrame(json.loads(urlopen(mag_url,  timeout=15).read().decode("utf-8")))

        for name, df in (("wind", wind), ("mag", mag)):
            if df.empty:
                print(f"RTSW {name} feed returned no records.")
                return None
            if "active" not in df.columns:
                print(f"RTSW {name} feed has no 'active' column. "
                      f"Columns: {list(df.columns)}")
                return None

        # Keep only the spacecraft SWPC currently designates as active.
        wind = wind[wind["active"].astype(bool)]
        mag  = mag[mag["active"].astype(bool)]
        if wind.empty or mag.empty:
            print("RTSW feeds contain no rows flagged active.")
            return None

        srcs = (sorted(wind["source"].dropna().unique()),
                sorted(mag["source"].dropna().unique()))
        print(f"RTSW active source -- wind: {srcs[0]}, mag: {srcs[1]}")

        wind["Epoch"] = pd.to_datetime(wind["time_tag"])
        mag["Epoch"]  = pd.to_datetime(mag["time_tag"])

        for col in ["proton_density", "proton_speed", "proton_temperature"]:
            wind[col] = pd.to_numeric(wind.get(col), errors="coerce")
        for col in ["bx_gse", "by_gsm", "bz_gsm", "bt"]:
            mag[col] = pd.to_numeric(mag.get(col), errors="coerce")

        wind = (wind.dropna(subset=["Epoch"]).set_index("Epoch").sort_index()
                [["proton_density", "proton_speed"]])
        mag  = (mag.dropna(subset=["Epoch"]).set_index("Epoch").sort_index()
                [["bx_gse", "by_gsm", "bz_gsm"]])
        wind = wind[~wind.index.duplicated(keep="first")]
        mag  = mag[~mag.index.duplicated(keep="first")]

        combined = wind.join(mag, how="inner").rename(columns={
            "proton_density": "density",
            "proton_speed":   "speed",
        })
        combined = combined[["density", "speed", "bx_gse", "by_gsm", "bz_gsm"]]

        if combined.empty:
            print("RTSW wind and mag feeds share no common timestamps.")
            return None

        print(f"Fetched {len(combined)} real-time data points "
              f"({combined.index.min()} -> {combined.index.max()})")
        return combined

    except Exception as e:
        print(f"Error fetching NOAA real-time data: {e}")
        return None


def _load_solarwind_realtime(config: dict, vars_to_keep: Optional[List[str]] = None,
                             supermag_userid: Optional[str] = None) -> pd.DataFrame:
    """
    Build the model input DataFrame from live NOAA SWPC feeds instead of
    Mirrors the column layout of _load_solarwind_omni() so the same
    scaler and sequence-building logic applies.

    SML / SMU / SME are attempted from the SuperMAG web service first. Because
    SuperMAG has its own processing lag, they are frequently unavailable for a
    live window, in which case they are NaN-filled and predictions from the
    sci model are degraded. SYM_H and ASY_H have no real-time source here and
    are always NaN-filled. The source actually used is recorded in
    .attrs['index_source'].
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

    # F10.7 — time-varying where available; see _apply_f107.
    solarwind = _apply_f107(solarwind)

    # Rename NOAA columns to match training feature names
    # The RTSW mag feed carries bx_gse separately from bx_gsm, so BX_GSE is
    # now filled with a true GSE component rather than the GSM one the
    # retired feed forced us to use.
    solarwind = solarwind.rename(columns={
        "bx_gse":  "BX_GSE",
        "by_gsm":  "BY_GSM",
        "bz_gsm":  "BZ_GSM",
        "speed":   "Vx",
        "density": "proton_density",
    })

    # Vx from the RTSW feed is a positive speed magnitude; training used
    # OMNI's negative convention.
    solarwind = _enforce_vx_sign(solarwind, source="NOAA RTSW")

    # SuperMAG indices are unavailable in real time — insert NaN columns so
    # the DataFrame has the right shape, then interpolate what we can
    # Try SuperMAG first. It carries its own processing lag, so for a live
    # window this often returns nothing -- but when it does have data, using
    # it is strictly better than the NaN stub below.
    supermag_cols = ["SML", "SMU", "SYM_H", "ASY_H", "SME"]
    sm = _fetch_supermag_indices(solarwind.index.min().to_pydatetime(),
                                 solarwind.index.max().to_pydatetime(),
                                 userid=supermag_userid)
    if sm is not None:
        sm = sm.reindex(solarwind.index, method="nearest",
                        tolerance=pd.Timedelta("1min"))
        solarwind["SML"] = sm["SML"]
        solarwind["SMU"] = sm["SMU"]
        solarwind["SME"] = sm["SME"]
        solarwind.attrs["index_source"] = "supermag"
        print(f"SuperMAG SML/SMU/SME cover {sm['SML'].notna().mean():.1%} of "
              f"the real-time window. Note this does not include SYM_H or "
              f"ASY_H, which have no real-time source; the sci model needs "
              f"them and will not run on this window.")
    else:
        needs_sm = any(c in vars_to_keep for c in ("SML", "SMU", "SME"))
        if needs_sm:
            print("SuperMAG indices unavailable for the real-time window "
                  "(SuperMAG's processing lag exceeds 24 h). This model "
                  "requires SML/SMU, so it cannot run on a live window -- "
                  "use the operational ('op') model, which excludes them.")
        else:
            print("SuperMAG indices unavailable for the real-time window "
                  "(processing lag exceeds 24 h). This model does not use "
                  "them, so it is unaffected.")
        solarwind.attrs["index_source"] = "unavailable"

    for col in supermag_cols:
        if col not in solarwind.columns:
            solarwind[col] = np.nan

    # Fill short gaps; raise if an input has no real-time source at all.
    solarwind = _fill_and_validate(
        solarwind, vars_to_keep,
        model_name=config.get("version", ""))

    solarwind = solarwind[vars_to_keep]
    solarwind.dropna(inplace=True)

    if solarwind.empty:
        raise RuntimeError(
            "No complete rows remain after gap filling; every timestamp is "
            "missing at least one model input.")

    return solarwind


# ══════════════════════════════════════════════════════════════════════════════
# SuperMAG indices  (SML / SMU / SME)
#
# The model is trained on SuperMAG indices, not on the AE-ring AU/AL indices
# carried by OMNI. These are different measurements, so the two are not
# interchangeable and a substitution is recorded rather than assumed.
# ══════════════════════════════════════════════════════════════════════════════

# Registered SuperMAG web-service user. Override with the SUPERMAG_USERID
# environment variable, or by passing supermag_userid= to the loader.
SUPERMAG_USERID = os.environ.get("SUPERMAG_USERID", "acorn_user")

# Flag string for SuperMAGGetIndices. 'indicesall' returns the full index set;
# 'all' additionally returns solar wind columns that are not needed here.
SUPERMAG_FLAGS = "indicesall"


def _fetch_supermag_indices(startdt: datetime.datetime,
                            enddt: datetime.datetime,
                            userid: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Fetch SML / SMU / SME from the SuperMAG web service for [startdt, enddt].

    Returns a 1-min DataFrame indexed by UTC datetime with columns
    SML, SMU, SME, or None if the indices could not be retrieved (package
    missing, request failed, or the window falls outside SuperMAG coverage).

    Returning None rather than raising lets the caller decide whether an
    AE-index substitution is acceptable; it never silently substitutes here.
    """
    userid = userid or SUPERMAG_USERID
    if not userid:
        print("No SuperMAG userid configured; cannot fetch SML/SMU.")
        return None

    # The PyPI package ships as a directory with an empty __init__.py, so the
    # callable lives in the supermag_api.supermag_api submodule rather than at
    # the top level. Older/flat installs expose it directly; try both, then
    # fall back to scanning submodules so a layout change does not silently
    # disable SuperMAG and push everything onto the AE substitution.
    SuperMAGGetIndices = None
    try:
        import supermag_api as _sm
    except ImportError:
        print("supermag_api not installed (pip install supermag-api); "
              "cannot fetch SML/SMU.")
        return None

    SuperMAGGetIndices = getattr(_sm, "SuperMAGGetIndices", None)
    if SuperMAGGetIndices is None:
        import importlib
        try:
            _sub = importlib.import_module("supermag_api.supermag_api")
            SuperMAGGetIndices = getattr(_sub, "SuperMAGGetIndices", None)
        except ImportError:
            pass
    if SuperMAGGetIndices is None and hasattr(_sm, "__path__"):
        import importlib
        import pkgutil
        for _m in pkgutil.iter_modules(_sm.__path__):
            try:
                _sub = importlib.import_module(f"supermag_api.{_m.name}")
            except Exception:
                continue
            SuperMAGGetIndices = getattr(_sub, "SuperMAGGetIndices", None)
            if SuperMAGGetIndices is not None:
                break
    if SuperMAGGetIndices is None:
        print("supermag_api is installed but SuperMAGGetIndices was not found "
              "in it or its submodules; check the package version.")
        return None

    start  = [startdt.year, startdt.month, startdt.day,
              startdt.hour, startdt.minute, getattr(startdt, "second", 0)]
    extent = int((enddt - startdt).total_seconds())
    if extent <= 0:
        return None

    print(f"Fetching SuperMAG indices SML, SMU, SME "
          f"(userid={userid}, flags={SUPERMAG_FLAGS!r}): "
          f"{startdt:%Y-%m-%d %H:%M} -> {enddt:%Y-%m-%d %H:%M} "
          f"({extent}s, {extent // 60} min expected)")

    try:
        status, sm_indices = SuperMAGGetIndices(
            userid, start, extent, SUPERMAG_FLAGS, FORMAT="list")
    except Exception as e:
        print(f"SuperMAG request failed ({e}); SML/SMU unavailable.")
        return None

    # The client returns status 1 on success. Anything else is a failed or
    # empty query -- including a window newer than SuperMAG has processed.
    if status != 1 or sm_indices is None or len(sm_indices) == 0:
        print(f"SuperMAG returned status {status} with "
              f"{0 if sm_indices is None else len(sm_indices)} records; "
              f"SML/SMU unavailable for {startdt} -> {enddt}.")
        return None

    # The response carries ~65 keys per record, several of which are lists
    # (SMEr, SMLr, SMLrstid ... the per-MLT-sector breakdowns). Building a
    # DataFrame from all of them gives ragged object columns, so pull only
    # the scalar keys we need.
    wanted = ("SML", "SMU", "SME")
    missing = [k for k in ("tval",) + wanted if k not in sm_indices[0]]
    if missing:
        print(f"SuperMAG response is missing {missing}. "
              f"Keys: {list(sm_indices[0].keys())[:10]}...")
        return None

    out = pd.DataFrame({
        k: pd.to_numeric([rec.get(k) for rec in sm_indices], errors="coerce")
        for k in wanted
    }, index=pd.to_datetime([rec["tval"] for rec in sm_indices], unit="s"))

    # SuperMAG marks missing data with a sentinel (999999) rather than null,
    # so unmasked it reads as a valid measurement: coverage looks complete,
    # gap filling is skipped, and the sentinel reaches the scaler as though it
    # were a real index value. Mask on magnitude rather than equality, since
    # variants (999999.0, 9999999) appear across products.
    #
    # The physical ranges are nowhere near the sentinel: SML/SMU rarely exceed
    # a few thousand nT even in severe storms, so 100000 separates them
    # cleanly without risking a real extreme value.
    n_before = out.notna().sum()
    out = out.mask(out.abs() >= 100000)
    n_masked = n_before - out.notna().sum()
    if n_masked.any():
        print("Masked SuperMAG fill values: "
              + ", ".join(f"{k}: {int(v)}" for k, v in n_masked.items() if v))

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]

    valid = out[["SML", "SMU"]].notna().all(axis=1).mean()
    print(f"SuperMAG returned {len(out)} records "
          f"({out.index.min()} -> {out.index.max()}); "
          f"{valid:.1%} have valid SML and SMU after fill masking "
          f"(response carried {len(sm_indices[0])} keys per record)")

    if out[["SML", "SMU"]].isna().all().any():
        print("SuperMAG returned SML/SMU columns that are entirely NaN.")
        return None

    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)

    return out


def _promote_supermag_indices(solarwind: pd.DataFrame,
                              startdt: datetime.datetime,
                              enddt: datetime.datetime,
                              allow_ae_substitution: bool = False,
                              supermag_userid: Optional[str] = None) -> pd.DataFrame:
    """
    Populate SML / SMU / SME on `solarwind`, preferring SuperMAG.

    If SuperMAG is unavailable and allow_ae_substitution is True, OMNI's
    AL_INDEX / AU_INDEX are used instead and the frame is tagged
    solarwind.attrs['index_source'] = 'ae_substituted'. If it is False
    (the default) the call raises rather than quietly degrading.

    The AE substitution is a real change of quantity, not a rename -- see the
    note on _OMNI_COL_MAP.
    """
    sm = _fetch_supermag_indices(startdt, enddt, userid=supermag_userid)

    if sm is not None:
        sm = sm.reindex(solarwind.index, method="nearest",
                        tolerance=pd.Timedelta("1min"))
        coverage = sm["SML"].notna().mean()
        solarwind["SML"] = sm["SML"]
        solarwind["SMU"] = sm["SMU"]
        solarwind["SME"] = sm["SME"]
        solarwind.attrs["index_source"] = "supermag"
        solarwind.attrs["index_coverage"] = float(coverage)
        if coverage < 0.99:
            print(f"Note: SuperMAG SML/SMU/SME cover {coverage:.1%} of the "
                  f"requested timestamps; the remainder is NaN.")
        return solarwind

    have_ae = {"AL_INDEX", "AU_INDEX"}.issubset(solarwind.columns)
    if not allow_ae_substitution:
        raise RuntimeError(
            "SML/SMU could not be retrieved from SuperMAG, and AE substitution "
            "is disabled.\n"
            "  - Check SUPERMAG_USERID and that supermag-api is installed.\n"
            "  - SuperMAG has its own processing lag; very recent windows may "
            "not be available yet.\n"
            "  - To accept degraded indices, pass allow_ae_substitution=True. "
            "AL/AU are a different index family and will bias predictions, "
            "most strongly during active periods."
        )

    if not have_ae:
        raise RuntimeError(
            "SML/SMU unavailable from SuperMAG and OMNI AL_INDEX/AU_INDEX are "
            "also absent; cannot populate the SuperMAG inputs."
        )

    print("WARNING: substituting OMNI AL_INDEX/AU_INDEX for SML/SMU. "
          "These come from a 12-station ring rather than SuperMAG's ~100+ "
          "station network, are systematically smaller in magnitude, and "
          "diverge most during active periods. Predictions will be biased "
          "relative to the training distribution.")
    solarwind["SML"] = solarwind["AL_INDEX"]
    solarwind["SMU"] = solarwind["AU_INDEX"]
    solarwind["SME"] = solarwind["SMU"] - solarwind["SML"]
    solarwind.attrs["index_source"] = "ae_substituted"
    solarwind.attrs["index_coverage"] = float(solarwind["SML"].notna().mean())
    return solarwind


# OMNI variable names -> training feature names.
#
# AU_INDEX / AL_INDEX are deliberately NOT renamed to SMU / SML. They are a
# different index family: AU/AL come from the 12-station AE observatory ring,
# while SMU/SML are envelopes over SuperMAG's ~100+ station network. SuperMAG
# resolves electrojet excursions the sparser ring misses, so |SML| generally
# exceeds |AL|, and the gap widens with activity -- exactly the disturbed
# conditions this model targets. Training (data_prep.py) reads true SMU/SML
# from the SuperMAG feather file, so silently substituting AU/AL here would
# feed the model a systematically different quantity than it learned on.
#
# They are carried through under their own names so that
# _promote_supermag_indices() can decide explicitly whether to use them.
_OMNI_COL_MAP = {
    "Vx":             "Vx",
    "BX_GSE":         "BX_GSE",
    "BY_GSM":         "BY_GSM",
    "BZ_GSM":         "BZ_GSM",
    "proton_density": "proton_density",
    "SYM_H":          "SYM_H",
    "ASY_H":          "ASY_H",
    "AU_INDEX":       "AU_INDEX",
    "AL_INDEX":       "AL_INDEX",
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


def _fetch_omni_cdf(dt: datetime.datetime,
                    refresh: bool = False,
                    check_version: bool = True) -> Path:
    """
    Return a local path to the monthly 1-min OMNI CDF for the month containing
    dt, downloading it if needed.

    Parameters
    ----------
    refresh : bool
        Ignore any cached copy and re-download.
    check_version : bool
        Ask SPDF which version is current and re-download if the cached copy
        is superseded. OMNI months are revised after first publication (v01 ->
        v02 and beyond) as calibrations are finalised, so a cache keyed only
        on year/month will keep serving provisional data indefinitely. The
        check costs one directory listing per month per session.

        Recent months are the ones most likely to be revised, and also the
        ones most likely to be in a prediction window, so this defaults on.
        Pass check_version=False for offline or bulk work.
    """
    import requests as _requests

    _OMNI_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    prefix   = f"omni_hro_1min_{dt.year}{dt.month:02d}"
    existing = sorted(_OMNI_CACHE_DIR.glob(f"{prefix}*.cdf"))

    if existing and not refresh:
        # Highest version on disk, not merely the first glob hit.
        cached = existing[-1]
        if not check_version:
            return cached
        try:
            remote_fn = _resolve_omni_cdf_filename(dt)
        except Exception as e:
            # Offline or listing unavailable: a cached file beats no file.
            print(f"Could not check OMNI version for {dt:%Y-%m} ({e}); "
                  f"using cached {cached.name}.")
            return cached

        if remote_fn == cached.name:
            return cached

        # Superseded: fetch the newer version but keep the old one on disk.
        # Comparing versions is how a revision that moves results gets found,
        # and that is impossible if the previous file has been deleted. Old
        # versions are never read (the highest is always selected) so they
        # cost only disk space.
        print(f"OMNI {dt:%Y-%m}: cached {cached.name} superseded by "
              f"{remote_fn}; downloading the newer version and keeping "
              f"{cached.name} for comparison.")
        fn = remote_fn
    else:
        if refresh and existing:
            print(f"OMNI {dt:%Y-%m}: refresh requested; re-resolving current "
                  f"version. {len(existing)} existing file(s) are kept.")
        fn = _resolve_omni_cdf_filename(dt)

    local = _OMNI_CACHE_DIR / fn
    url   = f"{_OMNI_BASE_URL}/{dt.year}/{fn}"

    print(f"Downloading OMNI CDF: {url}")
    r = _requests.get(url, timeout=120)
    r.raise_for_status()

    # Write to a temporary name first so an interrupted download cannot leave
    # a truncated file that later looks like a valid cache hit.
    tmp = local.with_suffix(local.suffix + ".part")
    tmp.write_bytes(r.content)
    tmp.replace(local)
    print(f"Saved to {local}")

    return local


def list_omni_cache() -> pd.DataFrame:
    """
    List cached OMNI CDFs with their sizes, flagging months held at more than
    one version. Superseded versions are retained rather than deleted, so this
    is how to find them.
    """
    if not _OMNI_CACHE_DIR.exists():
        return pd.DataFrame(columns=["file", "month", "version", "mb"])

    rows = []
    for f in sorted(_OMNI_CACHE_DIR.glob("omni_hro_1min_*.cdf")):
        stem = f.stem.replace("omni_hro_1min_", "")
        parts = stem.split("_")
        rows.append({"file": f.name,
                     "month": parts[0][:6] if parts else "",
                     "version": parts[-1] if len(parts) > 1 else "",
                     "mb": round(f.stat().st_size / 1e6, 1)})
    df = pd.DataFrame(rows)
    if not df.empty:
        dupes = df["month"].duplicated(keep=False)
        if dupes.any():
            print("Months held at multiple versions:")
            print(df[dupes].to_string(index=False))
    return df


def clear_omni_cache(year: Optional[int] = None,
                     month: Optional[int] = None) -> int:
    """
    Delete cached OMNI CDFs and return the number removed.

    Nothing else in this module deletes cached files -- superseded versions
    are kept so they can be compared against their replacements. This is the
    only way to remove them, and it is deliberate rather than automatic.

    With no arguments, clears the whole cache. Pass year (and optionally
    month) to clear a subset.
    """
    if not _OMNI_CACHE_DIR.exists():
        return 0
    if year is None:
        pattern = "omni_hro_1min_*.cdf"
    elif month is None:
        pattern = f"omni_hro_1min_{year}*.cdf"
    else:
        pattern = f"omni_hro_1min_{year}{month:02d}*.cdf"

    n = 0
    for f in _OMNI_CACHE_DIR.glob(pattern):
        try:
            f.unlink()
            n += 1
        except OSError as e:
            print(f"Could not remove {f.name}: {e}")
    print(f"Removed {n} cached OMNI file(s) matching {pattern}")
    return n


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
                         vars_to_keep: Optional[List[str]] = None,
                         allow_ae_substitution: bool = False,
                         supermag_userid: Optional[str] = None,
                         refresh_cache: bool = False,
                         check_cdf_version: bool = True) -> pd.DataFrame:
    """
    Fetch OMNI 1-min solar wind data for [startdt, enddt] directly from NASA
    SPDF, with local CDF caching (~/.cache/omni_cdfs/). No third-party
    packages beyond cdflib (already required by processing_omni.py) are needed.

    SML / SMU / SME come from the SuperMAG web service, matching training.
    OMNI's AU_INDEX / AL_INDEX are a different index family and are used only
    when allow_ae_substitution=True; the source is recorded in
    the returned frame's .attrs['index_source'].

    Monthly CDFs are cached under ~/.cache/omni_cdfs/. By default the cache is
    version-checked against SPDF, since OMNI months are revised after first
    publication; pass check_cdf_version=False to skip that (offline use), or
    refresh_cache=True to force a re-download. Superseded versions are kept on
    disk -- see list_omni_cache() -- and only the highest version is read.

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

    # OMNI is always attempted first, whatever the window's age.
    #
    # A fixed lag cutoff was previously used to decide this in advance, but a
    # hardcoded guess is systematically wrong: OMNI's actual publication lag
    # moves, and a too-conservative value routes windows to the NOAA live feed
    # that OMNI could in fact serve. That silently costs SYM_H / ASY_H, which
    # have no real-time source, and so silently downgrades the sci model.
    # Asking the archive is cheap and always correct; guessing is not.
    frames = []
    errors = []
    for month_dt in months:
        try:
            cdf_path = _fetch_omni_cdf(month_dt,
                                       refresh=refresh_cache,
                                       check_version=check_cdf_version)
            df       = _read_omni_cdf(cdf_path, startdt, enddt)
            if df is not None and not df.empty:
                frames.append(df)
            else:
                errors.append(f"  {month_dt.strftime('%Y-%m')}: no rows in range")
        except Exception as e:
            errors.append(f"  {month_dt.strftime('%Y-%m')}: {e}")

    if not frames:
        # Genuinely outside the archive (or the fetch failed). Now -- and only
        # now -- fall back to the live feed.
        print(f"OMNI returned no data for {startdt:%Y-%m-%d} -> {enddt:%Y-%m-%d}:")
        print("\n".join(errors))
        print("Falling back to the NOAA real-time feed. Note this feed covers "
              "only the last ~24 h and carries no SYM_H / ASY_H, so the sci "
              "model will not run on it.")
        return _load_solarwind_realtime(
            config, vars_to_keep=vars_to_keep, supermag_userid=supermag_userid)

    if errors:
        print("Warning: some months could not be fetched:\n" + "\n".join(errors))

    solarwind = pd.concat(frames).sort_index()
    solarwind = solarwind[~solarwind.index.duplicated(keep="first")]

    # Report how much of the request OMNI actually covered. A window that
    # straddles the publication boundary returns partial data rather than
    # failing, and silently predicting on a truncated window is worse than
    # knowing it was truncated.
    req_start, req_end = pd.Timestamp(startdt), pd.Timestamp(enddt)
    got_start, got_end = solarwind.index.min(), solarwind.index.max()
    expected = max(int((req_end - req_start).total_seconds() // 60), 1)
    print(f"OMNI covered {got_start} -> {got_end} "
          f"({len(solarwind)}/{expected} minutes of the request).")
    if got_end < req_end - pd.Timedelta(hours=1):
        print(f"Note: OMNI stops {req_end - got_end} short of the requested "
              f"end. This is the publication lag; the archive has not caught "
              f"up to that date yet.")

    # SML / SMU / SME: fetched from SuperMAG, which is what the model was
    # trained on. Falls back to OMNI's AE-ring AL/AU only if explicitly
    # permitted, and records which source was used in solarwind.attrs.
    solarwind = _promote_supermag_indices(
        solarwind, startdt, enddt,
        allow_ae_substitution=allow_ae_substitution,
        supermag_userid=supermag_userid,
    )

    # OMNI Vx is already negative; assert it rather than assume it.
    solarwind = _enforce_vx_sign(solarwind, source="OMNI")

    # Cyclical month encoding
    months_col = solarwind.index.month
    solarwind["month"]     = months_col
    solarwind["sin_month"] = np.sin(months_col * 2 * np.pi / 12)
    solarwind["cos_month"] = np.cos(months_col * 2 * np.pi / 12)

    # F10.7 — must vary across a historical window; see _apply_f107.
    solarwind = _apply_f107(solarwind, startdt, enddt)

    # Fill short gaps; raise if an input is wholly unavailable.
    solarwind = _fill_and_validate(
        solarwind, vars_to_keep,
        model_name=config.get("version", ""))

    solarwind = solarwind[vars_to_keep]
    solarwind.dropna(inplace=True)

    if solarwind.empty:
        raise RuntimeError(
            "No complete rows remain after gap filling; every timestamp is "
            "missing at least one model input.")

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
        allow_ae_substitution: bool    = False,
        supermag_userid: Optional[str] = None,
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

        # SML/SMU source policy. See _promote_supermag_indices: AE-ring AL/AU
        # are a different index family from SuperMAG's SML/SMU, so falling
        # back to them is opt-in rather than automatic.
        self._allow_ae_substitution = allow_ae_substitution
        self._supermag_userid = supermag_userid

        # SHAP climatology background, built lazily and reused across calls.
        self._clim_background = None
        self._config_path = config_path

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
            solarwind = _load_solarwind_realtime(
                self.config, supermag_userid=self._supermag_userid)
        else:
            if startdt is None or enddt is None:
                raise ValueError(
                    "startdt and enddt must be provided for OMNI historical fetch."
                )
            print(f"Fetching historical OMNI data: {startdt} -> {enddt}")
            solarwind = _load_solarwind_omni(
                self.config, startdt, enddt,
                allow_ae_substitution=self._allow_ae_substitution,
                supermag_userid=self._supermag_userid)

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

    def explain(
        self,
        timestamp=None,
        target: str = "overall",
        mlat_range=None,
        mlt_range=None,
        channel: int = 0,
        baseline: str = "climatology",
        n_background: int = 200,
        background=None,
        absolute: bool = True,
        seed: int = 0,
    ) -> dict:
        """
        SHAP attribution for a prediction: which driver, at which lag, moved
        the regional mean |FAC|.

        Parameters
        ----------
        timestamp : str, optional
            Timestamp to explain. Defaults to the latest available.
        target : str
            'overall', a SHAP_REGIONS label such as 'R1 Dusk', or 'custom'
            with mlat_range / mlt_range.
        mlat_range, mlt_range : tuple, optional
            (low, high) degrees and (start, end) hours for a custom region.
            MLT may wrap through midnight, e.g. (21, 2).
        channel : int
            0 for the mean field, 1 for the predicted std.
        baseline : str
            'climatology' (default) samples the training record, stratified
            over activity, so values read as "what makes this moment unusual
            relative to the conditions the model learned". Stable, and
            comparable between predictions.
            'window' samples the currently fetched period instead, answering
            "what makes this moment unusual for today" -- cheaper, but the
            reference moves every run so values cannot be compared across
            events.
        n_background : int
            Reference samples. SHAP values are differences from this baseline,
            so the baseline defines what the numbers mean.
        background : np.ndarray, optional
            Explicit background of shape (M, T, F), already scaled. Overrides
            `baseline`.
        absolute : bool
            Take |FAC| before the regional mean. Keep True for multi-sheet
            regions, where signed averaging cancels R1 against R2.

        Returns
        -------
        dict with keys:
            shap        (T, F) SHAP values, lag x parameter
            param_total (F,)   sum of |shap| over lag, per parameter
            param_pct   (F,)   the same as a percentage
            lag_total   (T,)   sum of |shap| over parameters, per lag
            params, timestamp, label, rows, cols, base_value, prediction
        """
        rows, cols, label = resolve_shap_target(target, mlat_range, mlt_range)
        ctx = self._shap_context(timestamp=timestamp, baseline=baseline,
                                 n_background=n_background,
                                 background=background, seed=seed)
        return self._explain_one(ctx, rows, cols, label,
                                 channel=channel, absolute=absolute)

    def _shap_context(self, timestamp=None, baseline="climatology",
                      n_background=200, background=None, seed=0) -> dict:
        """
        Fetch the data and reference set for SHAP, once.

        Separated from the attribution itself so that explaining several
        regions uses one fetch. Repeating the fetch per region is not only
        slower: a long loop can straddle a data update, and regions explained
        either side of it would be attributed against different inputs while
        appearing to be one coherent set.
        """
        try:
            import shap as _shap
        except ImportError:
            raise ImportError("shap is required: pip install shap")

        mean, std, time, sequences = self.predict(timestamp=timestamp)
        if sequences.ndim == 2:
            sequences = sequences[np.newaxis, ...]
        x = sequences[-1:]                                   # (1, T, F)

        if background is None:
            if baseline == "climatology":
                # Cached on the instance: rebuilding per call would reload the
                # whole training record, and a background that shifted between
                # calls would make successive explanations incomparable.
                if self._clim_background is None:
                    self._clim_background = build_climatology_background(
                        config_path=str(self._config_path),
                        model_variant=self._model_variant,
                        n_samples=n_background, seed=seed,
                        scaler=self._scaler)
                background = self._clim_background
                baseline_desc = "training climatology"
            elif baseline == "window":
                pool = self._all_sequences(exclude_last=True)
                if pool is None or len(pool) == 0:
                    print("Only one sequence available; using a zero "
                          "(training-mean) baseline.")
                    background = np.zeros_like(x)
                else:
                    rng = np.random.default_rng(seed)
                    take = rng.choice(len(pool),
                                      size=min(n_background, len(pool)),
                                      replace=False)
                    background = pool[take]
                baseline_desc = "recent window (NOT comparable across runs)"
            else:
                raise ValueError(
                    f"baseline must be 'climatology' or 'window', got {baseline!r}")
        else:
            baseline_desc = "caller-supplied"
        background = np.asarray(background, dtype=np.float32)

        print(f"SHAP context ready: {len(background)} background samples "
              f"({baseline_desc}), timestamp {time}")

        try:
            phys = self._scaler.inverse_transform(x[0])
        except Exception:
            phys = None

        return {"x": x, "background": background, "time": time,
                "baseline_desc": baseline_desc, "physical": phys,
                "mean": mean, "std": std, "shap_mod": _shap}

    def _explain_one(self, ctx, rows, cols, label, channel=0, absolute=True):
        """Attribute one region using an already-built context."""
        _shap = ctx["shap_mod"]
        x, background = ctx["x"], ctx["background"]

        print(f"  SHAP: {label!r}  {len(rows)}x{len(cols)} cells")

        wrapper = _RegionalMeanWrapper(self._model, rows, cols,
                                       channel=channel, absolute=absolute)
        wrapper.eval().to(DEVICE)

        bg_t = torch.tensor(background, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        x_t  = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(DEVICE)

        # GradientExplainer construction runs the model; no_grad keeps it from
        # holding a graph over the whole background set.
        with torch.no_grad():
            explainer = _shap.GradientExplainer(wrapper, bg_t)
        raw = explainer.shap_values(x_t)

        vals = raw[0] if isinstance(raw, list) else raw
        vals = np.asarray(vals)
        vals = np.squeeze(vals)                      # -> (T, F)
        if vals.ndim != 2:
            vals = vals.reshape(x.shape[1], x.shape[2])

        with torch.no_grad():
            pred_val = float(wrapper(x_t).cpu().numpy().ravel()[0])
            base_val = float(wrapper(bg_t).cpu().numpy().mean())

        params = list(self.config["input_params"])
        absum  = np.abs(vals).sum(axis=0)
        total  = absum.sum()
        signed = vals.sum(axis=0)

        # Peak-contributing lag per parameter, and the sign there. Useful for
        # reading response time: Bz peaking at 20-40 min is the reconnection
        # delay, while F107 peaking anywhere is an artefact of it being
        # constant within a day.
        peak_lag = np.abs(vals).argmax(axis=0)
        peak_val = vals[peak_lag, np.arange(vals.shape[1])]

        return {
            "shap":         vals,
            "params":       params,
            "param_total":  absum,
            "param_pct":    100.0 * absum / total if total > 0 else absum * 0,
            "param_signed": signed,
            # Share of the net effect, ranking on the signed sum. A driver
            # whose contributions cancel across the lookback did not move the
            # prediction, however large its individual values were, so this is
            # the more meaningful importance measure for most purposes;
            # param_pct is retained for magnitude questions.
            "signed_pct":   (100.0 * np.abs(signed) / np.abs(signed).sum()
                             if np.abs(signed).sum() > 0 else signed * 0),
            "cancellation": np.divide(signed, absum,
                                      out=np.zeros_like(signed),
                                      where=absum > 0),
            "peak_lag":     peak_lag,
            "peak_value":   peak_val,
            "physical":     ctx["physical"],
            "lag_total":    np.abs(vals).sum(axis=1),
            "lag_signed":   vals.sum(axis=1),
            "timestamp":   ctx["time"],
            "label":       label,
            "channel":     channel,
            "rows":        rows,
            "cols":        cols,
            "base_value":  base_val,
            "prediction":  pred_val,
            "baseline":    ctx["baseline_desc"],
        }

    def explain_regions(self, regions=None, timestamp=None, channel=0,
                        baseline="climatology", n_background=200,
                        background=None, absolute=True, seed=0) -> dict:
        """
        Attribute every region in one pass, sharing a single data fetch and a
        single background set.

        Returns {label: explain_result}. Because all regions are attributed
        against identical inputs and an identical reference, the results are
        directly comparable with each other -- which is not guaranteed when
        explain() is called in a loop, since each call refetches.
        """
        regions = regions if regions is not None else SHAP_REGIONS
        ctx = self._shap_context(timestamp=timestamp, baseline=baseline,
                                 n_background=n_background,
                                 background=background, seed=seed)
        out = {}
        for reg in regions:
            rows = mlat_to_indices(reg["mlat_low"], reg["mlat_high"])
            cols = mlt_to_indices(reg["mlt_start"], reg["mlt_end"])
            out[reg["label"]] = self._explain_one(
                ctx, rows, cols, reg["label"],
                channel=channel, absolute=absolute)
        return out

    def _all_sequences(self, exclude_last: bool = True):
        """
        Every sequence buildable from the currently loaded window, for use as
        a SHAP background.
        """
        T = self._time_history
        n = len(self._sw_values)
        if n <= T:
            return None
        stack = np.stack([self._sw_values[i - T:i] for i in range(T, n)], axis=0)
        N, TT, F = stack.shape
        scaled = self._scaler.transform(stack.reshape(N * TT, F)).reshape(N, TT, F)
        return scaled[:-1] if exclude_last and len(scaled) > 1 else scaled


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
        allow_ae_substitution: bool   = False,
        supermag_userid: Optional[str] = None,
    ):

        # Same merged config as the PyTorch wrapper; model_type selects
        # which block ('sci' or 'op'), rather than naming a separate file.
        self.config = utils.load_config(model_type, config_path)

        self._time_history  = self.config.get("time_history", 60)
        self._ampere_delay  = self.config.get("ampere_delay", 0)
        self._lookback_limit = lookback_limit

        # See FACInference: AE-ring AL/AU are not SuperMAG SML/SMU.
        self._allow_ae_substitution = allow_ae_substitution
        self._supermag_userid = supermag_userid
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
            solarwind = _load_solarwind_realtime(
                self.config, vars_to_keep=input_cols,
                supermag_userid=self._supermag_userid)
        else:
            if startdt is None or enddt is None:
                raise ValueError("startdt and enddt required for OMNI fetch.")
            print(f"Fetching OMNI data for TF model: {startdt} -> {enddt}")
            solarwind = _load_solarwind_omni(
                self.config, startdt, enddt, vars_to_keep=input_cols,
                allow_ae_substitution=self._allow_ae_substitution,
                supermag_userid=self._supermag_userid,
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

# ══════════════════════════════════════════════════════════════════════════════
# SHAP attribution
#
# Explains which solar wind drivers, at which lag, produced a given prediction.
# The attribution target is selectable, because "what drove this prediction" is
# not one question: the drivers of total activity, of a named current system,
# and of one patch of sky are different quantities.
# ══════════════════════════════════════════════════════════════════════════════

# Grid geometry, matching data_prep / shap_values.py.
# MLAT runs 40-90 deg over 50 bins with index 0 at the pole (90 deg).
_MLAT_MIN, _MLAT_MAX = 40.0, 90.0
_N_MLAT, _N_MLT      = 50, 24

# The regions used for the regional HSS / NRMSE evaluation, so SHAP results
# can be read against those metrics directly.
SHAP_REGIONS = [
    {'mlat_low': 80.0, 'mlat_high': 90.0, 'mlt_start':  9, 'mlt_end': 14, 'label': 'R0 Dayside'},
    {'mlat_low': 80.0, 'mlat_high': 90.0, 'mlt_start': 15, 'mlt_end': 20, 'label': 'R0 Dusk'},
    {'mlat_low': 80.0, 'mlat_high': 90.0, 'mlt_start': 21, 'mlt_end':  2, 'label': 'R0 Nightside'},
    {'mlat_low': 80.0, 'mlat_high': 90.0, 'mlt_start':  3, 'mlt_end':  8, 'label': 'R0 Dawn'},
    {'mlat_low': 70.0, 'mlat_high': 79.0, 'mlt_start':  9, 'mlt_end': 14, 'label': 'R1 Dayside'},
    {'mlat_low': 70.0, 'mlat_high': 79.0, 'mlt_start': 15, 'mlt_end': 20, 'label': 'R1 Dusk'},
    {'mlat_low': 70.0, 'mlat_high': 79.0, 'mlt_start': 21, 'mlt_end':  2, 'label': 'R1 Nightside'},
    {'mlat_low': 70.0, 'mlat_high': 79.0, 'mlt_start':  3, 'mlt_end':  8, 'label': 'R1 Dawn'},
    {'mlat_low': 50.0, 'mlat_high': 69.0, 'mlt_start':  9, 'mlt_end': 14, 'label': 'R2 Dayside'},
    {'mlat_low': 50.0, 'mlat_high': 69.0, 'mlt_start': 15, 'mlt_end': 20, 'label': 'R2 Dusk'},
    {'mlat_low': 50.0, 'mlat_high': 69.0, 'mlt_start': 21, 'mlt_end':  2, 'label': 'R2 Nightside'},
    {'mlat_low': 50.0, 'mlat_high': 69.0, 'mlt_start':  3, 'mlt_end':  8, 'label': 'R2 Dawn'},
]


def mlat_to_indices(mlat_low: float, mlat_high: float) -> List[int]:
    """Row indices covering [mlat_low, mlat_high]. Index 0 is the pole."""
    bw       = (_MLAT_MAX - _MLAT_MIN) / _N_MLAT
    idx_pole = max(int((_MLAT_MAX - mlat_high) / bw), 0)
    idx_eq   = min(int((_MLAT_MAX - mlat_low) / bw), _N_MLAT - 1)
    return list(range(idx_pole, idx_eq + 1))


def mlt_to_indices(mlt_start: int, mlt_end: int) -> List[int]:
    """Column indices from mlt_start to mlt_end, wrapping through midnight."""
    s, e = int(mlt_start) % _N_MLT, int(mlt_end) % _N_MLT
    if s > e:
        return list(range(s, _N_MLT)) + list(range(0, e + 1))
    return list(range(s, e + 1))


def resolve_shap_target(target="overall", mlat_range=None, mlt_range=None):
    """
    Resolve an attribution target into (row_indices, col_indices, label).

    target : str
        'overall'                -> the whole grid
        a region label           -> one of SHAP_REGIONS, e.g. 'R1 Dusk'
        'custom'                 -> use mlat_range and mlt_range
    mlat_range : (low, high) in degrees, e.g. (70, 79)
    mlt_range  : (start, end) in hours, wrapping allowed, e.g. (21, 2)
    """
    if target == "overall":
        return list(range(_N_MLAT)), list(range(_N_MLT)), "Overall"

    if target == "custom" or (mlat_range is not None and mlt_range is not None):
        if mlat_range is None or mlt_range is None:
            raise ValueError("custom target needs both mlat_range and mlt_range")
        rows = mlat_to_indices(*mlat_range)
        cols = mlt_to_indices(*mlt_range)
        if not rows:
            raise ValueError(f"mlat_range {mlat_range} selects no rows "
                             f"(grid covers {_MLAT_MIN}-{_MLAT_MAX} deg)")
        label = (f"MLAT {mlat_range[0]:g}-{mlat_range[1]:g}, "
                 f"MLT {mlt_range[0]:g}-{mlt_range[1]:g}")
        return rows, cols, label

    match = [r for r in SHAP_REGIONS if r["label"].lower() == str(target).lower()]
    if not match:
        raise ValueError(
            f"Unknown target {target!r}. Use 'overall', 'custom', or one of: "
            + ", ".join(r["label"] for r in SHAP_REGIONS))
    r = match[0]
    return (mlat_to_indices(r["mlat_low"], r["mlat_high"]),
            mlt_to_indices(r["mlt_start"], r["mlt_end"]),
            r["label"])


class _RegionalMeanWrapper(torch.nn.Module):
    """
    Wraps ACORN so it emits one scalar per sample: the mean of |FAC| over the
    selected region.

    The absolute value is taken before averaging. R1 and R2 currents are
    oppositely signed, so a signed regional mean would cancel them against
    each other and attribute near-zero importance to drivers that in fact
    control both sheets.
    """

    def __init__(self, model, rows, cols, channel=0, absolute=True):
        super().__init__()
        self.model = model
        self.register_buffer("rows", torch.as_tensor(rows, dtype=torch.long))
        self.register_buffer("cols", torch.as_tensor(cols, dtype=torch.long))
        self.channel  = channel
        self.absolute = absolute

    def forward(self, x):
        out = self.model(x)
        if out.dim() == 3:
            out = out.unsqueeze(0)
        # Trim midnight-looping padding on 50x26 variants.
        if out.shape[3] > _N_MLT:
            out = out[:, :, :, 1:-1]
        field = out[:, self.channel, :, :]
        field = field[:, self.rows, :][:, :, self.cols]
        if self.absolute:
            field = field.abs()
        return field.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)


# ── Climatology background for SHAP ──────────────────────────────────────────

_BACKGROUND_CACHE_DIR = Path(os.path.expanduser("~/.cache/acorn_shap_background"))


def build_climatology_background(config_path: str = "config.json",
                                 model_variant: Optional[str] = None,
                                 n_samples: int = 200,
                                 seed: int = 0,
                                 stratify: bool = True,
                                 cache: bool = True,
                                 rebuild: bool = False,
                                 scaler=None) -> np.ndarray:
    """
    Sample input sequences from the training record to serve as a SHAP
    climatology baseline.

    Why this rather than the recent window: SHAP values are differences from
    a reference, so the reference defines the question. A background drawn
    from the last 24 h answers "what makes this moment unusual for today",
    which changes meaning every time it is run and cannot be compared across
    events. A background drawn from the full training record answers "what
    makes this moment unusual relative to the conditions the model learned" --
    stable, and comparable between one prediction and another.

    Sampling is stratified over solar wind driving by default. A uniform
    random sample of the record would be dominated by quiet intervals, since
    quiet time is most of the record, and would make every storm prediction
    look extreme for the same undifferentiated reason. Stratifying across
    activity deciles keeps disturbed conditions represented in the baseline.

    Returns an array of shape (n_samples, T, F), already scaled, ready to
    pass to FACInference.explain(background=...).
    """
    cfg = _resolve_data_dir(utils.load_config(model_variant, config_path))
    variant = cfg["model_name"]
    T = cfg.get("time_history", 60)
    params = list(cfg["input_params"])

    key = f"{variant}_n{n_samples}_T{T}_s{seed}_{'strat' if stratify else 'unif'}"
    cache_path = _BACKGROUND_CACHE_DIR / f"background_{key}.npz"

    if cache and cache_path.exists() and not rebuild:
        d = np.load(cache_path, allow_pickle=True)
        if list(d["params"]) == params:
            print(f"Loaded cached climatology background: {cache_path.name} "
                  f"{d['background'].shape}")
            return d["background"]
        print("Cached background has different input params; rebuilding.")

    import data_prep

    print(f"Building climatology background for '{variant}' "
          f"({n_samples} samples, T={T})...")
    prep = data_prep.PreparingData(model=variant)
    prep.loading_solarwind()
    sw = prep.solarwind[params].copy()

    vals = sw.to_numpy()
    n = len(vals)
    if n <= T:
        raise RuntimeError(f"Training record has only {n} rows; need > {T}.")

    # Candidate window end positions with no NaN anywhere in the window.
    ok = ~np.isnan(vals).any(axis=1)
    csum = np.concatenate([[0], np.cumsum(ok)])
    ends = np.array([i for i in range(T, n) if csum[i] - csum[i - T] == T])
    if len(ends) == 0:
        raise RuntimeError("No complete NaN-free windows in the training record.")
    print(f"  {len(ends)} complete windows available")

    rng = np.random.default_rng(seed)

    if stratify and len(ends) > n_samples:
        # Stratify on a coupling proxy: |Vx| * |Bz| at the window end, which
        # tracks the dayside reconnection driving that sets FAC magnitude.
        try:
            vx = np.abs(vals[ends, params.index("Vx")])
            bz = np.abs(vals[ends, params.index("BZ_GSM")])
            drive = vx * bz
        except ValueError:
            drive = np.abs(vals[ends]).sum(axis=1)

        deciles = np.quantile(drive, np.linspace(0, 1, 11))
        per_bin = max(1, n_samples // 10)
        picks = []
        for lo, hi in zip(deciles[:-1], deciles[1:]):
            m = (drive >= lo) & (drive <= hi)
            pool = ends[m]
            if len(pool) == 0:
                continue
            take = rng.choice(pool, size=min(per_bin, len(pool)), replace=False)
            picks.append(take)
        chosen = np.concatenate(picks)
        if len(chosen) < n_samples:
            extra = rng.choice(np.setdiff1d(ends, chosen),
                               size=min(n_samples - len(chosen),
                                        len(np.setdiff1d(ends, chosen))),
                               replace=False)
            chosen = np.concatenate([chosen, extra])
        chosen = chosen[:n_samples]
        print(f"  stratified across activity deciles")
    else:
        chosen = rng.choice(ends, size=min(n_samples, len(ends)), replace=False)

    windows = np.stack([vals[e - T:e] for e in chosen], axis=0)

    if scaler is None:
        scaler_name = cfg.get("scaler_name") or cfg.get("scaler")
        scaler_path = None
        if scaler_name:
            for base in (Path(config_path).resolve().parent,
                         Path(cfg.get("model_dir", "models/"))):
                p = Path(base) / scaler_name
                if p.exists():
                    scaler_path = p
                    break
        if scaler_path is None:
            raise RuntimeError(
                "No scaler available to build the background. Pass "
                "scaler=<fitted scaler>, e.g. FACInference's _scaler, or set "
                "'scaler_name' in the config.")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    N, TT, F = windows.shape
    scaled = scaler.transform(windows.reshape(N * TT, F)).reshape(N, TT, F)

    if cache:
        _BACKGROUND_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, background=scaled,
                            params=np.array(params, dtype=object),
                            timestamps=np.array(
                                [str(sw.index[e]) for e in chosen], dtype=object))
        print(f"  cached to {cache_path}")

    print(f"  background ready: {scaled.shape}")
    return scaled


def auto_inference(config_path: str = "config.json",
                   preferred: str = "sci",
                   fallback: str = "op",
                   realtime: bool = False,
                   timestamp=None,
                   startdt: Optional[datetime.datetime] = None,
                   enddt: Optional[datetime.datetime] = None,
                   verbose: bool = True,
                   **kwargs):
    """
    Build a FACInference using `preferred`, falling back to `fallback` when the
    preferred model's inputs are not all obtainable for the window.

    Returns (wrapper, variant_used) -- the variant is returned rather than
    only printed because these are different models, not two settings of one
    model. ACORN Sci and ACORN Op differ in architecture, inputs and skill, so
    a figure or a stored result must be labelled with whichever actually ran.
    Callers that need a specific model should instantiate FACInference
    directly instead of using this.

    The availability probe fetches the input frame once and checks it against
    the preferred model's input_params. That frame is discarded and refetched
    by the wrapper; for OMNI windows the CDF cache makes the second fetch
    cheap, but this does cost an extra SuperMAG call.

    Parameters
    ----------
    preferred, fallback : str
        Model variants to try, in order.
    realtime : bool
        Probe the live feed rather than the OMNI archive.
    startdt, enddt : datetime, optional
        Window to probe when realtime=False. Defaults to the last 24 h of
        whatever OMNI has published, which is what predict() with no
        arguments will use.
    **kwargs
        Passed through to FACInference (model_path, lookback_limit,
        allow_ae_substitution, supermag_userid, ...).
    """
    probe_config = _resolve_data_dir(utils.load_config(preferred, config_path))
    needed = probe_config["input_params"]

    # A timestamp implies a historical run. The probe window must cover the
    # model's lookback as well as the timestamp itself, or the probe would
    # pass on a window the prediction then cannot build a sequence from.
    if timestamp is not None:
        realtime = False
        ts = pd.Timestamp(timestamp).to_pydatetime()
        pad = datetime.timedelta(
            minutes=probe_config.get("time_history", 60)
                    + kwargs.get("lookback_limit", 10) + 10)
        if startdt is None:
            startdt = ts - pad
        if enddt is None:
            enddt = ts + datetime.timedelta(minutes=1)

    if verbose:
        when = ("real-time" if realtime
                else f"{startdt:%Y-%m-%d %H:%M} -> {enddt:%Y-%m-%d %H:%M}"
                if startdt is not None else "OMNI")
        print(f"Checking whether '{preferred}' inputs are available ({when})...")

    reason = None
    try:
        if realtime:
            _load_solarwind_realtime(
                probe_config, vars_to_keep=needed,
                supermag_userid=kwargs.get("supermag_userid"))
        else:
            if enddt is None:
                enddt = datetime.datetime.utcnow()
            if startdt is None:
                startdt = enddt - datetime.timedelta(days=1)
            _load_solarwind_omni(
                probe_config, startdt, enddt, vars_to_keep=needed,
                allow_ae_substitution=kwargs.get("allow_ae_substitution", False),
                supermag_userid=kwargs.get("supermag_userid"))
        chosen = preferred
    except Exception as e:
        reason = str(e).split("\n")[0]
        chosen = fallback

    if chosen != preferred:
        print("")
        print(f"  '{preferred}' cannot run on this window: {reason}")
        print(f"  Falling back to '{fallback}'.")
        print(f"  NOTE: these are different models. Output is "
              f"{utils.load_config(fallback, config_path).get('version', fallback)}, "
              f"not {probe_config.get('version', preferred)}; label results "
              f"accordingly.")
        print("")
    elif verbose:
        print(f"'{preferred}' inputs are available; using it.")

    wrapper = FACInference(config_path=config_path, model_variant=chosen,
                           realtime=realtime, **kwargs)
    return wrapper, chosen


def plot_region_shap_polar(results, params=None, ncols=4, figsize_per=3.0,
                           channel_label="net SHAP"):
    """
    One polar panel per input parameter, each filled with that parameter's net
    SHAP contribution in every region.

    `results` is the dict returned by explain_regions(). Regions that do not
    tile the grid are left blank rather than interpolated, so a gap is visibly
    a gap.

    Each parameter gets its own symmetric colour scale, because the drivers
    differ in magnitude by more than an order of magnitude and a shared scale
    would flatten the weaker ones to invisibility. Read within a panel, not
    across panels.
    """
    labels = list(results.keys())
    params = params or results[labels[0]]["params"]

    n = len(params)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per * ncols, figsize_per * nrows),
                             subplot_kw=dict(projection="polar"))
    axes = np.atleast_1d(axes).ravel()

    r, th = np.meshgrid(
        np.linspace(0, _N_MLAT, _N_MLAT, endpoint=False),
        np.linspace(0, 2 * np.pi, _N_MLT, endpoint=False),
    )

    for ax, pi in zip(axes, range(n)):
        grid = np.full((_N_MLAT, _N_MLT), np.nan)
        for lab in labels:
            e = results[lab]
            grid[np.ix_(e["rows"], e["cols"])] = e["param_signed"][pi]

        lim = np.nanmax(np.abs(grid))
        lim = lim if (lim and np.isfinite(lim)) else 1.0
        mesh = ax.pcolormesh(th, r, grid.T, cmap="bwr",
                             norm=mpl.colors.Normalize(-lim, lim),
                             shading="auto")

        ax.set_theta_zero_location("S")
        ax.set_theta_direction(1)
        ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
        ax.set_xticklabels([])
        ax.set_yticks(np.linspace(0, _N_MLAT, 5, endpoint=False))
        ax.set_yticklabels([])
        ax.set_ylim(0, 35)
        ax.set_title(params[pi], fontsize=11)
        fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.06).ax.tick_params(labelsize=7)

    for ax in axes[n:]:
        ax.set_visible(False)

    ts = results[labels[0]].get("timestamp")
    ts_str = ts.strftime("%Y-%m-%d %H:%M UT") if hasattr(ts, "strftime") else str(ts)
    plt.suptitle(f"{channel_label} by region and driver — {ts_str}", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_region_lag_heatmaps(results, ncols=4, figsize_per=2.9):
    """
    SHAP by lag and driver, one panel per region, on a shared symmetric scale
    so panels can be compared directly.
    """
    labels = list(results.keys())
    params = results[labels[0]]["params"]
    lim = max(np.abs(results[l]["shap"]).max() for l in labels) or 1.0

    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per * ncols * 1.25,
                                      figsize_per * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, lab in zip(axes, labels):
        v = results[lab]["shap"]
        im = ax.pcolormesh(np.arange(v.shape[1] + 1),
                           np.arange(v.shape[0] + 1), v,
                           cmap="bwr", vmin=-lim, vmax=lim)
        ax.set_title(lab, fontsize=10)
        ax.set_xticks(np.arange(len(params)) + 0.5)
        ax.set_xticklabels(params, rotation=90, fontsize=6)

    for ax in axes[len(labels):]:
        ax.set_visible(False)

    for ax in axes[::ncols]:
        ax.set_ylabel("lag (min)", fontsize=8)

    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02)
    cb.set_label("SHAP value")
    plt.suptitle("SHAP by lag and driver, per region", fontsize=13)
    plt.show()


def plot_shap_region(expl=None, rows=None, cols=None, label=None,
                     field=None, time=None, ax=None, show=True,
                     highlight_color="limegreen"):
    """
    Show which cells a SHAP target covers, on the polar grid.

    Pass either an explain() result (expl) or explicit rows/cols. If `field`
    is given (a 50x24 prediction), it is drawn underneath with the region
    outlined, so the attribution area can be read against the current pattern;
    otherwise the region is drawn as a filled mask.

    The region is built as a mask over the full grid rather than as a wedge
    patch from degree bounds. Wedges drawn from bounds leave hairline gaps at
    cell edges and mishandle MLT sectors that wrap through midnight (21-02);
    a mask follows exactly the cells the attribution used.
    """
    if expl is not None:
        rows = expl["rows"] if rows is None else rows
        cols = expl["cols"] if cols is None else cols
        label = expl["label"] if label is None else label
        time = expl.get("timestamp") if time is None else time
    if rows is None or cols is None:
        raise ValueError("Provide expl, or both rows and cols.")

    mask = np.zeros((_N_MLAT, _N_MLT), dtype=float)
    mask[np.ix_(rows, cols)] = 1.0

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(7, 7),
                               subplot_kw=dict(projection="polar"))
    else:
        fig = ax.figure

    r, th = np.meshgrid(
        np.linspace(0, _N_MLAT, _N_MLAT, endpoint=False),
        np.linspace(0, 2 * np.pi, _N_MLT, endpoint=False),
    )

    if field is not None:
        f = np.asarray(field)
        if f.ndim == 3:
            f = f[0]
        lim  = np.nanmax(np.abs(f)) or 1.0
        norm = mpl.colors.Normalize(-lim, lim)

        # Two layers of the same data: greyscale everywhere, colour only
        # inside the region. Nothing is emphasised by being drawn heavier --
        # the full field stays visible and readable, and colour marks the
        # selection rather than signalling larger values.
        outside = np.where(mask > 0, np.nan, f)
        inside  = np.where(mask > 0, f, np.nan)

        ax.pcolormesh(th, r, outside.T, cmap="Greys_r", norm=norm,
                      shading="auto")
        m = ax.pcolormesh(th, r, inside.T, cmap="bwr", norm=norm,
                          shading="auto")
        cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.08)
        cb.set_label(r"FAC ($\mu A/m^2$)")
    else:
        ax.pcolormesh(th, r, np.where(mask.T > 0, 1.0, np.nan),
                      cmap=mpl.colors.ListedColormap([highlight_color]),
                      shading="auto")

    # Outline the region. The mask is padded with a zero border and wrapped
    # in MLT so the contour closes properly for sectors that cross midnight
    # (21-02) instead of leaving an open seam at the 0/24 boundary.
    th_c = np.linspace(0, 2 * np.pi, _N_MLT, endpoint=False)
    r_c  = np.arange(_N_MLAT, dtype=float)

    dth = th_c[1] - th_c[0]
    th_w = np.concatenate([[th_c[0] - dth], th_c, [th_c[-1] + dth]])
    r_w  = np.concatenate([[r_c[0] - 1.0], r_c, [r_c[-1] + 1.0]])

    m_w = np.zeros((_N_MLAT + 2, _N_MLT + 2))
    m_w[1:-1, 1:-1] = mask
    m_w[1:-1, 0]    = mask[:, -1]   # wrap: column before 0 is column 23
    m_w[1:-1, -1]   = mask[:, 0]    # wrap: column after 23 is column 0

    ax.contour(th_w, r_w, m_w, levels=[0.5],
               colors=[highlight_color], linewidths=2.0)

    ax.set_theta_zero_location("S")
    ax.set_theta_direction(1)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels(['', '3', '', '9', '', '15', '', '21'])
    ax.set_yticks(np.linspace(0, _N_MLAT, 5, endpoint=False))
    ax.set_yticklabels(['', '80', '70', '60', '50'])
    ax.set_ylim(0, 35)

    n_cells = len(rows) * len(cols)
    title = f"{label}  ({n_cells} cells)" if label else f"{n_cells} cells"
    if time is not None:
        ts = time.strftime("%Y-%m-%d %H:%M UT") if hasattr(time, "strftime") else str(time)
        title = f"{title}\n{ts}"
    ax.set_title(title)

    if created and show:
        plt.tight_layout()
        plt.show()
    return ax


def plot_all_shap_regions(field=None, ncols=4, figsize_per=3.2):
    """Grid of all SHAP_REGIONS, for picking a target."""
    n = len(SHAP_REGIONS)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per * ncols, figsize_per * nrows),
                             subplot_kw=dict(projection="polar"))
    axes = np.atleast_1d(axes).ravel()
    for ax, reg in zip(axes, SHAP_REGIONS):
        rows = mlat_to_indices(reg["mlat_low"], reg["mlat_high"])
        cols = mlt_to_indices(reg["mlt_start"], reg["mlt_end"])
        plot_shap_region(rows=rows, cols=cols, label=reg["label"],
                         field=field, ax=ax, show=False)
        ax.set_title(reg["label"], fontsize=10)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    for ax in axes[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.show()


def testing_polar_plot(samples, time, labels, std_labels=None):
    """
    Quick polar plot for sanity-checking model outputs side by side.

    Panels whose label indicates an uncertainty field are drawn on a
    sequential purple scale starting at zero, with their own colorbar. Mean
    fields share one symmetric bwr scale so they are directly comparable to
    each other. The two are never put on a shared scale: a standard deviation
    is non-negative and a diverging map centred on zero would misrepresent it.

    Detection is by substring ('std', 'sigma', 'uncertainty', case-insensitive)
    rather than exact match, so 'ACORN Op std' is recognised. Pass std_labels
    explicitly to override.
    """
    if std_labels is None:
        keys = ("std", "sigma", "uncertainty")
        is_std = [any(k in str(l).lower() for k in keys) for l in labels]
    else:
        is_std = [l in std_labels for l in labels]

    theta_ticks = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    rad_ticks   = np.linspace(0, 50, 5, endpoint=False)
    rad_labels  = ['', '80', '70', '60', '50']

    # Symmetric scale over the mean panels only -- including a non-negative
    # std here would inflate the range and wash out the mean fields.
    mean_samples = [s for s, f in zip(samples, is_std) if not f]
    if mean_samples:
        scale = max(np.nanmax(np.abs(s)) for s in mean_samples)
        scale = scale if scale > 0 else 1.0
    else:
        scale = 1.0
    scale_map = mpl.colors.Normalize(vmin=-scale, vmax=scale)

    # Std panels share their own scale, anchored at zero.
    std_samples = [s for s, f in zip(samples, is_std) if f]
    if std_samples:
        std_max = max(np.nanmax(s) for s in std_samples)
        std_map = mpl.colors.Normalize(vmin=0, vmax=std_max if std_max > 0 else 1.0)
    else:
        std_map = None

    fig, axes = plt.subplots(
        ncols=len(samples), nrows=1,
        figsize=(6 * len(samples), 10),
        subplot_kw=dict(projection='polar'),
    )
    if len(samples) == 1:
        axes = np.array([axes])

    ts_str = time.strftime("%Y-%m-%d %H:%M UT") if hasattr(time, "strftime") else str(time)
    plt.suptitle(ts_str, fontsize=20)

    r, th = np.meshgrid(
        np.linspace(0, 50, 50, endpoint=False),
        np.linspace(0, 2 * np.pi, 24, endpoint=False),
    )

    mean_handle, std_handle = None, None
    mean_axes, std_axes = [], []

    for ax, sample, label, flag in zip(axes, samples, labels, is_std):
        ax.set_title(label)
        ax.set_theta_zero_location('S')
        ax.set_theta_direction(1)
        if flag:
            std_handle = ax.pcolormesh(th, r, sample.T, cmap='Purples',
                                       norm=std_map)
            std_axes.append(ax)
        else:
            mean_handle = ax.pcolormesh(th, r, sample.T, cmap='bwr',
                                        norm=scale_map)
            mean_axes.append(ax)
        ax.set_xticks(theta_ticks)
        ax.set_xticklabels(['', '3', '', '9', '', '15', '', '21'])
        ax.set_yticks(rad_ticks)
        ax.set_yticklabels(rad_labels)
        ax.set_ylim(0, 35)

    if mean_handle is not None:
        cb = fig.colorbar(mean_handle, ax=mean_axes, orientation='vertical',
                          fraction=0.046, pad=0.08)
        cb.set_label(r"FAC ($\mu A/m^2$)")
    if std_handle is not None:
        cb = fig.colorbar(std_handle, ax=std_axes, orientation='vertical',
                          fraction=0.046, pad=0.08)
        cb.set_label(r"$\sigma$ ($\mu A/m^2$)")
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
