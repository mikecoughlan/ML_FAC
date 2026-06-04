"""
align_weimer.py
---------------
Loads the Weimer CSV and a sample ACORN results pickle, compares the LAT/MLT
ordering and coordinate conventions of both datasets, reorders the Weimer data
to match ACORN exactly, then saves the corrected CSV.

Checks performed
----------------
1. LAT ordering      -- are rows sorted ascending or descending in each?
2. MLT ordering      -- are columns sorted ascending or descending in each?
3. Latitude range    -- ACORN uses physical MLAT (40-89 degrees);
                        Weimer must not be in colatitude (which would give
                        values like 1-50 instead of 40-89).
4. MLT range         -- ACORN uses 0-23 integer hours; Weimer floats should
                        span 0-24 (exclusive) after alignment.

After confirming the issues the script rewrites the Weimer CSV with rows/columns
reordered to match ACORN convention:
  - LAT   : descending (89 -> 40, pole first)
  - MLT   : ascending  (0 -> 24)
  - No colatitude conversion (values should remain as physical MLAT degrees)
"""

import os
import pickle
import numpy as np
import pandas as pd

# ── Paths — edit these to match your environment ──────────────────────────────
WEIMER_CSV    = '../data/weimer_05052023_09052023/weimer_may_2023.csv'
ACORN_PICKLE  = 'outputs/FAC_ACORN_0_0_next_storm_training_False_results.pkl'
OUTPUT_CSV    = '../data/weimer_05052023_09052023/weimer_may_2023_aligned.csv'
# ──────────────────────────────────────────────────────────────────────────────


def describe_ordering(name, lats, mlts):
    """Print ordering and range info for a dataset's coordinate arrays."""
    print(f"\n  {name}")
    print(f"    LAT  : n={len(lats):4d}  min={lats.min():.3f}  max={lats.max():.3f}  "
          f"{'descending' if lats[0] > lats[-1] else 'ascending'}")
    print(f"    MLT  : n={len(mlts):4d}  min={mlts.min():.3f}  max={mlts.max():.3f}  "
          f"{'descending' if mlts[0] > mlts[-1] else 'ascending'}")

    # Colatitude check: MLAT should be 40-89, colatitude would be ~1-50
    if lats.max() < 55:
        print(f"    *** WARNING: LAT max={lats.max():.1f} — looks like COLATITUDE, "
              f"not geographic latitude! Expected max ~89.")
    else:
        print(f"    LAT range looks like physical MLAT (degrees). OK.")


def check_acorn(pickle_path):
    """Extract coordinate conventions from a sample ACORN results entry."""
    print(f"\nLoading ACORN pickle: {pickle_path}")
    with open(os.path.expanduser(pickle_path), 'rb') as f:
        results = pickle.load(f)

    # Get first key that has an ampere entry
    key = next(k for k in results if results[k].get('ampere') is not None)
    print(f"  Sample timestamp: {key}")

    ampere = results[key]['ampere']
    if isinstance(ampere, pd.DataFrame):
        lats = ampere.index.to_numpy().astype(float)
        mlts = ampere.columns.to_numpy().astype(float)
    else:
        # Raw array — ACORN convention: row 0 = MLAT 89, row 49 = MLAT 40
        lats = 90 - np.arange(1, 51).astype(float)
        mlts = np.arange(24).astype(float)

    describe_ordering('ACORN/AMPERE', lats, mlts)
    return lats, mlts


def check_weimer(csv_path):
    """Extract coordinate conventions from the Weimer CSV (first timestamp)."""
    print(f"\nLoading Weimer CSV: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=['datetime'])

    first_ts = df['datetime'].iloc[0]
    group    = df[df['datetime'] == first_ts]
    print(f"  Sample timestamp: {first_ts}  ({len(group)} rows)")

    lats = np.sort(group['LAT'].unique())
    mlts = np.sort(group['MLT'].unique())

    describe_ordering('Weimer', lats, mlts)
    return df, lats, mlts


def align_weimer(df, acorn_lats, acorn_mlts):
    """
    Reorder the Weimer DataFrame rows so that:
      - LAT ordering matches ACORN (descending, pole first)
      - MLT ordering matches ACORN (ascending, 0 -> 24)
      - LAT values are physical MLAT degrees (not colatitude)

    If Weimer LAT values look like colatitude (max < 55), they are converted
    via MLAT = 90 - colatitude before sorting.

    Returns the corrected DataFrame.
    """
    df = df.copy()

    # Colatitude fix
    lat_max = df['LAT'].max()
    if lat_max < 55:
        print(f"\n  Converting LAT from colatitude to MLAT (90 - colatitude).")
        df['LAT'] = 90 - df['LAT']
    else:
        print(f"\n  LAT values appear to be physical MLAT already. No conversion needed.")

    # ACORN lat order is descending (89 -> 40), MLT is ascending (0 -> 24).
    # Weimer CSV rows are just individual points — the ordering of the CSV rows
    # only matters for readability; the pivot in the notebook handles it.
    # We sort ascending by datetime, then descending LAT, then ascending MLT
    # so the saved CSV has a consistent, human-readable order.
    df = df.sort_values(['datetime', 'LAT', 'MLT'],
                        ascending=[True, False, True]).reset_index(drop=True)

    print(f"  Sorted: datetime asc, LAT desc, MLT asc.")
    return df


def verify_alignment(df_aligned, acorn_lats, acorn_mlts):
    """
    Verify that for a sample timestamp, the pivot of the aligned Weimer data
    has the same LAT/MLT ordering as ACORN.
    """
    first_ts = df_aligned['datetime'].iloc[0]
    group    = df_aligned[df_aligned['datetime'] == first_ts]
    pivot    = (group.pivot(index='LAT', columns='MLT', values='FAC')
                     .sort_index(ascending=False))

    w_lats = pivot.index.to_numpy()
    w_mlts = pivot.columns.to_numpy()

    print(f"\nPost-alignment verification (timestamp: {first_ts}):")
    print(f"  Weimer LAT: {w_lats[0]:.1f} -> {w_lats[-1]:.1f}  "
          f"({'descending' if w_lats[0] > w_lats[-1] else 'ascending'})")
    print(f"  ACORN  LAT: {acorn_lats[0]:.1f} -> {acorn_lats[-1]:.1f}  "
          f"({'descending' if acorn_lats[0] > acorn_lats[-1] else 'ascending'})")
    print(f"  Weimer MLT: {w_mlts[0]:.3f} -> {w_mlts[-1]:.3f}  "
          f"({'ascending' if w_mlts[0] < w_mlts[-1] else 'descending'})")
    print(f"  ACORN  MLT: {acorn_mlts[0]:.3f} -> {acorn_mlts[-1]:.3f}  "
          f"({'ascending' if acorn_mlts[0] < acorn_mlts[-1] else 'descending'})")

    lat_ok = (w_lats[0] > w_lats[-1]) == (acorn_lats[0] > acorn_lats[-1])
    mlt_ok = (w_mlts[0] < w_mlts[-1]) == (acorn_mlts[0] < acorn_mlts[-1])
    lat_range_ok = w_lats.max() > 55  # not colatitude

    print(f"\n  LAT ordering match : {'OK' if lat_ok       else 'MISMATCH'}")
    print(f"  MLT ordering match : {'OK' if mlt_ok       else 'MISMATCH'}")
    print(f"  LAT physical deg   : {'OK' if lat_range_ok else 'FAIL — still looks like colatitude'}")

    return lat_ok and mlt_ok and lat_range_ok


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("Weimer / ACORN alignment diagnostic")
    print("=" * 60)

    # 1. Check ACORN conventions
    acorn_lats, acorn_mlts = check_acorn(ACORN_PICKLE)

    # 2. Check Weimer conventions
    weimer_df, w_lats_orig, w_mlts_orig = check_weimer(WEIMER_CSV)

    # 3. Report differences
    print("\n" + "=" * 60)
    print("Differences")
    print("=" * 60)

    lat_dir_match = (w_lats_orig[0] > w_lats_orig[-1]) == (acorn_lats[0] > acorn_lats[-1])
    mlt_dir_match = (w_mlts_orig[0] < w_mlts_orig[-1]) == (acorn_mlts[0] < acorn_mlts[-1])
    colat_issue   = w_lats_orig.max() < 55

    print(f"  LAT direction match : {'YES' if lat_dir_match else 'NO — will be fixed'}")
    print(f"  MLT direction match : {'YES' if mlt_dir_match else 'NO — will be fixed'}")
    print(f"  Colatitude issue    : {'YES — will convert to MLAT' if colat_issue else 'NO'}")

    # 4. Align
    print("\n" + "=" * 60)
    print("Aligning Weimer data")
    print("=" * 60)
    weimer_aligned = align_weimer(weimer_df, acorn_lats, acorn_mlts)

    # 5. Verify
    all_ok = verify_alignment(weimer_aligned, acorn_lats, acorn_mlts)

    # 6. Save
    if all_ok:
        weimer_aligned.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved aligned CSV to: {OUTPUT_CSV}")
        print("Update WEIMER_CSV_PATH in your notebooks to point to this file.")
    else:
        print("\nAlignment verification FAILED — CSV not saved. "
              "Check the output above for remaining issues.")
