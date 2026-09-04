# ACORN

**A**ttention **C**onvolutional **R**esidual **N**etwork — a probabilistic deep learning model for predicting high-latitude field-aligned currents (FACs) in the northern hemisphere ionosphere from upstream solar wind conditions.

ACORN predicts both a **mean** and a **standard deviation** for every cell of a 50 × 24 (MLAT × MLT) polar grid, giving a calibrated uncertainty alongside each prediction rather than a single point estimate. It is trained against AMPERE observations using a frequency-weighted Continuous Ranked Probability Score (CRPS).

This repository contains the full pipeline: data preparation, model definition, training, inference, and SHAP-based interpretability analysis.

---

## Table of contents

- [Models](#models)
- [Installation](#installation)
- [Data setup](#data-setup)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Repository layout](#repository-layout)
- [Pipeline in detail](#pipeline-in-detail)
- [Output files](#output-files)
- [Inference](#inference)
- [Interpretability](#interpretability)
- [Citation](#citation)

---

## Models

Two model variants ship with this repository. They share an architecture and differ in their input set and capacity.

| | **ACORN Sci** | **ACORN Op** |
|---|---|---|
| Selector | `sci` | `op` |
| Purpose | Science / retrospective analysis | Operational / real-time forecasting |
| Input features | 12 | 8 |
| Residual blocks | 3 | 2 |
| Layers per block | 3 | 2 |
| Parameters | 8,882,340 | 4,056,506 |

**Sci** uses the full input set, including ground-based geomagnetic indices (`SML`, `SMU`, `SYM_H`, `ASY_H`) that are only available after the fact. It is the more accurate model and the one used for the scientific results.

**Op** is restricted to quantities available in real time from upstream solar wind monitors, making it suitable for forecasting. It trades some accuracy for operational availability.

Both predict on the same 50 × 24 MLAT × MLT grid covering 40°–90° magnetic latitude, using a 60-minute lookback window of solar wind inputs.

---

## Installation

Python 3.9 or newer is required (the codebase uses `from __future__ import annotations` with PEP 604 type syntax).

```bash
git clone <repository-url>
cd acorn
pip install -r requirements.txt
```

### Dependencies

| Package | Used for |
|---|---|
| `torch` | ACORN model definition, training, inference |
| `numpy`, `pandas` | Array and tabular data handling |
| `scikit-learn` | Input scaling (`StandardScaler`), train/test splitting |
| `netCDF4` | Reading raw AMPERE data files |
| `cdflib` | Reading OMNI solar wind CDF files |
| `matplotlib` | Plotting |
| `tqdm` | Progress bars |
| `shap` | Interpretability analysis (`shap_values.py` only) |
| `requests` | Fetching OMNI data on demand during inference |
| `pyarrow` | Feather file I/O for the solar wind record |

`tensorflow` is an **optional** dependency, needed only for `TFACInference`, the wrapper for the Kunduri baseline model. The rest of the pipeline runs without it.

---

## Data setup

The pipeline expects the following layout under the directory given by `data_dir` in `config.json` (default `data/`). None of this data ships with the repository; see [Citation](#citation) for sources.

```
data/
├── ampere_data/
│   └── ampere.YYYY*.nc              Raw AMPERE netCDF files (2019 onward)
├── sw_data/
│   ├── omni/
│   │   └── omni_10_min_interp.feather   Produced by processing_omni.py
│   └── F107/
│       └── fluxtable.txt            F10.7 solar radio flux
├── indicies/
│   └── supermag_indicies.feather    SuperMAG SML / SMU indices
└── prepared/                        Created automatically; holds all caches
```

**Only AMPERE data from 2019 onward is used.** Earlier products come from a different processing release. The cutoff is set by `AMPERE_START` / `AMPERE_YEARS` in `data_prep.py`.

### Preparing the OMNI record

The raw OMNI 1-minute CDFs must be converted to a single feather file before anything else runs:

```bash
export AIMFAHR_OMNI_DIR=/path/to/omni    # must contain an hro2_1min/ subdirectory
python processing_omni.py
```

This reads every yearly CDF, replaces per-variable fill values with `NaN`, interpolates gaps of up to 10 minutes, keeps only the columns listed in `TO_KEEP`, and writes `omni_10_min_interp.feather`. Move that file to `data/sw_data/omni/`.

On the first run, check the printed list of kept and dropped columns — `TO_KEEP` is an explicit allow-list, and any variable name that does not match your OMNI files will be reported as missing.

---

## Quick start

```bash
# 1. Prepare the OMNI record (once)
python processing_omni.py

# 2. Train the science model
python model_training.py

# 3. Train the operational model
ACORN_MODEL=op python model_training.py
```

Data preparation runs automatically as the first stage of training. It is slow the first time — parsing several years of AMPERE netCDF files — and cached thereafter.

Model selection is by environment variable because the configuration is read at module import, before command-line arguments are parsed:

```bash
ACORN_MODEL=sci python model_training.py    # explicit (same as the default)
ACORN_MODEL=op  python model_training.py
```

With no variable set, the `active_model` key in `config.json` decides.

---

## Configuration

Everything is driven by a single `config.json`. There is no need to edit code to change a run.

```json
{
  "active_model": "sci",
  "shared": { ... settings common to both models ... },
  "sci":    { ... science model overrides ... },
  "op":     { ... operational model overrides ... }
}
```

A model's effective configuration is `shared` updated with its own block, so a model block may override any shared key. Load it directly with:

```python
import utils
config = utils.load_config('sci')     # or 'op', or None for active_model
```

### Shared settings

| Key | Default | Meaning |
|---|---|---|
| `data_dir` | `data/` | Root of all input data and caches |
| `model_dir` | `models/` | Where trained weights are written |
| `outputs_dir` | `outputs/` | Where results dictionaries are written |
| `random_seed` | `42` | Seed for splitting and initialisation |
| `time_history` | `60` | Lookback window, in minutes |
| `output_size` | `[50, 24]` | Output grid, MLAT × MLT |
| `ampere_delay` | `0` | Lag applied to AMPERE targets, in minutes |
| `loss` | `weighted_crps` | Objective; see `LOSS_REGISTRY` |
| `learning_rate` | `1e-6` | Adam learning rate |
| `batch_size` | `64` | Training batch size |
| `epochs` | `500` | Maximum epochs |
| `early_stop_patience` | `25` | Epochs without validation improvement before stopping |
| `shuffling_split_data` | `true` | Shuffle before train/val/test split |
| `specific_test_storms` | two dates | Storms whose months are held out of training for case studies |

### Per-model settings

| Key | Meaning |
|---|---|
| `version` | Human-readable model name, e.g. `ACORN_Sci` |
| `input_params` | Ordered list of input features |
| `bk_input_params` | Input list for the Kunduri baseline model (see `TFACInference`) |
| `model_config` | Architecture, passed straight to `ACORN(**model_config)` |

### Architecture settings (`model_config`)

| Key | Meaning |
|---|---|
| `in_channels` / `out_channels` | 1 in; 2 out (mean and standard deviation) |
| `base_channels` | Channel width at the stem |
| `depth` | Encoder/decoder depth |
| `num_res_blocks`, `layers_per_block` | Residual block structure |
| `channel_mult` | Channel growth factor per level |
| `use_cbam`, `cbam_reduction` | Convolutional Block Attention Module |
| `use_attention_gates` | Attention gates on skip connections |
| `dropout_rate`, `dropout_depth` | Dropout configuration |
| `conv_head_*` | Width, depth and dropout of the refinement head |

The refinement head is always built and has no on/off switch.

---

## Repository layout

| File | Role |
|---|---|
| `config.json` | Single configuration file for both models |
| `processing_omni.py` | Bulk conversion of raw OMNI CDFs to one feather file |
| `data_prep.py` | `PreparingData` — builds train/val/test sets from raw data |
| `model_classes.py` | `ACORN` and its building blocks |
| `custom_loss_functions.py` | CRPS and MSE losses, weighted and unweighted |
| `model_training.py` | Training loop, early stopping, evaluation |
| `inference.py` | `FACInference` (ACORN) and `TFACInference` (Kunduri baseline) wrappers |
| `shap_values.py` | Regional SHAP interpretability analysis |
| `utils.py` | Config loading, canonical filenames, AMPERE readers |
| `plotting_utils.py` | Polar plotting conventions |

---

## Pipeline in detail

`PreparingData` runs five stages, each cached:

1. **`loading_solarwind`** — merges the OMNI record, F10.7 flux, and SuperMAG indices into one DataFrame, trims it to the AMPERE period, and adds cyclical month features.

2. **`loading_ampere`** — reads AMPERE netCDF files into a dictionary keyed by timestamp, masks the ±1e30 fill values, and zeroes current densities below `NOISE_FLOOR` (0.1 µA/m²) to suppress inversion speckle. Cached to `ampere.pkl`.

3. **`split_sequences`** — converts the flat time series into `(time_history, n_features)` input sequences, one per AMPERE timestamp.

4. **`processing`** — segments the record by calendar month, splits train/validation/test over whole months, fits a `StandardScaler` on the training set only, applies it to all splits, and caches the result.

Splitting over whole months rather than individual timestamps keeps a sequence and its target on the same side of the split. Months containing a `specific_test_storms` entry are removed from the training pool and placed directly in the test set, so a case-study storm is never seen during training.

### Cache invalidation

Filenames carry only the model name, so runs differing in another setting write to the same path. To prevent a stale cache being reused silently, each cache is written with a small `.json` sidecar recording the settings behind it. On load, a mismatch prints a per-key warning:

```
WARNING: cache data/prepared/prepared_sci.pkl was built with different settings:
    input_params: cached=[...]  current=[...]
    Delete the cache and re-run, or results will not match the current config.
```

The warning does not stop the run. **After changing anything upstream, delete the affected cache rather than relying on it.**

---

## Output files

| Path | Contents |
|---|---|
| `data/prepared/ampere.pkl` | Cached AMPERE targets (model-independent) |
| `data/prepared/sequences_<model>.pkl` | Input sequences aligned to targets |
| `data/prepared/scaler_<model>.pkl` | Fitted `StandardScaler` |
| `data/prepared/prepared_<model>.pkl` | Final train/val/test dictionaries |
| `models/acorn_<model>.pt` | Trained weights |
| `outputs/results_<model>.pkl` | Test-set predictions and observations |
| `loss_tracker/loss_<model>.feather` | Per-epoch training and validation loss |
| `shap/shap_<model>_<explainer>_<region>.pkl` | SHAP attributions by region |

All paths are defined in one place — the filename helpers in `utils.py` — so training and inference cannot disagree about where a file lives.

---

## Inference

```python
from inference import FACInference

model = FACInference()                          # science model by default

# One timestamp -> (50, 24)
mean, std, times = model.predict(timestamp="2023-05-06 12:00:00")

# A time range, or a whole day -> (N, 50, 24)
mean, std, times = model.predict(start="2023-05-06 00:00:00",
                                 end="2023-05-07 00:00:00")
mean, std, times = model.predict(date="2023-05-06")
```

Pass exactly one of `timestamp`, `date`, or the `start`/`end` pair. `mean` and `std` are in µA/m², on the 50 × 24 MLAT × MLT grid; positive values are upward (out of the ionosphere) current.

`FACInference` fetches OMNI data on demand from NASA SPDF and caches the CDFs under `~/.cache/omni_cdfs/`, so no local OMNI archive is needed for inference — only for training.

For the operational model, or an explicit checkpoint:

```python
model = FACInference(model_variant="op")
model = FACInference(model_path="models/acorn_sci.pt")
```

`TFACInference` provides the same interface for the **Kunduri** model — the Keras/TensorFlow baseline that ACORN is compared against in the paper. It requires `tensorflow`, and reads its inputs from the `bk_input_params` list in `config.json`.

### Plotting

```python
from plotting_utils import plot_fac_polar
fig, ax, mesh = plot_fac_polar(mean[0])
fig.colorbar(mesh, ax=ax, label=r'FAC ($\mu A/m^2$)')
```

All polar plots follow one convention, applied by `plotting_utils.polar_axis`: midnight at the bottom, MLT increasing counter-clockwise (dawn left, dusk right), and a radial axis in colatitude labelled in MLAT. Route new polar plots through these helpers so figures stay comparable.

---

## Interpretability

`shap_values.py` computes SHAP attributions to answer which solar wind inputs, at which lags, drive predictions in a given part of the ionosphere.

ACORN produces 1200 outputs per timestep, which is neither tractable nor interpretable to explain cell by cell. Instead the grid is reduced to physically meaningful regions — the R0, R1 and R2 current sheets and MLT sectors — and `RegionWrapper` wraps the model so it returns a single scalar per region, which SHAP then explains.

```bash
python shap_values.py                  # science model
ACORN_MODEL=op python shap_values.py   # operational model
```

Runs are checkpointed per batch, so an interrupted run resumes from completed batches rather than recomputing them.

---

## Citation

If you use this code, please cite the accompanying paper *(details to be added on publication)*.

### Data sources

| Dataset | Source |
|---|---|
| AMPERE field-aligned currents | [ampere.jhuapl.edu](http://ampere.jhuapl.edu) |
| OMNI solar wind | [NASA SPDF / OMNIWeb](https://omniweb.gsfc.nasa.gov) |
| SuperMAG indices | [supermag.jhuapl.edu](https://supermag.jhuapl.edu) |
| F10.7 solar flux | Natural Resources Canada |
| Weimer model | [CCMC](https://ccmc.gsfc.nasa.gov) |

Please cite each dataset according to its own usage policy.
