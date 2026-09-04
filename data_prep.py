"""
data_prep.py
============

Builds the training, validation and test sets for ACORN from raw solar
wind and AMPERE observations.

Pipeline
--------
1. loading_solarwind   OMNI + F10.7 + SuperMAG indices -> one DataFrame.
2. loading_ampere      AMPERE netCDF (or cached pickle) -> {timestamp: grid}.
3. split_sequences     Turn the flat time series into (time_history, n_features)
                       input sequences, one per AMPERE timestamp.
4. processing          Align inputs to targets, split train/val/test,
                       fit and apply scalers, and cache the result.

The record is segmented by calendar month, and the train/validation/test
split is made over whole months so that a sequence and its target never
straddle the split. Months containing a named `specific_test_storms`
entry are automatically added to the testing set.

Everything is driven by config.json, selected by model name. Nothing is
read at import time, so one process can prepare both datasets in turn:

    sci = PreparingData('sci')
    op = PreparingData('op')

Caching
-------
Several stages write pickles under `<data_dir>/prepared/` and are
skipped on a later run if the matching file already exists. Each cache
gets a small .json sidecar recording the settings that produced it, and
loading warns if those differ from the current config -- but after
changing anything upstream, delete the relevant pickle rather than
relying on the warning.

Only AMPERE data from 2019 onward is used (AMPERE NEXT); see AMPERE_START below.

Data layout
-----------
All paths are relative to `data_dir` from the config, which defaults to
`data/`:

    data/sw_data/omni/omni_10_min_interp.feather   from processing_omni.py
    data/sw_data/F107/fluxtable.txt                F10.7 solar flux
    data/indicies/supermag_indicies.feather        SuperMAG SME/SML/SMU
    data/ampere_data/                              raw AMPERE netCDF
    data/prepared/                                 outputs and caches
"""

from __future__ import annotations

import datetime
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import utils

pd.options.mode.chained_assignment = None

# AMPERE current densities smaller than this in magnitude are set to zero
# before training. Units are uA/m^2.
NOISE_FLOOR = 0.1

# AMPERE NEXT data period starting in 2019: the solar wind data is
# trimmed to match and only these years are ingested.
AMPERE_START = '2019-01-01'
AMPERE_END = '2025-12-31'
AMPERE_YEARS = range(2019, 2026)   # end-exclusive: covers 2019-2025 inclusive

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')



class PreparingData():
	"""
	A class to handle preprocessing and preparation of solar wind and AMPERE datasets
	for machine learning. This class initializes parameters, loads configuration, and
	sets up attributes for subsequent processing methods.
	"""

	def __init__(self,

				# Which model to prepare data for: 'sci' or 'op'. Passing
				# this rather than reading a module-level constant lets one
				# process prepare both datasets. None uses config.json's
				# 'active_model'.
				model: Optional[str] = None,

				# Path to the consolidated config file.
				config_path: str | Path = utils.DEFAULT_CONFIG_PATH,

				# Additional keyword arguments that will override defaults or add new attributes.
				**kwargs):

		# ---------------------------------------------------------------------
		# 1. Load configuration from external JSON file
		# ---------------------------------------------------------------------
		# load_config merges config.json's 'shared' block with the block
		# for the chosen model, and validates the model name.
		self.config = utils.load_config(model, config_path)

		# Retained so cache filenames and log messages can name the model
		# and the file that produced a given prepared dataset.
		self.model_name = self.config['model_name']

		# Cache directory, created if absent
		os.makedirs(os.path.join(self.config.get('data_dir', 'data/'), 'prepared'),
					exist_ok=True)
		self.config_path = Path(self.config['config_path'])

		# ---------------------------------------------------------------------
		# 2. Save input arguments as instance attributes
		# ---------------------------------------------------------------------
		self.data_dir = self.config.get("data_dir", "data/")			# Base directory containing all input and cached data
		self.version = self.config["version"]	 # Human-readable model name, e.g. 'ACORN_Sci'
		self.vars_to_keep = self.config["input_params"]	# List of input features to retain

		# Storm extraction configuration (optional)
		self.length = self.config.get("length",360)						 # Minimum storm sequence length
		self.patience = self.config.get("patience",120)					 # Allowed tolerance for brief non-storm periods

		# Optional testing setup
		self.time_history = self.config.get("time_history", 60)									# Length of input time history sequences
		self.specific_test_storms = self.config.get("specific_test_storms", None)				# List of storms to force into test set
		self.ampere_delay = self.config.get("ampere_delay", 0)									# Delay to apply to AMPERE data in minutes

		# Format string for converting between string timestamps and datetime objects
		self.datetime_format = '%Y-%m-%d %H:%M:%S'

		 # ---------------------------------------------------------------------
		 # 3. Apply additional keyword arguments
		 # ---------------------------------------------------------------------
		# This allows dynamic overriding or addition of instance attributes.
		# For example, passing extra flags or hyperparameters without modifying the signature.
		self.__dict__.update(kwargs)

		# Ensure that 'specific_test_storms' is defined, even if passed via kwargs
		self.specific_test_storms = self.__dict__.get('specific_test_storms', None)


	def loading_solarwind(self):
		"""
		Loads solar wind data and F10.7 flux, preprocesses both datasets,
		merges them on timestamp, selects variables, and removes NaNs.

		Stores:
			self.solarwind : fully processed pd.DataFrame
		"""

		print("Loading solar wind data...")

		# ---------------------------------------------------------
		# 1. Load solar wind data (Feather = faster on large frames)
		# ---------------------------------------------------------
		sw_path = os.path.join(self.data_dir, "sw_data", "omni", "omni_10_min_interp.feather")
		self.solarwind = pd.read_feather(sw_path)

		# Ensure index is a DatetimeIndex
		if not isinstance(self.solarwind.index, pd.DatetimeIndex):

			# If an 'Epoch' column exists, assume that is the timestamp
			if "Epoch" in self.solarwind.columns:
				self.solarwind.set_index("Epoch", inplace=True, drop=True)

			# Convert index to datetime using user's known format
			self.solarwind.index = pd.to_datetime(
				self.solarwind.index,
				format=self.datetime_format,
				errors="coerce"		# safer than raising
			)

		# ---------------------------------------------------------
		# 2. Add cyclical month encoding (vectorized)
		# ---------------------------------------------------------
		months = self.solarwind.index.month

		self.solarwind["month"] = months
		self.solarwind["sin_month"] = np.sin(months * 2 * np.pi / 12)
		self.solarwind["cos_month"] = np.cos(months * 2 * np.pi / 12)

		# -------------------------------------------------------------
		# 3. Load and preprocess F10.7 flux data and SuperMAG indicies
		# -------------------------------------------------------------
		f107_path = os.path.join(self.data_dir, "sw_data", "F107", "fluxtable.txt")
		indicies_path = os.path.join(self.data_dir, "indicies", "supermag_indicies.feather")

		# Regex whitespace split is faster and more consistent
		self.F107 = pd.read_csv(f107_path, sep=r"\s+")

		# Loading indicies data and setting datetime index.
		# Feather cannot store a non-default index, so the timestamp is
		# usually round-tripped as a 'Date_UTC' COLUMN. Converting the
		# RangeIndex instead would silently produce 1970 timestamps and a
		# join that matches nothing, so promote the column when present.
		self.indicies = pd.read_feather(indicies_path)
		if "Date_UTC" in self.indicies.columns:
			self.indicies.set_index("Date_UTC", inplace=True, drop=True)
		self.indicies.index = pd.to_datetime(self.indicies.index)

		if not self.indicies.index.is_monotonic_increasing:
			self.indicies.sort_index(inplace=True)

		# Drop first header-like row (as in original code)
		self.F107 = self.F107.iloc[1:]

		# Convert flux measurement to float
		self.F107["F107"] = self.F107["fluxadjflux"].astype(float)

		# ---- IMPORTANT: Group multiple flux measurements per day ----
		# Compute daily mean flux for dates encoded as YYYYMMDD
		daily_f107 = (
			self.F107
			.groupby("fluxdate")["F107"]
			.mean()
		)

		# Convert 'fluxdate' into a DatetimeIndex (+20:00 shift)
		daily_f107.index = (
			pd.to_datetime(daily_f107.index, format="%Y%m%d")
			+ datetime.timedelta(hours=20)
		)

		# Replace with final cleaned flux table
		self.F107 = daily_f107.to_frame()

		# ---------------------------------------------------------
		# 4. Merge F107 with solar wind dataframe
		# ---------------------------------------------------------
		self.solarwind = self.solarwind.join(self.F107, how="left")
		self.solarwind = self.solarwind.join(self.indicies, how='left')

		# Fill missing F107 values with a linear interpolation
		self.solarwind["F107"] = self.solarwind["F107"].interpolate("linear")

		# ---------------------------------------------------------
		# 5. Ensure variable list exists
		# ---------------------------------------------------------
		if self.vars_to_keep is None:
			raise ValueError("You must provide a list of variables to keep.")

		# ---------------------------------------------------------
		# 6. Restrict to the AMPERE data period
		# ---------------------------------------------------------
		# Trim to the AMPERE period; see AMPERE_START.
		self.solarwind = self.solarwind[AMPERE_START:]

		# ---------------------------------------------------------
		# 7. Keep only model-required variables
		# ---------------------------------------------------------
		self.solarwind = self.solarwind[self.vars_to_keep]

		# ---------------------------------------------------------
		# 8. Remove any remaining NaNs
		# ---------------------------------------------------------
		self.solarwind.dropna(inplace=True)


	def loading_ampere(self, ampere_from_cdf: bool = False):
		"""
		Loads AMPERE current density data.

		Two modes:
		----------
		1) ampere_from_cdf=True
			- Reads raw .nc files
			- Processes them with unpacking_current_density()
			- Saves the processed dict as a pickle

		2) ampere_from_cdf=False
			- Loads the pre-computed AMPERE pickle files

		Returns
		-------
		None
			Sets self.ampere to a dict of:
			{timestamp_string : pivot-table or array}
		"""
		print("Loading AMPERE data...")

		# directory paths
		ampere_dir = Path(self.data_dir) / "ampere_data"

		# assemble file list for the AMPERE data period
		def collect_files(years=AMPERE_YEARS):
			all_files = []
			for yr in years:
				pattern = f"ampere.{yr}*.nc"
				all_files.extend(sorted((ampere_dir).glob(pattern)))
			return all_files

		# -------------------------------------------------------
		# MODE 1 — Load directly from CDF files
		# -------------------------------------------------------
		if ampere_from_cdf:
			print("Processing AMPERE netCDF files...")
			ampere_dict = {}
			for nc_file in tqdm.tqdm(collect_files(),
							desc="Reading AMPERE netCDF", unit="file"):
				data = utils.unpacking_current_density(nc_file, pivot_or_array="pivot")
				ampere_dict.update(data)

			print(f"Loaded {len(ampere_dict)} AMPERE timesteps.")

			# Cache so later runs skip the (slow) netCDF parse entirely.
			with open(utils.ampere_file(self.config), "wb") as f:
				pickle.dump(ampere_dict, f)

			self.ampere = ampere_dict

		# -------------------------------------------------------
		# MODE 2 - Load from the cached pickle
		# -------------------------------------------------------
		else:
			path = Path(utils.ampere_file(self.config))
			if not path.exists():
				raise FileNotFoundError(f"Missing AMPERE pickle: {path}")
			print("Loading AMPERE from pickle...")
			with open(path, "rb") as f:
				self.ampere = pickle.load(f)

		# Zero out very small current densities to reduce noise.
		for key in tqdm.tqdm(self.ampere.keys(),
						desc='Zeroing sub-threshold FAC noise', unit='step'):
			self.ampere[key] = np.where(
				np.abs(self.ampere[key]) >= NOISE_FLOOR, self.ampere[key], 0
			)


	def split_sequences(self, time_stamps: Optional[List[pd.Timestamp]] = None, n_steps: int = 60) -> Dict[pd.Timestamp, np.ndarray]:
		"""
		Optimized version of `split_sequences()`.

		This version prepares model-ready input sequences by slicing directly
		from NumPy arrays rather than re-filtering the DataFrame for each timestamp.
		It is significantly faster for large datasets while producing identical output.

		Each sequence corresponds to a contiguous block of `n_steps` samples
		ending at a given timestamp in `time_stamps`.

		Parameters
		----------
		time_stamps : list of pandas.Timestamp, optional
			List of timestamps marking the endpoints of desired input sequences.
			Only timestamps that exist in `self.solarwind.index` are used.
		n_steps : int, default = 30
			Number of timesteps per sequence (the window size).

		Returns
		-------
		dict[pd.Timestamp, np.ndarray]
			Dictionary mapping each valid timestamp to its corresponding NumPy array
			of shape (n_steps, n_features), representing the input window.
		"""

		print("Splitting the sequences....")

		# --- Step 1: Setup and data extraction ---
		df = self.solarwind.copy()
		data_values = df.to_numpy()			# Convert full dataset to NumPy array for fast slicing
		index = df.index					# Pandas index (time-based)
		time_stamps = pd.to_datetime(time_stamps)

		# Filter only timestamps that exist in the dataset’s index
		valid_timestamps = [t for t in time_stamps if t in index]

		# Precompute a mapping from timestamp → integer position
		# This avoids calling .get_loc() repeatedly inside the loop
		index_to_pos = {ts: pos for pos, ts in enumerate(index)}

		# --- Step 2: Initialize output container ---
		X: Dict[pd.Timestamp, np.ndarray] = {}

		# --- Step 3: Iterate efficiently through timestamps ---
		for ts in tqdm.tqdm(valid_timestamps,
						desc="Building input sequences", unit="seq"):
			# Get the integer index of the timestamp minus the ampere delay
			end_pos = index_to_pos[ts] - self.ampere_delay

			# Ensure enough data exists before this point to form a complete sequence
			start_pos = end_pos - n_steps + 1
			if start_pos < 0:
				continue	# Skip timestamps without enough preceding history

			# Slice directly from the NumPy array
			# This avoids creating a temporary DataFrame
			window = data_values[start_pos:end_pos + 1, :]

			# Verify the window length before storing
			if window.shape[0] == n_steps:
				ts = ts.strftime(format=self.datetime_format)
				X[ts] = window

		# --- Step 4: Return dictionary of sequences ---
		print(f'Shape of last element of the resulting dict: {X[ts].shape}')
		print(f'Total number of resulting sequences: {len(X)}')

		return X

	def processing(self):
		"""
		Main preprocessing pipeline that:
			1. Loads solar wind and AMPERE data
			2. Optionally extracts only storm intervals
			3. Generates monthly (or storm-based) segmentation windows
			4. Splits data into train/val/test
			5. Merges AMPERE and OMNI sequences
			6. Removes NaN-containing samples
			7. Scales the input sequences
		Returns
		-------
		train, val, test : dicts
			Dictionaries of samples where each entry contains:
				{
					'input'	: (n_steps, n_features),
					'ampere' : AMPERE matrix for that timestamp
				}
		"""

		# ------------------------------------------------------------------
		# 1. LOAD SOLARWIND + AMPERE
		# ------------------------------------------------------------------

		self.loading_solarwind()
		# AMPERE normally loaded from pickle (much faster than CDF)

		if os.path.exists(utils.ampere_file(self.config)):
			ampere_from_cdf=False
		else:
			ampere_from_cdf=True

		self.loading_ampere(ampere_from_cdf=ampere_from_cdf)

		# ------------------------------------------------------------------
		# 2. MONTHLY SEGMENTATION
		# ------------------------------------------------------------------
		# Monthly segmentation windows spanning the AMPERE period.
		start_date = AMPERE_START
		end_date = AMPERE_END
		segmented_list = pd.date_range(
			start=pd.to_datetime(start_date),
			end=pd.to_datetime(end_date),
			freq='MS'
		).tolist()

		# ------------------------------------------------------------------
		# 3. LOAD PRE-SPLIT SEQUENCES IF THEY ALREADY EXIST
		# ------------------------------------------------------------------

		split_path = utils.sequences_file(self.config)

		if os.path.exists(split_path) and utils.cache_meta_check(split_path, self.config):
			print("Loading split data....")
			with open(split_path, 'rb') as f:
				merged_dict = pickle.load(f)

		else:
			print("Split data not found. Prepping....")
			ampere_keys = list(self.ampere.keys())

			# --------------------------------------------------------------
			# 4. GENERATE OMNI INPUT SEQUENCES (Sliding window with length == time history)
			# --------------------------------------------------------------
			omni_dict = self.split_sequences(
				time_stamps=ampere_keys,
				n_steps=self.time_history	# <-- heavy operation
			)

			# --------------------------------------------------------------
			# 5. MERGE OMNI + AMPERE USING SHARED KEYS
			# --------------------------------------------------------------
			common_keys = set(ampere_keys) & omni_dict.keys()

			# Fast filter: remove samples containing NaNs

			merged_dict = {
				key: {
					"input": omni_dict[key],
					"ampere": self.ampere[key]
				}
				for key in common_keys
				if (
					np.isnan(omni_dict[key]).any() == False
					and np.isnan(self.ampere[key]).any().any() == False
					and len(self.ampere[key])>0

				)
				}
			print(f"Number of samples after removing NaNs: {len(merged_dict)}")


			# Save for future runs, with a sidecar recording the settings
			# that produced it (see utils.cache_meta_write).
			with open(split_path, 'wb') as f:
				pickle.dump(merged_dict, f)
			utils.cache_meta_write(split_path, self.config)

		# ------------------------------------------------------------------
		# 6. SPECIAL TEST STORMS (OPTIONAL)
		# ------------------------------------------------------------------
		if self.specific_test_storms:
			# Hold out the month containing each named storm, so a case
			# study is never seen during training. The month is removed
			# from the pool of segments and added directly to the test set.
			test_storm_list = []
			for storm in self.specific_test_storms:
				storm_date = pd.to_datetime(storm)
				month_start = storm_date.replace(day=1, hour=0, minute=0, second=0)
				segmented_list = [seg for seg in segmented_list if seg != month_start]
				test_storm_list.append(month_start)

			print('Specific storm months held out for testing:')
			print(f'{test_storm_list}')

		# ------------------------------------------------------------------
		# 7. TRAIN / VAL / TEST SPLITTING
		# ------------------------------------------------------------------

		train_times, test_times = train_test_split(
			segmented_list,
			test_size=0.1,
			shuffle=self.config["shuffling_split_data"],
			random_state=self.config["random_seed"],
		)

		train_times, val_times = train_test_split(
			train_times,
			test_size=0.2,	# gives about 15% val
			shuffle=self.config["shuffling_split_data"],
			random_state=self.config["random_seed"],
		)

		if self.specific_test_storms:
			print(f'Test time length before adding specific storms: {len(test_times)}')
			test_times = test_times + test_storm_list
			print(f'Final test time length after adding specific storms: {len(test_times)}')
		# ------------------------------------------------------------------
		# 8. EXPAND MONTHLY WINDOWS INTO MINUTE TIMESTAMPS
		# ------------------------------------------------------------------

		# Expand monthly segments into 1-min timestamps
		train_date_range = pd.concat([
			pd.Series(pd.date_range(start=t, end=t + pd.offsets.MonthBegin(), freq='min', inclusive='left'))
			for t in train_times
		])

		val_date_range = pd.concat([
			pd.Series(pd.date_range(start=t, end=t + pd.offsets.MonthBegin(), freq='min', inclusive='left'))
			for t in val_times
		])

		test_date_range = pd.concat([
			pd.Series(pd.date_range(start=t, end=t + pd.offsets.MonthBegin(), freq='min', inclusive='left'))
			for t in test_times
		])

		# Convert into Series indexed by datetime (values unused)
		train_times = pd.Series(np.nan, index=train_date_range)
		val_times = pd.Series(np.nan, index=val_date_range)
		test_times = pd.Series(np.nan, index=test_date_range)

		# ------------------------------------------------------------------
		# 9. ALIGN MERGED DICTIONARY WITH TRAIN/VAL/TEST SETS
		# ------------------------------------------------------------------
		ampere_dates = pd.Series(np.nan, index=pd.to_datetime(list(merged_dict.keys())))

		# Use intersection between sequence timestamps & AMPERE timestamps
		train_dates = pd.concat([train_times.to_frame(), ampere_dates.to_frame()], axis=1, join='inner').index
		val_dates = pd.concat([val_times.to_frame(), ampere_dates.to_frame()], axis=1, join='inner').index
		test_dates = pd.concat([test_times.to_frame(), ampere_dates.to_frame()], axis=1, join='inner').index

		# Convert to string format expected by dictionary keys
		train_dates = train_dates.strftime(self.datetime_format)
		val_dates = val_dates.strftime(self.datetime_format)
		test_dates = test_dates.strftime(self.datetime_format)

		# Dictionary slicing
		train = {k: merged_dict[k] for k in train_dates}
		val = {k: merged_dict[k] for k in val_dates}
		test = {k: merged_dict[k] for k in test_dates}

		# ------------------------------------------------------------------
		# 10. FIT SCALER ON TRAINING INPUT SEQUENCES
		# ------------------------------------------------------------------

		# Guard against an empty split. np.vstack on an empty list raises
		# "need at least one array to concatenate", which says nothing
		# about the cause -- usually that the AMPERE record covers far
		# fewer months than the segmentation window, so the random month
		# split assigned every available month to val/test.
		if not train:
			raise ValueError(
				f"Training split is empty: {len(train_dates)} train timestamps "
				f"matched AMPERE data (val={len(val)}, test={len(test)}). "
				"This usually means the AMPERE record covers fewer months than "
				f"the segmentation window ({AMPERE_START} to {AMPERE_END}). "
				"Check that the AMPERE files cover the expected period, and that "
				"AMPERE_START / AMPERE_END match the data actually present."
			)

		# Stack all input arrays into one matrix for fitting (fastest approach)
		scaling_array = np.vstack([sample["input"] for sample in train.values()])
		print(f"Scaling array shape: {scaling_array.shape}")

		scaler = StandardScaler()
		scaler.fit(scaling_array)

		# ------------------------------------------------------------------
		# 11. APPLY SCALING TO TRAIN / VAL / TEST
		# ------------------------------------------------------------------

		def scale_dict(d, label):
			for key in tqdm.tqdm(d, desc=f"Scaling {label} inputs", unit="seq"):
				# scale input only (ampere is target)
				d[key]["input"] = scaler.transform(d[key]["input"])
				d[key]["ampere"] = np.array(d[key]["ampere"])

		scale_dict(train, "training")
		scale_dict(val, "validation")
		scale_dict(test, "testing")

		# Save scaler for inference
		with open(utils.scaler_file(self.config), "wb") as f:
			pickle.dump(scaler, f)

		return train, val, test


	def __call__(self):
		'''
		Calling the data prep class.

		Returns:
			train, val, test (dicts): dictionaries containing the training, validation and testing data

		'''
		prepared_path = utils.prepared_file(self.config)
		print(prepared_path)

		# The filename carries only the model name, so a cache built under
		# a different input set would otherwise be reused silently. The
		# sidecar records the settings behind each cache and this check
		# reports any mismatch.
		if os.path.exists(prepared_path) and utils.cache_meta_check(prepared_path, self.config):
			print('Loading pre-processed data....')
			with open(prepared_path, 'rb') as f:
				data = pickle.load(f)
			train = data['train']
			val = data['val']
			test = data['test']


		else:

			print(f'Prepared data not found. Beginning data preparation for {self.model_name}....')

			train, val, test = self.processing()

			print('Data processing complete... Saving results....')

			with open(prepared_path, 'wb') as f:
				pickle.dump({'train':train, 'val':val, 'test':test}, f)

			# Record the settings behind this cache for the check above.
			utils.cache_meta_write(prepared_path, self.config)


		return train, val, test
