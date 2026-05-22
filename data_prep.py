
from __future__ import annotations

import argparse
# Importing the libraries
import datetime
import gc
import glob
import json
import math
import os
import pickle
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
import tqdm
from scipy.stats import boxcox
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

import utils

# from torchsummary import summary
# from torchvision.models.feature_extraction import (create_feature_extractor,
#														get_graph_node_names)


pd.options.mode.chained_assignment = None

os.environ["CDF_LIB"] = "~/CDF/lib"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# Loading CONFIG json file
with open('config.json', 'r') as f:
	CONFIG = json.load(f)


class PreparingData():
	"""
	A class to handle preprocessing and preparation of solar wind and AMPERE datasets
	for machine learning. This class initializes parameters, loads configuration, and
	sets up attributes for subsequent processing methods.
	"""

	def __init__(self,

				# Additional keyword arguments that will override defaults or add new attributes.
				**kwargs):

		# ---------------------------------------------------------------------
		# 1. Load configuration from external JSON file
		# ---------------------------------------------------------------------
		with open('config.json', 'r') as f:
			# self.config is a dictionary containing general parameters such as:
			# 'version', 'input_params', 'ampere_version', 'data_version', etc.
			self.config = json.load(f)

		# ---------------------------------------------------------------------
		# 2. Save input arguments as instance attributes
		# ---------------------------------------------------------------------
		self.data_dir = self.config.get("data_dir", "../../../../data/mkcoughl/")			# Base directory containing data files
		self.version = self.config["version"]	 # Version identifier for dataset
		self.vars_to_keep = self.config["input_params"]	# List of input features to retain
		self.ampere_version = self.config["ampere_version"]	# Version identifier for AMPERE data
		self.data_version = self.config["data_version"]	# Version identifier for solar wind/OMNI data

		# Storm extraction configuration
		self.extract_storms = self.config.get("extract_storms",False)		 # Boolean flag to extract storms
		self.length = self.config.get("length",360)						 # Minimum storm sequence length
		self.patience = self.config.get("patience",120)					 # Allowed tolerance for brief non-storm periods
		self.lead = self.config.get("lead",1440)							 # Lead time to include before disturbed time found using extraction method
		self.recovery = self.config.get("recovery",2880)					 # Recovery time to include after disturbed time found using extraction method
		self.substorm_lead = self.config.get("substorm_lead",360)		 		# Lead time for substorms
		self.substorm_recovery = self.config.get("substorm_recovery",240) 		# Recovery time for substorms
		self.storm_lead = self.config.get("storm_lead",1440)			 # Lead time for storms
		self.storm_recovery = self.config.get("storm_recovery",2880)	 # Recovery time for storms
		self.storm_extract_param = self.config.get("storm_extract_param","AE_INDEX")	# Parameter name used to detect storms
		self.storm_extract_limit = self.config.get("storm_extract_limit",600)	# Threshold for storm detection

		# Optional testing setup
		self.time_history = self.config.get("time_history", 60)									# Length of input time history sequences
		self.specific_test_storms = self.config.get("specific_test_storms", None)					# List of storms to force into test set
		self.eras = self.config.get("eras", "next")																# Which eras to include: 'block_1', 'next', 'both'
		self.ampere_delay = self.config.get("ampere_delay", 0)												# Delay to apply to AMPERE data in minutes
		self.use_disturbed_time_list = self.config["use_disturbed_time_list"]			# Whether to load a disturbed time list

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

	def testing_polar_plot(self, sample):

		theta_ticks = np.linspace(0, 2*np.pi, 8, endpoint=False)
		theta_labels = ['0', '3', '6', '9', '', '15', '18', '21']
		rad_ticks = np.linspace(0,50,5, endpoint=False)
		rad_labels = ['', '80', '70', '60', '50']

		fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(10, 7), subplot_kw=dict(projection='polar'), gridspec_kw={'wspace':-0.1, 'hspace':0.1})
		axs.set_theta_zero_location('S')
		r,th = np.meshgrid(np.linspace(0,50,50, endpoint=False), np.linspace(0, 2*np.pi, 24, endpoint=False))
		axs.pcolormesh(th, r, sample.T, cmap='bwr')
		# axs[0,0].invert_yaxis()
		axs.set_xticks(theta_ticks)
		axs.set_xticklabels(['', '3', '', '9', '', '15', '', '21'])
		axs.set_yticks(rad_ticks)
		axs.set_yticklabels(rad_labels)
		axs.set_ylim(00,35)
		plt.savefig('plots/testing_fig.png')


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
		sw_path = self.data_dir + "sw_data/omni/omni_10_min_interp.feather"
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
		f107_path = self.data_dir + "sw_data/F107/fluxtable.txt"
		indicies_path = self.data_dir + "indicies/supermag_indicies.feather"

		# Regex whitespace split is faster and more consistent
		self.F107 = pd.read_csv(f107_path, sep=r"\s+")

		# Loading indicies data and setting datetime index
		self.indicies = pd.read_feather(indicies_path)
		self.indicies.index = pd.to_datetime(self.indicies.index)

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
		# 6. Filter by era selection
		# ---------------------------------------------------------
		if self.eras == "block_1":
			self.solarwind = self.solarwind[:"2018-01-01 00:00:00"]

		elif self.eras == "next":
			self.solarwind = self.solarwind["2018-01-01 00:00:00":]

		elif self.eras == "both":
			pass	# keep entire dataset

		else:
			raise KeyError('Choose between "block_1", "next", or "both".')

		# ---------------------------------------------------------
		# 7. Keep only model-required variables
		# ---------------------------------------------------------
		self.solarwind = self.solarwind[self.vars_to_keep]

		# ---------------------------------------------------------
		# 8. Optionally remove the storm extraction variable
		# ---------------------------------------------------------
		if not self.extract_storms and self.storm_extract_param in self.solarwind.columns:
			self.solarwind.drop(self.storm_extract_param, axis=1, inplace=True)

		# ---------------------------------------------------------
		# 9. Remove any remaining NaNs
		# ---------------------------------------------------------
		self.solarwind.dropna(inplace=True)


	def day_of_year_to_month_day(self, year, day_of_year, fractional_hour):
		"""
		Convert day of year to month and day.

		Parameters
		----------
		year : int
			Year (e.g., 2023)
		day_of_year : int
			Day of year (1-365 or 1-366 for leap years)
		fractional_hour : float
			Fractional hour (e.g., 14.5 for 14:30)

		Returns
		-------
		str
			Date and time in 'YYYY-MM-DD HH:MM:SS' string format
		"""
		hours = int(fractional_hour)
		minutes = int((fractional_hour*60) % 60)

		date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(day_of_year) - 1) + datetime.timedelta(hours=hours, minutes=minutes, seconds=int(0))
		return str(date.year) + '-' + str(date.month).zfill(2) + '-' + str(date.day).zfill(2) + ' ' + str(hours).zfill(2) + ':' + str(minutes).zfill(2) + ':' + str(int(0)).zfill(2)


	def unpacking_current_density(
		self,
		file: str | Path,
		pivot_or_array: str = "pivot"
		) -> dict:
		"""
		Extracts AMPERE current density data from a single netCDF file and returns a
		dictionary keyed by the timestamp string for each hourly record.

		Parameters
		----------
		file : str or Path
			Path to the AMPERE netCDF file.
		pivot_or_array : {'pivot', 'array'}
			Determines the output format for each timestamp:
			- 'array' : returns a 1D numpy array for jPar
			- 'pivot' : returns a 2D pivot table (lat x MLT)

		Returns
		-------
		dict
			{timestamp_string : pivot-table or array}
			One entry per hour in the file.
		"""
		file = Path(file)
		current_density_dict = {}

		try:
			# open netCDF file
			cdf = netCDF4.Dataset(file)

			# extract time-related variables once (faster)
			years = cdf.variables["year"][:]
			doys = cdf.variables["doy"][:]
			times = cdf.variables["time"][:]	# fractional hours

			jpar = cdf.variables["jPar"][:]	 # shape: (hours, points)
			mlt = cdf.variables["mlt_hr"][:]	# shape: (hours, points)
			lat = cdf.variables["cLat_deg"][:]	# shape: (hours, points)
			jpar[jpar>1e30] = np.nan
			jpar[jpar<-1e30] = np.nan
			n_hours = len(times)

			for hour in range(n_hours):
				# convert Y, DOY, fractional hour → datetime string
				timestamp = self.day_of_year_to_month_day(
					years[hour], doys[hour], times[hour]
				)

				if pivot_or_array == "array":
					# 1D array: shape (points,)
					current_density_dict[timestamp] = np.array(jpar[hour, :])
					# current_density_dict[timestamp][current_density_dict[timestamp]>1e30] = np.nan

				elif pivot_or_array == "pivot":
					# Build DataFrame → pivot to 2D (lat x MLT)
					df = pd.DataFrame({
						"current_density": jpar[hour, :],
						"mlt": mlt[hour, :],
						"lat": lat[hour, :]
					})
					# df['current_density'][df['current_density'] > 1e30] = np.nan
					current_density_dict[timestamp] = df.pivot_table(
						index="lat",
						columns="mlt",
						values="current_density"
					)

				else:
					raise ValueError("pivot_or_array must be either 'pivot' or 'array'")

		except KeyError as e:
			print(f"KeyError {e} encountered in file {file}. Skipping file.")

		return current_density_dict


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
		prepared_dir = Path(self.data_dir) / "prepared_data"

		# Year ranges
		block_1_years = range(2009, 2018)
		next_years = range(2018, 2025)

		# assemble file lists
		def collect_files(years):
			all_files = []
			for yr in years:
				pattern = f"ampere.{yr}*.nc"
				all_files.extend(sorted((ampere_dir).glob(pattern)))
			return all_files

		# -------------------------------------------------------
		# MODE 1 — Load directly from CDF files
		# -------------------------------------------------------
		if ampere_from_cdf:
			block_1_dict = {}
			next_dict = {}

			# --- BLOCK 1 ---
			if self.eras in ("block_1", "both"):
				print("Processing Block 1 AMPERE CDF files...")
				for nc_file in tqdm.tqdm(collect_files(block_1_years)):
					data = self.unpacking_current_density(nc_file, pivot_or_array="pivot")
					block_1_dict.update(data)

				# save pickle
				with open(prepared_dir / f"ampere_block_1_{self.ampere_version}.pkl", "wb") as f:
					pickle.dump(block_1_dict, f)

				self.ampere = block_1_dict

			# --- NEXT ERA ---
			if self.eras in ("next", "both"):
				print("Processing Next-era AMPERE CDF files...")
				for nc_file in tqdm.tqdm(collect_files(next_years)):
					data = self.unpacking_current_density(nc_file, pivot_or_array="pivot")
					next_dict.update(data)
				print(len(next_dict))
				print(os.getcwd())
				with open(prepared_dir / f"ampere_next_{self.ampere_version}.pkl", "wb") as f:
					pickle.dump(next_dict, f)

				self.ampere = next_dict

			# --- BOTH ERAS MERGED ---
			if self.eras == "both":
				self.ampere = {**block_1_dict, **next_dict}
				del block_1_dict, next_dict
				gc.collect()

			return

		# -------------------------------------------------------
		# MODE 2 — Load from pickle
		# -------------------------------------------------------
		if self.eras not in ("block_1", "next", "both"):
			raise ValueError('eras must be one of "block_1", "next", "both"')

		# Load Block 1
		if self.eras in ("block_1", "both"):
			path = prepared_dir / f"ampere_block_1_{self.ampere_version}.pkl"
			if not path.exists():
				raise FileNotFoundError(f"Missing AMPERE pickle: {path}")
			print("Loading block_1 AMPERE from pickle...")
			with open(path, "rb") as f:
				block_1_dict = pickle.load(f)
			self.ampere = block_1_dict

		# Load Next
		if self.eras in ("next", "both"):
			path = prepared_dir / f"ampere_next_{self.ampere_version}.pkl"
			if not path.exists():
				raise FileNotFoundError(f"Missing AMPERE pickle: {path}")
			print("Loading next-era AMPERE from pickle...")
			with open(path, "rb") as f:
				next_dict = pickle.load(f)
			self.ampere = next_dict

		# Merge both
		if self.eras == "both":
			self.ampere = {**block_1_dict, **next_dict}
			del block_1_dict, next_dict
			gc.collect()

		# # setting anything below plus or minus 0.1 to zero to reduce noise
		# for key in self.ampere.keys():
		# 	self.ampere[key] = np.where(np.abs(self.ampere[key]) >= 0.1, self.ampere[key], 0)


	def checking_for_storm(self, i: int, param_values: np.ndarray) -> Tuple[int, Optional[pd.Series]]:
		"""
		Scans through solar wind data starting from index `i` (using pre-cached NumPy array)
		to determine if a "storm" event occurs.

		This version performs identically to the original but runs much faster because
		it uses NumPy array indexing instead of slow pandas `.iloc` lookups inside loops.

		A "storm" is defined as a contiguous or mostly contiguous sequence of points where
		the chosen solar wind parameter (`self.storm_extract_param`) exceeds a defined
		threshold (`self.storm_extract_limit`) for at least a minimum number of points
		(`self.length`), allowing brief interruptions up to `self.patience`.

		Parameters
		----------
		i : int
			Current index in the solar wind dataset to start checking from.
		param_values : np.ndarray
			NumPy array of values from `self.solarwind[self.storm_extract_param]`.
			Precomputed once for speed — avoids repeated `.iloc` lookups.

		Returns
		-------
		Tuple[int, Optional[pd.Series]]
			- Updated index `i` after scanning this region.
			- Extracted pandas Series representing the storm, or None if no valid storm found.
		"""

		# initial_index : Starting point for extraction, includes pre-storm lead points
		initial_index = i - self.lead

		# length_counter : Counts how many consecutive points satisfy the storm condition
		# patience_counter : Counts tolerated interruptions (non-storm points)
		length_counter, patience_counter = 0, 0

		# n : Total number of data points in the parameter array
		n = len(param_values)

		# --- Main loop: scan forward through the dataset ---
		while i < n:

			# Current parameter value from the pre-cached array
			current_value = param_values[i]

			# Check whether current value meets the storm condition.
			# Handles both positive and negative thresholds.
			condition_met = (
				(current_value <= self.storm_extract_limit and self.storm_extract_limit < 0)
				or
				(current_value >= self.storm_extract_limit and self.storm_extract_limit > 0)
			)

			if condition_met:
				# Condition satisfied → extend current storm run
				patience_counter = 0		# reset patience window
				length_counter += 1			# increment storm length
				i += 1						# move to next data point

			else:
				# Condition not met → potential break in storm sequence

				if patience_counter <= self.patience:
					# Still within tolerance window → allow this gap
					patience_counter += 1
					i += 1

				elif patience_counter > self.patience and length_counter < self.length:
					# Too many breaks and not enough valid points → discard attempt
					return i, None

				else:
					# Storm sequence long enough → finalize and return
					i += self.recovery	# Skip ahead to avoid overlapping detections
					storm_data = self.solarwind[self.storm_extract_param].iloc[initial_index:i]
					return i, storm_data

		# If end of dataset reached without completing a storm
		return i, None


	def storm_extract(self) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
		"""
		Extracts all storm events from the solar wind dataset using a faster,
		hybrid approach based on NumPy array access.

		This version preserves all the logic of the original `storm_extract()`
		but avoids pandas `.iloc` lookups inside loops, providing substantial
		speed improvements on large datasets.

		Returns
		-------
		Tuple[pd.Series, List[pd.Series]]
			- A concatenated pandas Series containing all detected dist_times.
			- A list of individual storm Series for separate analysis.
		"""

		# param_values : Pre-cached NumPy array of the target solar wind parameter.
		# Accessing values directly from this array avoids pandas overhead.

		# dist_times : List to hold each detected storm segment as a pandas Series
		dist_times: List[pd.DataFrame] = []
		substorms: List[pd.DataFrame] = []
		storms: List[pd.DataFrame] = []

		if not self.use_disturbed_time_list:
			param_values = self.solarwind[self.storm_extract_param].to_numpy()

			# n : Total number of samples in the dataset
			n = len(param_values)

			# i : Current scanning index within the data
			i = 0

			# --- Main scanning loop ---
			while i < n:

				# Current parameter value from pre-cached array
				current_value = param_values[i]

				# Check if this point potentially marks the start of a storm.
				potential_start = (
					(current_value <= self.storm_extract_limit and self.storm_extract_limit < 0)
					or
					(current_value >= self.storm_extract_limit and self.storm_extract_limit > 0)
				)

				if potential_start:
					# Attempt to extract the full storm from this index
					i, storm = self.checking_for_storm(i, param_values)

					# If a valid storm was detected (a pandas Series is returned)
					if isinstance(storm, pd.Series):
						if storm.index.duplicated().any():
							raise ValueError("Duplicated indices found in extracted storm data.")

						if not dist_times:
							# No previous dist_times → first detection
							dist_times.append(storm)

						elif storm.index.isin(dist_times[-1].index).any():
							# Overlapping indices with last storm → merge them
							dist_times[-1] = pd.concat([dist_times[-1], storm])
							dist_times[-1] = dist_times[-1][~dist_times[-1].index.duplicated(keep='first')]

						else:
							# Distinct storm → append as new segment
							dist_times.append(storm)

				# Move forward in the dataset (even if no storm found)
				i += 1

		else:
			substorm_list = pd.read_csv('substorm_time_list.csv')
			substorm_list['Date_UTC'] = pd.to_datetime(substorm_list['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
			storm_list = pd.read_csv('storm_time_list.csv')
			storm_list['Date_UTC'] = pd.to_datetime(storm_list['Date_UTC'], format='%Y-%m-%d %H:%M:%S')

			for date in tqdm.tqdm(substorm_list['Date_UTC'], desc="Processing substorm time list"):
				start = date - pd.Timedelta(minutes=self.substorm_lead)
				end = date + pd.Timedelta(minutes=self.substorm_recovery)

				if start < self.solarwind.index[0] or end > self.solarwind.index[-1]:
					continue

				dist_time = self.solarwind[(self.solarwind.index >= start) & (self.solarwind.index <= end)][self.storm_extract_param]

				if not substorms:
					# No previous substorms → first detection
					substorms.append(dist_time)

				elif dist_time.index.isin(substorms[-1].index).any():
					# Overlapping indices with last dist_time → merge them
					substorms[-1] = pd.concat([substorms[-1], dist_time])
					substorms[-1] = substorms[-1][~substorms[-1].index.duplicated(keep='first')]

				else:
					substorms.append(dist_time)

			for date in tqdm.tqdm(storm_list['Date_UTC'], desc="Processing storm time list"):
				start = date - pd.Timedelta(minutes=self.storm_lead)
				end = date + pd.Timedelta(minutes=self.storm_recovery)

				if start < self.solarwind.index[0] or end > self.solarwind.index[-1]:
					continue

				dist_time = self.solarwind[(self.solarwind.index >= start) & (self.solarwind.index <= end)][self.storm_extract_param]

				if not storms:
					# No previous storms → first detection
					storms.append(dist_time)

				elif dist_time.index.isin(storms[-1].index).any():
					# Overlapping indices with last dist_time → merge them
					storms[-1] = pd.concat([storms[-1], dist_time])
					storms[-1] = storms[-1][~storms[-1].index.duplicated(keep='first')]

				else:
					storms.append(dist_time)

			# checking for overlapping indices between dist_times and storms
			for storm in tqdm.tqdm(storms.copy(), desc='Checking for overlapping indices between storms and substorms'):
				substorms_to_remove = []
				for j,substorm in enumerate(substorms.copy()):
					if substorm.index.isin(storm.index).any():
						# Overlapping indices with storm → merge them
						storm = pd.concat([substorm, storm])
						storm = storm[~storm.index.duplicated(keep='first')]
						substorms_to_remove.append(j)
				# removing the merged substorms from the substorm list
				for j in sorted(substorms_to_remove, reverse=True):
					del substorms[j]

			# after substomrs are merged with the storms, checking to make sure the storms now don't overlap
			for storm in tqdm.tqdm(storms.copy(), desc='Checking for overlapping indices between storms now that substomrs have been merged'):
				for j,other_storm in enumerate(storms.copy()):
					if storm.equals(other_storm):
						continue
					if other_storm.index.isin(storm.index).any():
						# Overlapping indices with last storm → merge them
						storm = pd.concat([storm, other_storm])
						storm = storm[~storm.index.duplicated(keep='first')]
						storms.remove(other_storm)

			dist_times = storms + substorms

		# If no dist_times were detected, return empty outputs
		if not dist_times:
			return pd.Series(dtype=float), []

		# Combine all dist_times into a single Series for convenience
		combined_times = pd.concat(dist_times)

		return combined_times, dist_times


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
		df.drop(self.storm_extract_param, axis=1, inplace=True, errors='ignore')  # Remove storm extraction column if present
		data_values = df.to_numpy()			# Convert full dataset to NumPy array for fast slicing
		index = df.index					# Pandas index (time-based)
		time_stamps = pd.to_datetime(time_stamps)
		n = len(index)

		# Filter only timestamps that exist in the dataset’s index
		valid_timestamps = [t for t in time_stamps if t in index]

		# Precompute a mapping from timestamp → integer position
		# This avoids calling .get_loc() repeatedly inside the loop
		index_to_pos = {ts: pos for pos, ts in enumerate(index)}

		# --- Step 2: Initialize output container ---
		X: Dict[pd.Timestamp, np.ndarray] = {}

		# --- Step 3: Iterate efficiently through timestamps ---
		for ts in tqdm.tqdm(valid_timestamps, desc="Processing timestamps"):
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


	def checking_for_extra_time(self, times: pd.Series) -> pd.Series:
		"""
		checking for extra time that needs to be added because of specific test storms listed
		in the config file. If config flie contains a specific test storm that is listed as
		YYYY-MM format, then the entire month is added to the test set.

		Args:
			times (pd.Series): series of datetime indices
		Returns:
			pd.Series: updated series of datetime indices
		"""
		for storm in self.specific_test_storms:
			print(storm)
			try:
				datetime.datetime.strptime(storm, "%Y-%m")
				truncated_month = True
			except ValueError:
				truncated_month = False

			if truncated_month:
				# adding entire month to the test set
				print('Entering truncated month loop')
				year, month = storm.split('-')
				start_date = datetime.datetime(int(year), int(month), 1, 0, 0)
				if month == '12':
					end_date = datetime.datetime(int(year)+1, 1, 1, 0, 0)
				else:
					end_date = datetime.datetime(int(year), int(month)+1, 1, 0, 0)

				extra_dates = pd.date_range(start=start_date, end=end_date, freq='min', inclusive='left')
				times = pd.concat([times, pd.Series(np.nan, index=extra_dates)], axis=0)

			else:
				# if day is listed, add 5 days on either side of the labeled date
				print('Entering truncated time loop')
				storm_time = datetime.datetime.strptime(storm, self.datetime_format)
				start_date = storm_time - datetime.timedelta(days=5)
				end_date = storm_time + datetime.timedelta(days=5)

				extra_dates = pd.date_range(start=start_date, end=end_date, freq='min', inclusive='left')
				times = pd.concat([times, pd.Series(np.nan, index=extra_dates)], axis=0)

		times = times[~times.index.duplicated(keep='first')]

		return times


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

		if os.path.exists(self.data_dir+f'prepared_data/ampere_{self.eras}_{self.ampere_version}.pkl'):
			ampere_from_cdf=False
		else:
			ampere_from_cdf=True

		self.loading_ampere(ampere_from_cdf=ampere_from_cdf)

		# ------------------------------------------------------------------
		# 2. STORM EXTRACTION (OPTIONAL)
		# ------------------------------------------------------------------

		if self.extract_storms:
			# segmented_timestamps: datetime index of intervals
			# segmented_list: list of separate storm segments (Series)
			print("Extracting storms from solar wind data....")
			segmented_timestamps, segmented_list = self.storm_extract()

			print('Checking for extra time to add based on specific test storms....')
			segmented_timestamps = self.checking_for_extra_time(segmented_timestamps)
			print(f'Length of segmented list: {len(segmented_list)}')
			# Keep only AMPERE timestamps inside storm intervals
			print("Filtering AMPERE data to storm intervals....")
			segmented_timestamps_index = segmented_timestamps.index
			self.ampere = {
				key: val for key, val in tqdm.tqdm(self.ampere.items())
				if key in segmented_timestamps_index
			}
		else:
			# Monthly segmentation windows from 2009 → 2025
			if self.eras == "block_1":
				start_date = '2009-07-01'
				end_date = '2018-01-01'
			elif self.eras == "next":
				start_date = '2018-01-01'
				end_date = '2025-08-01'
			else:
				start_date = '2009-07-01'
				end_date = '2025-08-01'
			segmented_list = pd.date_range(
				start=pd.to_datetime(start_date),
				end=pd.to_datetime(end_date),
				freq='MS'
			).tolist()

		# ------------------------------------------------------------------
		# 3. LOAD PRE-SPLIT SEQUENCES IF THEY ALREADY EXIST
		# ------------------------------------------------------------------

		split_path = (
			f"{self.data_dir}prepared_data/"
			f"sequence_split_{self.data_version}_ampere_{self.ampere_version}_"
			f"storm_extracted_{self.extract_storms}_{self.eras}_eras.pkl"
		)

		if os.path.exists(split_path):
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


			# Save for future runs
			with open(split_path, 'wb') as f:
				pickle.dump(merged_dict, f)

		# ------------------------------------------------------------------
		# 6. SPECIAL TEST STORMS (OPTIONAL)
		# ------------------------------------------------------------------
		if self.specific_test_storms:
			test_storm_list, segmented_to_remove = [],[]
			for storm in self.specific_test_storms:
				try:
					datetime.datetime.strptime(storm, "%Y-%m")
					truncated_month = True
				except ValueError:
					truncated_month = False
				# can do the splitting directly on teh storms if extracted
				if not truncated_month and self.extract_storms:
					for i, extracted_storm in enumerate(segmented_list):
						# Corrected membership check
						if datetime.datetime.strptime(storm, self.datetime_format) in extracted_storm.index:
							test_storm_list.append(extracted_storm)
							segmented_to_remove.append(i)
				elif truncated_month and self.extract_storms:
					storm_date = pd.to_datetime(storm)
					storm = pd.Series(pd.date_range(start=storm_date, end=storm_date + pd.offsets.MonthBegin(), freq='min', inclusive='left'))
					for i, extracted_storm in enumerate(segmented_list):
						# Corrected membership check
						if extracted_storm.index.isin(storm).any():
							segmented_to_remove.append(i)
					temp_solar = self.solarwind.copy()
					temp_solar.index = pd.to_datetime(temp_solar.index)
					try:
						storm = temp_solar[self.storm_extract_param][storm.iloc[0]:storm.iloc[-1]]
						test_storm_list.append(storm)
					except KeyError:
						print(f'Storm {storm} not found in solarwind data! Double check the eras being used!')

				# if storms not extracted just use the month adn year to get the time period containing the storm
				else:
					storm_date = pd.to_datetime(storm)
					month_start = storm_date.replace(day=1, hour=0, minute=0, second=0)
					segmented_list = [
						seg for seg in segmented_list
						if seg != month_start
					]
					test_storm_list.append(pd.to_datetime(storm))
			if len(segmented_to_remove) > 0:
				print(len(segmented_to_remove))
				for to_remove in sorted(segmented_to_remove, reverse=True):
					segmented_list.pop(to_remove)
			print(f'Checking if this specific storms were extracted for testing:')
			print(f'{test_storm_list}')
		# ------------------------------------------------------------------
		# 7. TRAIN / VAL / TEST SPLITTING
		# ------------------------------------------------------------------

		train_times, test_times = train_test_split(
			segmented_list,
			test_size=0.1,
			shuffle=CONFIG["shuffling_split_data"],
			random_state=self.config["random_seed"],
		)

		train_times, val_times = train_test_split(
			train_times,
			test_size=0.2,	# gives about 15% val
			shuffle=CONFIG["shuffling_split_data"],
			random_state=self.config["random_seed"],
		)

		if self.specific_test_storms:
			print(f'Test time length before adding specific storms: {len(test_times)}')
			test_times = test_times + test_storm_list
			print(f'Final test time length after adding specific storms: {len(test_times)}')
		# ------------------------------------------------------------------
		# 8. EXPAND MONTHLY WINDOWS INTO MINUTE TIMESTAMPS
		# ------------------------------------------------------------------

		if not self.extract_storms:
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

		else:
			# Storm intervals already contain minute timestamps
			train_times = pd.concat(train_times)
			val_times = pd.concat(val_times)
			test_times = pd.concat(test_times)

		print(f"Train index duplicates: {train_times.index.duplicated().sum()}")
		print(f"Val index duplicates: {val_times.index.duplicated().sum()}")
		print(f"Test index duplicates: {test_times.index.duplicated().sum()}")

		# ------------------------------------------------------------------
		# 9. ALIGN MERGED DICTIONARY WITH TRAIN/VAL/TEST SETS
		# ------------------------------------------------------------------
		ampere_dates = pd.Series(np.nan, index=pd.to_datetime(list(merged_dict.keys())))
		print(f'Total AMPERE duplicates {ampere_dates.index.duplicated().sum()}')

		# Use intersection between sequence timestamps & AMPERE timestamps
		train_dates = pd.concat([train_times.to_frame(), ampere_dates.to_frame()], axis=1, join='inner').index
		val_dates = pd.concat([val_times.to_frame(), ampere_dates.to_frame()], axis=1, join='inner').index
		test_dates = pd.concat([test_times.to_frame(), ampere_dates.to_frame()], axis=1, join='inner').index

		# Convert to string format expected by dictionary keys
		train_dates = train_dates.strftime(self.datetime_format)
		val_dates = val_dates.strftime(self.datetime_format)
		test_dates = test_dates.strftime(self.datetime_format)

		# Dictionary slicing (VERY fast)
		train = {k: merged_dict[k] for k in train_dates}
		val = {k: merged_dict[k] for k in val_dates}
		test = {k: merged_dict[k] for k in test_dates}

		# ------------------------------------------------------------------
		# 10. FIT SCALER ON TRAINING INPUT SEQUENCES
		# ------------------------------------------------------------------

		# Stack all input arrays into one matrix for fitting (fastest approach)
		scaling_array = np.vstack([sample["input"] for sample in train.values()])
		print(f"Scaling array shape: {scaling_array.shape}")

		scaler = StandardScaler()
		scaler.fit(scaling_array)

		# ------------------------------------------------------------------
		# 11. APPLY SCALING TO TRAIN / VAL / TEST
		# ------------------------------------------------------------------

		def scale_dict(d, label):
			for key in tqdm.tqdm(d, desc=f"Scaling {label}"):
				# scale input only (ampere is target)
				d[key]["input"] = scaler.transform(d[key]["input"])
				d[key]["ampere"] = np.array(d[key]["ampere"])

		scale_dict(train, "training")
		scale_dict(val, "validation")
		scale_dict(test, "testing")


		# Save scaler for inference
		with open(f"{self.data_dir}prepared_data/scaler_{self.eras}_{self.data_version}.pkl", "wb") as f:
			pickle.dump(scaler, f)

		return train, val, test


	def __call__(self):
		'''
		Calling the data prep class without the TWINS data for this version of the model.

		Returns:
			train, val, test (dicts): dictionaries containing the training, validation and testing data

		'''

		if os.path.exists(self.data_dir+f'prepared_data/fully_prepared_{self.eras}_{self.data_version}_disturbed_time_{self.extract_storms}.pkl'):
			print('Loading pre-processed data....')
			with open(self.data_dir+f'prepared_data/fully_prepared_{self.eras}_{self.data_version}_disturbed_time_{self.extract_storms}.pkl', 'rb') as f:
				data = pickle.load(f)
			train = data['train']
			val = data['val']
			test = data['test']


		else:

			print(f'Prepared data not found. Beginning data preparation for version {self.data_version}....')

			train, val, test = self.processing()

			print('Data processing complete... Saving results....')

			with open(self.data_dir+f'prepared_data/fully_prepared_{self.eras}_{self.data_version}_disturbed_time_{self.extract_storms}.pkl', 'wb') as f:
				pickle.dump({'train':train, 'val':val, 'test':test}, f)


		return train, val, test
