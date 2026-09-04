"""
utils.py
========

Shared helpers with no project-internal dependencies.

This module is deliberately kept free of torch and matplotlib imports so
it stays cheap to import from data-prep, training, inference and
notebook contexts alike. Plotting helpers belong elsewhere.

Contents
--------
load_config                Merge shared + per-model settings from config.json.
Filenames / cache_meta_*   Canonical output filenames and cache provenance.
MODEL_NAMES                Valid model selectors.
day_of_year_to_month_day   AMPERE day-of-year timestamps -> datetime string.
unpacking_current_density  Read AMPERE netCDF FAC files into a dict keyed
                           by timestamp.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
from tqdm import tqdm

# Default location of the consolidated config file.
DEFAULT_CONFIG_PATH = 'config.json'

# The two models this project ships. 'sci' is the science model, which
# uses the full input set including SuperMAG indices; 'op' is the
# operational model, restricted to inputs available in real time.
MODEL_NAMES = ('sci', 'op')


def load_config(model=None, config_path=DEFAULT_CONFIG_PATH):
	"""
	Build the effective configuration for one model.

	config.json is organised as a 'shared' block plus one block per
	model. The effective config is 'shared' updated with the chosen
	model's block, so a model may override any shared setting while the
	common settings are stated only once.

	Two extra keys are added to the returned dictionary: 'model_name',
	the selector used, and 'config_path', where it came from. Both are
	used to build cache filenames and to report which configuration
	produced a given result.

	Parameters
	----------
	model : {'sci', 'op'}, optional
		Which model to configure. Case-insensitive, so 'Sci' and 'sci'
		are equivalent. Defaults to the file's 'active_model'.
	config_path : str or Path
		Path to the consolidated config file.

	Returns
	-------
	dict
		The merged configuration.

	Raises
	------
	FileNotFoundError
		If config_path does not exist.
	ValueError
		If `model` is not one of MODEL_NAMES, or the file is missing a
		block it needs.

	Examples
	--------
	>>> sci = load_config('sci')
	>>> op = load_config('op')
	"""
	config_path = Path(config_path)

	if not config_path.exists():
		raise FileNotFoundError(
			f'Config file not found: {config_path}. Expected config.json in '
			'the working directory, or an explicit path.'
		)

	with open(config_path, 'r') as f:
		raw = json.load(f)

	# Fall back to the file's own default when no model is named.
	if model is None:
		model = raw.get('active_model')
		if model is None:
			raise ValueError(
				f"No model given and {config_path} has no 'active_model' key. "
				f'Pass one of {MODEL_NAMES}.'
			)

	model = str(model).strip().lower()
	if model not in MODEL_NAMES:
		raise ValueError(
			f'Unknown model {model!r}. Expected one of {MODEL_NAMES}.'
		)

	if 'shared' not in raw:
		raise ValueError(f"{config_path} is missing its 'shared' block.")
	if model not in raw:
		raise ValueError(f"{config_path} is missing a {model!r} block.")

	# Shallow merge: a model block replaces a shared key outright rather
	# than merging into it. That keeps model_config unambiguous -- each
	# model states its architecture in full, so no shared default can
	# leak into it unnoticed.
	config = dict(raw['shared'])
	config.update(raw[model])

	# Recorded so downstream code can name the model in cache filenames
	# and log which configuration was used.
	config['model_name'] = model
	config['config_path'] = str(config_path)

	return config


# ---------------------------------------------------------------------------
# Canonical filenames
# ---------------------------------------------------------------------------
#
# One place defining every file the pipeline reads or writes. Names are
# plain and carry only the model they belong to -- no version tags, era
# suffixes or boolean flags baked into the string.
#
# The cost of dropping those suffixes is that two runs differing only in
# a setting like `input_params` now write to the SAME filename, so a cache
# from one silently satisfies the other. cache_meta_write / cache_meta_check
# below close that gap: each cache gets a small .json sidecar recording the
# settings that produced it, and loading warns loudly on a mismatch.
#



def model_file(config):
	"""Trained weights for a model, e.g. 'models/acorn_sci.pt'."""
	return str(Path(config.get('model_dir', 'models/')) /
				f"acorn_{config['model_name']}.pt")


def scaler_file(config):
	"""Fitted input scaler, e.g. 'data/prepared/scaler_sci.pkl'."""
	return str(Path('models') / 'scalers' /
				f"scaler_{config['model_name']}.pkl")


def prepared_file(config):
	"""Fully prepared train/val/test data, e.g. 'data/prepared/prepared_sci.pkl'."""
	return str(Path(config.get('data_dir', 'data/')) / 'prepared' /
				f"prepared_{config['model_name']}.pkl")


def sequences_file(config):
	"""Split input sequences, e.g. 'data/prepared/sequences_sci.pkl'."""
	return str(Path(config.get('data_dir', 'data/')) / 'prepared' /
				f"sequences_{config['model_name']}.pkl")


def ampere_file(config):
	"""Cached AMPERE targets, e.g. 'data/prepared/ampere.pkl'.

	Model-independent: the targets are the same whichever model consumes
	them, so no model name appears here. Written on the first run from
	the raw netCDF files and reused thereafter.
	"""
	return str(Path(config.get('data_dir', 'data/')) / 'prepared' / 'ampere.pkl')


def results_file(config):
	"""Test-set predictions, e.g. 'outputs/results_sci.pkl'."""
	return str(Path(config.get('outputs_dir', 'outputs/')) /
				f"results_{config['model_name']}.pkl")


def loss_file(config):
	"""Per-epoch loss history, e.g. 'loss_tracker/loss_sci.feather'."""
	return str(Path('loss_tracker') / f"loss_{config['model_name']}.feather")


def shap_file(config, explainer, tag):
	"""SHAP attributions, e.g. 'shap/shap_sci_gradient_R1_dawn.pkl'."""
	return str(Path('shap') /
				f"shap_{config['model_name']}_{explainer}_{tag}.pkl")


# Settings that change a cache's CONTENTS without changing its filename.
# Recorded in the sidecar so a mismatch can be detected.
CACHE_KEYS = ('model_name', 'input_params', 'time_history', 'ampere_delay')


def cache_meta_write(path, config):
	"""
	Write a .json sidecar recording the settings behind a cache file.

	Called right after a cache is written. Failures are non-fatal: a
	missing sidecar costs a warning later, not a crash now.
	"""
	meta = {k: config.get(k) for k in CACHE_KEYS}
	try:
		with open(str(path) + '.json', 'w') as f:
			json.dump(meta, f, indent=2)
	except OSError as e:
		print(f'WARNING: could not write cache metadata for {path}: {e}')


def cache_meta_check(path, config):
	"""
	Compare a cache's sidecar against the current config.

	Returns True if the cache matches (or cannot be checked), False if it
	was built with different settings. A False result means the cache on
	disk answers a different question than the one being asked -- delete
	it and re-run rather than trusting it.
	"""
	sidecar = Path(str(path) + '.json')
	if not sidecar.exists():
		# Pre-dates the sidecar convention. Say so rather than failing.
		print(f'NOTE: no metadata sidecar for {path}; cannot verify it '
				'matches the current config.')
		return True

	with open(sidecar, 'r') as f:
		meta = json.load(f)

	differences = {k: (meta.get(k), config.get(k))
					for k in CACHE_KEYS if meta.get(k) != config.get(k)}

	if differences:
		print(f'WARNING: cache {path} was built with different settings:')
		for k, (was, now) in differences.items():
			print(f'    {k}: cached={was!r}  current={now!r}')
		print('    Delete the cache and re-run, or results will not match '
				'the current config.')
		return False

	return True


def day_of_year_to_month_day(year, day_of_year, fractional_hour):
	"""
	Convert an AMPERE (year, day-of-year, fractional-hour) triple to a
	timestamp string.

	AMPERE netCDF files store time as three separate variables rather
	than a single epoch, so this reassembles them into the
	'YYYY-MM-DD HH:MM:SS' strings used as dictionary keys throughout the
	pipeline. Leap years are handled implicitly by timedelta arithmetic.

	Seconds are always zero: AMPERE products are on a 2-minute cadence
	and the fractional hour resolves cleanly to whole minutes.

	Parameters
	----------
	year : int
		Year (e.g., 2023)
	day_of_year : int
		Day of year (1-365, or 1-366 for leap years)
	fractional_hour : float
		Hour of day as a decimal (e.g., 14.5 for 14:30)

	Returns
	-------
	str
		Date and time in 'YYYY-MM-DD HH:MM:SS' string format
	"""
	hours = int(fractional_hour)
	minutes = int((fractional_hour * 60) % 60)

	date = (datetime(year, 1, 1)
			+ timedelta(days=int(day_of_year) - 1)
			+ timedelta(hours=hours, minutes=minutes, seconds=int(0)))

	# Built by hand rather than with strftime so the hour/minute fields
	# come from the fractional-hour arithmetic above, not from `date`.
	return (str(date.year) + '-' + str(date.month).zfill(2) + '-'
			+ str(date.day).zfill(2) + ' ' + str(hours).zfill(2) + ':'
			+ str(minutes).zfill(2) + ':' + str(int(0)).zfill(2))


def unpacking_current_density(file, pivot_or_array='pivot'):
	"""
	Read field-aligned current density from an AMPERE netCDF file.

	Each record in the file is one time step holding flat arrays of jPar
	together with the mlt/colatitude coordinate of every sample. Two
	output forms are offered:

	  'pivot' -- a DataFrame pivoted to (lat x mlt), i.e. the physical
	             grid layout. Use this to plot or to compare against
	             model output cell by cell.
	  'array' -- the raw flat jPar vector, coordinates discarded. Use
	             this only when the caller already knows the layout.

	Fill values
	-----------
	AMPERE marks missing samples with values of magnitude ~1e30. These
	are replaced with NaN before anything else touches the data; left in
	place they would dominate any mean, scaler or plot colour range they
	reached.

	Note on the flat form
	---------------------
	Elsewhere in this project AMPERE arrays cached in pickles come back
	flat with shape (1200,) and must be restored with
	`.reshape(24, 50).T`, reshaping as (50, 24)
	silently transposes the grid and is a recurring source of bugs. The
	'pivot' option avoids the issue entirely by carrying the coordinates
	through.

	Parameters
	----------
	file : str or Path
		Path to the AMPERE netCDF file.
	pivot_or_array : {'pivot', 'array'}
		'pivot' for a 2D (lat x mlt) DataFrame, 'array' for the flat 1D
		numpy array.

	Returns
	-------
	current_density_dict : dict
		Dictionary mapping 'YYYY-MM-DD HH:MM:SS' timestamp strings to
		current density data in the requested form. Empty if the file is
		missing an expected variable.

	Raises
	------
	ValueError
		If pivot_or_array is neither 'pivot' nor 'array'.
	"""
	file = Path(file)
	current_density_dict = {}

	try:
		cdf = nc.Dataset(file)

		# Pull each variable out once rather than re-indexing the netCDF
		# object inside the loop, which is markedly slower.
		years = cdf.variables["year"][:]
		doys = cdf.variables["doy"][:]
		times = cdf.variables["time"][:]      # fractional hours
		jpar = cdf.variables["jPar"][:]       # shape: (records, points)
		mlt = cdf.variables["mlt_hr"][:]      # shape: (records, points)
		lat = cdf.variables["cLat_deg"][:]    # shape: (records, points)

		# Mask AMPERE's fill values, which appear at both signs.
		jpar[jpar > 1e30] = np.nan
		jpar[jpar < -1e30] = np.nan

		for record in tqdm(range(len(times)),
						desc=f'Unpacking {file.name}', unit='step', leave=False):

			# Rebuild the timestamp from the separate year/doy/time vars.
			timestamp = day_of_year_to_month_day(
				years[record], doys[record], times[record]
			)

			if pivot_or_array == "array":
				current_density_dict[timestamp] = np.array(jpar[record, :])

			elif pivot_or_array == "pivot":
				# pivot_table reconstructs the (lat x mlt) grid from the
				# flat jPar/mlt/lat triples stored for this record.
				current_density_dict[timestamp] = pd.DataFrame({
					"current_density": jpar[record, :],
					"mlt": mlt[record, :],
					"lat": lat[record, :],
				}).pivot_table(index="lat", columns="mlt", values="current_density")

			else:
				raise ValueError("pivot_or_array must be either 'pivot' or 'array'")

	except KeyError as e:
		# Some AMPERE files are truncated or missing a variable. Skip
		# them rather than aborting a multi-year ingest, but say so.
		print(f"KeyError {e} encountered in file {file}. Skipping file.")

	return current_density_dict
