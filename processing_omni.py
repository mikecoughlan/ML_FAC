"""
processing_omni.py
==================

Bulk pre-processing of raw OMNI 1-minute solar wind CDFs into a single
feather file for downstream training and evaluation.

Reads every yearly `hro2_1min` CDF under the OMNI directory, converts
each to a DataFrame, replaces the per-variable fill values (e.g.
9999.99) with NaN, drops columns the ML pipeline does not consume, trims
to the configured year range, and writes one concatenated feather file.

This is an offline, run-once-per-data-refresh script. It is NOT used at
inference time -- inference.py fetches and parses OMNI CDFs on demand
from NASA SPDF with its own reader (`_read_omni_cdf`). Be aware the two
readers use different fill-value strategies: this file reads FILLVAL
from each variable's CDF attributes, while inference.py uses a
hardcoded `_OMNI_FILLVALS` table with a 99999.0 fallback.

Adapted from a script by Victor A. Pinto.

Configuration
-------------
Set the OMNI source directory with the AIMFAHR_OMNI_DIR environment
variable, or pass `omni_dir` explicitly to processing_omni(). It should
contain an `hro2_1min/` subdirectory of yearly .cdf files.
"""

import glob
import os

import cdflib
import numpy as np
import pandas as pd
from tqdm import tqdm

# muting some pandas warnings
pd.options.mode.chained_assignment = None

# Source directory for the raw OMNI CDFs. Overridable via environment so
# no machine-specific path is baked into the repository.
OMNI_DIR = os.path.expanduser(os.environ.get('AIMFAHR_OMNI_DIR', 'data/omni/'))

# Interpolation settings for clean_omni(). LIMIT also appears in the
# output filename, so changing it changes the file that data_prep.py
# expects to read.
METHOD = 'linear'   # interpolation method
LIMIT = 10          # max consecutive NaNs to fill across a gap

# Year range to retain in the combined output.
START_YEAR = 2009
END_YEAR = 2025

# Columns retained in the combined output.
#
# This is an allow-list rather than a drop-list so the contents of the
# feather file are stated explicitly here, instead of being whatever the
# CDF happened to contain minus a few removals. Adding a variable to the
# model means adding it here first.
#
# Anything present in the CDF but absent from this list is dropped
# silently; processing_omni() reports both directions so the list can be
# checked against real files.
TO_KEEP = [
	# --- Directly consumed as model inputs (see config input_params) ---
	'BX_GSE',           # IMF Bx, GSE
	'BY_GSM',           # IMF By, GSM
	'BZ_GSM',           # IMF Bz, GSM
	'Vx',               # solar wind velocity, x component
	'proton_density',   # solar wind proton density
	'SYM_H',            # ring current index
	'ASY_H',            # asymmetric ring current index

	# --- Auroral electrojet indices ---
	# Retained as candidate inputs. Note the Sci model uses the SuperMAG
	# SML/SMU indices instead, merged in later by data_prep.py rather
	# than sourced here.
	'AE_INDEX',
	'AL_INDEX',
	'AU_INDEX',
	'PC_N_INDEX',       # polar cap north index

	# --- Additional solar wind state ---
	# Not currently model inputs; kept so alternative input sets can be
	# tried without regenerating the feather file.
	'Vy',
	'Vz',
	'T',                # proton temperature
	'Pressure',         # solar wind flow pressure
	'SYM_D',
	'ASY_D',

	# --- Spacecraft and bow shock geometry ---
	'IMF',              # source spacecraft ID for the IMF data
	'x', 'y', 'z',              # spacecraft position, GSE
	'BSN_x', 'BSN_y', 'BSN_z',  # bow shock nose position

	# --- Energetic particle flux ---
	'Proton_flux_10_MeV',
	'Proton_flux_30_MeV',
	'Proton_flux_60_MeV',
	'Flux_FLAG',
]


def omnicdf2dataframe(file: str) -> pd.DataFrame:
	'''
	Load a CDF file and convert it to a pandas DataFrame.

	WARNING: This will not return the CDF attributes, just the variables.
	WARNING: Only works for CDFs whose variables all share one array
			 length (true of OMNI, not of CDFs generally).

	Fill values are resolved per-variable from each variable's own
	FILLVAL attribute, so this adapts automatically if OMNI changes a
	fill convention.

	Args:
		file (str): path to the CDF file to convert.

	Returns:
		pd.DataFrame: CDF contents, with an 'Epoch' datetime column.
	'''

	cdf = cdflib.CDF(file)
	cdfdict = {}

	for key in cdf.cdf_info().zVariables:
		cdfdict[key] = cdf[key]

	cdfdf = pd.DataFrame(cdfdict)

	# Replace each column's fill value with NaN, using that column's own
	# declared FILLVAL rather than a global sentinel.
	for col in tqdm(cdfdf.columns, desc=f'Cleaning {os.path.basename(file)}',
					unit='col', leave=False):
		cdfdf[col] = clean_omni(cdfdf[col], cdf.attget('FILLVAL', col).Data)

	# CDF epochs are an internal numeric format; decode to datetime64.
	if 'Epoch' in cdf.cdf_info().zVariables:
		cdfdf['Epoch'] = pd.to_datetime(cdflib.cdfepoch.encode(cdfdf['Epoch'].values))

	return cdfdf


def clean_omni(var: pd.Series, fill_value: float) -> pd.Series:
	'''
	Replace OMNI fill values with np.nan.

	Short gaps are then filled by linear interpolation, up to LIMIT
	consecutive missing minutes. Gaps longer than that are left as NaN
	for data_prep.py to handle, on the grounds that interpolating across
	a long dropout would invent solar wind structure that was never
	measured.

	Args:
		var (pd.Series): a single OMNI variable.
		fill_value (float): the sentinel marking missing data.

	Returns:
		pd.Series: the series with fill values replaced by NaN.
	'''
	var.loc[var == fill_value] = np.nan
	var.interpolate(method=METHOD, limit=LIMIT, inplace=True)

	return var


def processing_omni(omni_dir: str = None) -> pd.DataFrame:
	'''
	Read every yearly OMNI CDF, combine, trim and drop unused columns.

	Args:
		omni_dir (str, optional): directory holding `hro2_1min/*.cdf`.
			Defaults to OMNI_DIR.

	Returns:
		pd.DataFrame: combined OMNI data indexed by datetime, restricted
			to START_YEAR..END_YEAR.
	'''

	if omni_dir is None:
		omni_dir = OMNI_DIR

	omniFiles = glob.glob(os.path.join(omni_dir, 'hro2_1min/*.cdf'), recursive=True)
	print(f'Number of OMNI files found: {len(omniFiles)}')

	# Sorted so the concatenated frame is chronological.
	o = []
	for fil in tqdm(sorted(omniFiles), desc='Reading OMNI CDFs', unit='file'):
		cdf = omnicdf2dataframe(fil)
		o.append(cdf)

	# Timestamps are stripped to bare digits because the trim below uses
	# pandas' string-based datetime index slicing.
	omni_start_time = str(pd.Timestamp(START_YEAR, 1, 1))
	omni_start_time = omni_start_time.replace(' ', '').replace('-', '').replace(':', '')
	omni_end_time = str(pd.Timestamp(END_YEAR, 12, 31, 23, 59, 59))
	omni_end_time = omni_end_time.replace(' ', '').replace('-', '').replace(':', '')

	omniData = pd.concat(o, axis=0, ignore_index=True)
	omniData.index = omniData.Epoch
	omniData = omniData[omni_start_time:omni_end_time]

	# Reduce to the allow-list. Epoch is deliberately absent from
	# TO_KEEP: it has already become the index above, so keeping the
	# column too would duplicate it.
	present = [c for c in TO_KEEP if c in omniData.columns]
	missing = [c for c in TO_KEEP if c not in omniData.columns]
	dropped = [c for c in omniData.columns if c not in TO_KEEP]

	if missing:
		print(f'WARNING: {len(missing)} column(s) in TO_KEEP not found in the '
				f'OMNI files and will be absent from the output: {missing}')
	print(f'Keeping {len(present)} column(s); dropping {len(dropped)}: {dropped}')

	omniData = omniData[present]

	return omniData


def main():
	'''
	Process the raw OMNI CDFs and write the combined feather file.
	'''
	print('Entering main of preparing SW')

	omniData = processing_omni()

	omniData.to_feather(os.path.join(OMNI_DIR, f'omni_{LIMIT}_min_interp.feather'))


if __name__ == '__main__':

	main()

	print('It ran. Good job!')
