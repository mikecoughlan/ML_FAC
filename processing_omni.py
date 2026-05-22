############################################################################################
#
#	multi-station-dbdt-risk-assessment/preparing_SW_data.py
#
#	File for preparing the raw solar wind data from both ACE and OMNI. Takes the source
# 	files, up or down samples the ACE data as necessary to the 1-minute resoultion. Changes
# 	missing data format from eg. 999.999 to np.nan. Interpolates up to 15 minutes of missing
# 	data. Saves the data as an external file for use in later scripts.
#
#	SCRIPT ADAPTED FROM SIMILAR SCRIPT WRITTEN BY VICTOR A. PINTO
############################################################################################

# importing relevent packages
import glob
import os

import numpy as np
import pandas as pd

os.environ["CDF_LIB"] = "~/lib"

import cdflib

# muting some pandas warnings
pd.options.mode.chained_assignment = None

# defining reletive file paths
omniDir = os.path.expanduser('~/../../Volumes/TARDIS/AIMFAHR/sw_data/omni/')	# path to the omni data

method = 'linear'	# defining the interpolation method
limit = 10			# defining the limit of interploation


def break_dates(df:pd.DataFrame(), dateField:str, drop:bool=False, errors:str="raise"):
	'''
	Break_dates expands a column of df from a datetime64 to many columns containing
	the information from the date. This applies changes inplace.

	Args:
		df (pd.dataframe): df gain several new columns.
		dateField (string): A string that is the name of the date column you wish to
							expand. If it is not a datetime64 series, it will be converted
							to one with pd.to_datetime.
		drop (bool, optional): If true then the original date column will be removed.
								Defaults to False.
		errors (str, optional): if raise, will raise an error if present during datatime
								conversion. Defaults to "raise".

	Modified from FastAI software by Victor Pinto.
	'''

	field = df[dateField]
	field_dtype = field.dtype
	if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
		field_dtype = np.datetime64

	if not np.issubdtype(field_dtype, np.datetime64):
		df[dateField] = field = pd.to_datetime(field, infer_datetime_format=True, errors=errors)

	attr = ['Year', 'Month', 'Day', 'Dayofyear', 'Hour', 'Minute']

	for n in attr: df[n] = getattr(field.dt, n.lower())
	if drop: df.drop(dateField, axis=1, inplace=True)


def omnicdf2dataframe(file:str) -> pd.DataFrame():
	'''
	Load a CDF File and convert it in a Pandas DataFrame.

	WARNING: This will not return the CDF Attributes, just the variables.
	WARNING: Only works for CDFs of the same array length (OMNI)

	Args:
		file (cdf file): file input for conversion to a pd.dataframe

	Returns:
		pd.dataframe: cdf file converted to a pd.dataframe. Contains
						a datetime column named "Epoch".
	'''

	cdf = cdflib.CDF(file)
	cdfdict = {}

	for key in cdf.cdf_info().zVariables:
		cdfdict[key] = cdf[key]

	cdfdf = pd.DataFrame(cdfdict)

	for col in cdfdf.columns:
		cdfdf[col] = clean_omni(cdfdf[col], cdf.attget('FILLVAL',col).Data)

	if 'Epoch' in cdf.cdf_info().zVariables:
		cdfdf['Epoch'] = pd.to_datetime(cdflib.cdfepoch.encode(cdfdf['Epoch'].values))

	return cdfdf

def clean_omni(var:pd.Series(), fill_value:float) -> pd.Series():
	'''
	Remove filling numbers for missing data in OMNI data (1 min) and replace
	them with np.nan values.

	Args:
		df (pd.dataframe): dataframe containing OMNI data to be cleaned.

	Returns:
		pd.dataframe: cleaned dataframe.
	'''
	var.loc[var == fill_value] = np.nan
	var.interpolate(method=method, limit=limit)

	return var


def processing_omni() -> pd.DataFrame():
	'''
	Gets the AE_INDEX and the SYM_H indicies from the OMNI data as this is not contained in the ACE data.

	Returns:
		pd.Dataframe: pd.dataframe with the indicies and a column labeled Epoch containing teh datatime stamp
	'''

	# defining the beginning and ending years. Defined by the data available from ACE.
	syear = 2009
	eyear = 2025

	############################################################################################
	####### Load and pre-process solar wind data
	############################################################################################

	omniFiles = glob.glob(omniDir+'hro2_1min/*.cdf', recursive=True) # getting file names
	print(f'Number of OMNI files found: {len(omniFiles)}')

	# creating list of dataframes
	o = []
	for fil in sorted(omniFiles):
		cdf = omnicdf2dataframe(fil)
		o.append(cdf)

	omni_start_time = str(pd.Timestamp(syear,1,1))
	omni_start_time = omni_start_time.replace(' ', '').replace('-', '').replace(':', '')
	omni_end_time = str(pd.Timestamp(eyear,12,31,23,59,59))
	omni_end_time = omni_end_time.replace(' ', '').replace('-', '').replace(':', '')

	# combining the yearly dataframes into one large dataframe
	omniData = pd.concat(o, axis=0, ignore_index=True)
	# setting the index to a datetime index
	omniData.index = omniData.Epoch
	# trimming the dataframe to be in the time frame of interest
	omniData = omniData[omni_start_time:omni_end_time]

	to_drop = ['PLS', 'IMF_PTS', 'PLS_PTS', 'percent_interp',
			'Timeshift', 'RMS_Timeshift', 'RMS_phase', 'Time_btwn_obs',
			'RMS_SD_B', 'RMS_SD_fld_vec',
			'BY_GSE', 'BZ_GSE', 'F',
			'flow_speed', 'E', 'Beta', 'Mach_num',
			'Mgs_mach_num', 'Epoch', 'YR', 'Day', 'HR', 'Minute']

	# dropping unnecessary columns
	omniData = omniData.drop(to_drop, axis=1)

	return omniData


def main():
	'''
	Main function calling both the indicies and the ACE data processing functions.
	Saves the individual ACE and OMNI data as well as the combined dataset.
	'''
	print('Entering main of preparing SW')

	omniData = processing_omni()

	omniData.to_feather(omniDir+f'omni_{limit}_min_interp.feather')


if __name__ == '__main__':

	main()

	print('It ran. Good job!')