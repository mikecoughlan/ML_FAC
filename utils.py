import pandas as pd
import numpy as np
import os
import json
import pickle
import glob
import netCDF4 as nc
from datetime import datetime, timedelta


def day_of_year_to_month_day(year, day_of_year, fractional_hour):
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

	date = datetime(year, 1, 1) + timedelta(days=int(day_of_year) - 1) + timedelta(hours=hours, minutes=minutes, seconds=int(0))
	return str(date.year) + '-' + str(date.month).zfill(2) + '-' + str(date.day).zfill(2) + ' ' + str(hours).zfill(2) + ':' + str(minutes).zfill(2) + ':' + str(int(0)).zfill(2)

def unpacking_current_density(file, pivot_or_array='pivot'):

	"""
	Unpack current density data from a netCDF file.

	Parameters
	----------
	file : str
		Path to the netCDF file.
	pivot_or_array : str
		Specifies the format of the output data.
		Options are 'pivot' for a 2D pivot table DataFrame or 'array' for a 1D numpy array.

	Returns
	-------
	current_density_dict : dict
		Dictionary with datetime strings as keys and current density data as values.
	"""

	cdf = nc.Dataset(file)
	data_point_time = cdf.variables['time'][:]
	current_density_dict = {}
	for hour in range(len(data_point_time)):

		time_obj = day_of_year_to_month_day(cdf.variables['year'][:][hour],
											cdf.variables['doy'][:][hour],
											cdf.variables['time'][:][hour])

		if pivot_or_array == 'array':
			current_density_dict[time_obj] = np.array(cdf.variables['jPar'][:][hour,:])
		elif pivot_or_array == 'pivot':
			current_density_dict[time_obj] = pd.DataFrame({'current_density':cdf.variables['jPar'][:][hour,:],
														'mlt':cdf.variables['mlt_hr'][hour,:],
														'lat':cdf.variables['cLat_deg'][hour,:]}).pivot_table(index='lat',
														columns='mlt', values='current_density')
		else:
			raise ValueError("pivot_or_array must be either 'pivot' or 'array'")

	return current_density_dict