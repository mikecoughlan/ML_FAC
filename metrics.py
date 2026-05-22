import glob
import os
import pandas as pd
import numpy as np
import netCDF4
import pickle
import matplotlib.pyplot as plt
import utils
import datetime
import matplotlib as mpl
import gc
import glob
import netCDF4
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from custom_loss_functions import CRPS
from model_classes import *
from sklearn.metrics import mean_absolute_error as MAE

# with open('outputs/FAC_BK_1_3_both_results.pkl', 'rb') as f:
# 	both_results = pickle.load(f)

# with open('outputs/FAC_BK_1_3_block_1_results.pkl', 'rb') as f:
# 	block_1_results = pickle.load(f)

with open('outputs/FAC_BK_1_3_next_results.pkl', 'rb') as f:
	next_results = pickle.load(f)

# with open('outputs/FAC_BK_1_3_storms_noise_filtered_both_storm_training_True_results.pkl', 'rb') as f:
# 	both_noise_results = pickle.load(f)

# with open('outputs/FAC_BK_1_3_storms_noise_filtered_block_1_storm_training_True_results.pkl', 'rb') as f:
# 	block_1_noise_results = pickle.load(f)

with open('outputs/FAC_BK_1_3_storms_noise_filtered_next_storm_training_True_results.pkl', 'rb') as f:
	next_noise_results = pickle.load(f)

with open('outputs/FAC_BK_1_3_sme_next_storm_training_True_results.pkl', 'rb') as f:
	next_storm_results = pickle.load(f)

def plotting_real_vs_predicted(y_pred, y_true, title, mae):

	plt.figure(figsize=(8,8))
	plt.hist2d(y_true, y_pred, bins=100, cmap='Blues', norm=mpl.colors.LogNorm())
	# Fitting a line to the data and labeling it with equation of line
	m, b = np.polyfit(y_true, y_pred, 1)
	plt.plot(y_true, m*y_true + b, color='blue', label=f'YFit: y={m:.2f}x+{b:.2f}')
	plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--', lw=2, color='red')
	plt.xlabel('Measured Current Density (uA/m²)')
	plt.ylabel('Predicted Current Density (uA/m²)')
	plt.title(f'{title} - Corr: {np.corrcoef(y_true, y_pred)[0,1]:.2f}')
	plt.legend(loc='upper left')
	plt.text(0.05, 0.95, f'MAE: {mae:.2f} uA/m²', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
	plt.grid()
	plt.savefig(f'plots/regression_{title.replace(" ","_")}_real_vs_predicted.png')

def calculate_metrics(dict, title):
	y_pred = np.array(list(dict[key]['predicted'].flatten() for key in dict.keys()), dtype=np.float32)
	y_true = np.array(list(dict[key]['ampere'].flatten() for key in dict.keys()), dtype=np.float32)
	y_pred = y_pred
	#turning everything below plus or minus 0.1 to 0
	# y_pred[np.abs(y_pred)<0.1] = 0
	# y_true[np.abs(y_true)<0.1] = 0

	mae = MAE(y_true,y_pred)
	plotting_real_vs_predicted(y_pred.flatten(), y_true.flatten(), title, mae)
	mean = np.mean(np.abs(y_true))
	std = np.std(y_true)
	# corr = np.corrcoef(y_true,y_pred)

	return mae, mean, std

# both_block_1_keys = [key for key in both_results.keys() if pd.to_datetime(key).year < 2018]
# both_next_keys = [key for key in both_results.keys() if pd.to_datetime(key).year > 2018]

# both_block_1_noise_keys = [key for key in both_noise_results.keys() if pd.to_datetime(key).year < 2018]
# both_next_noise_keys = [key for key in both_noise_results.keys() if pd.to_datetime(key).year > 2018]

# both_next = {key: both_results[key] for key in both_next_keys}
# both_block_1 = {key: both_results[key] for key in both_block_1_keys}
# both_noise_next = {key: both_noise_results[key] for key in both_next_noise_keys}
# both_noise_block_1 = {key: both_noise_results[key] for key in both_block_1_noise_keys}

# block_1_mae = calculate_metrics(block_1_results, 'Block 1 Results')
# print(f'Block 1 - MAE: {block_1_mae[0]} mean: {block_1_mae[1]} std: {block_1_mae[2]}')
# both_block_1_mae = calculate_metrics(both_block_1, 'Both (Block 1 period) Results')
# print(f'Both (Block 1 period) - MAE: {both_block_1_mae[0]} mean: {both_block_1_mae[1]} std: {both_block_1_mae[2]}')

# block_1_noise_mae = calculate_metrics(block_1_noise_results, 'Block 1 noise Results')
# print(f'Block 1 noise - MAE: {block_1_noise_mae[0]} mean: {block_1_noise_mae[1]} std: {block_1_noise_mae[2]}')
# both_block_1_noise_mae = calculate_metrics(both_noise_block_1, 'Both (Block 1 period) noise Results')
# print(f'Both (Block 1 period) noise - MAE: {both_block_1_noise_mae[0]} mean: {both_block_1_noise_mae[1]} std: {both_block_1_noise_mae[2]}')


next_mae = calculate_metrics(next_results, 'Next Results')
print(f'\nNext - MAE: {next_mae[0]} mean: {next_mae[1]} std: {next_mae[2]}')
# both_next_mae = calculate_metrics(both_next, 'Both (next period) Results')
# print(f'Both (next period) - MAE: {both_next_mae[0]} mean: {both_next_mae[1]} std: {both_next_mae[2]}')

next_noise_mae = calculate_metrics(next_noise_results, 'Next noise Results')
print(f'\nNext noise - MAE: {next_noise_mae[0]} mean: {next_noise_mae[1]} std: {next_noise_mae[2]}')
# both_next_noise_mae = calculate_metrics(both_noise_next, 'Both (next period) noise Results')
# print(f'Both (next period) noise - MAE: {both_next_noise_mae[0]} mean: {both_next_noise_mae[1]} std: {both_next_noise_mae[2]}')
next_storm_mae = calculate_metrics(next_storm_results, 'Next disturbed time Results')
print(f'\nNext disturbed time - MAE: {next_storm_mae[0]} mean: {next_storm_mae[1]} std: {next_storm_mae[2]}')
