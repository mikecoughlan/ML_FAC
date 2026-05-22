####################################################################################
#
# model_training.py
#
# Pulls data from data_prep.py, uses models defined in model_classes.py, custom loss
# functions from custom_loss_functions.py. Trains a machine learning model and
# provides a simple evaluation of performance. Saves the testing data in a dict for
# further analysis.
#
####################################################################################


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
from typing import List

import matplotlib
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

import utils
from custom_loss_functions import (CRPS, WeightedCRPS,
                                   WeightedMeanSquaredError,
                                   create_bin_weights)
from data_prep import PreparingData
from model_classes_test import *

pd.options.mode.chained_assignment = None

working_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(working_dir)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# Loading CONFIG json file
with open('opp_config.json', 'r') as f:
	CONFIG = json.load(f)

if not os.path.exists(CONFIG["model_dir"]):
	os.makedirs(CONFIG["model_dir"])


model_file = f'{CONFIG["model_dir"]}{CONFIG["model"]}_{CONFIG["version"]}_{CONFIG["eras"]}.pt'


class Early_Stopping():
	'''
	Class to create an early stopping condition for the model.

	'''

	def __init__(self, decreasing_loss_patience=25, model_config=CONFIG['model_config']):
		'''
		Initializing the class.

		Args:
			decreasing_loss_patience (int): the number of epochs to wait before stopping the model if the validation loss does not decrease
			pretraining (bool): whether the model is being pre-trained. Just used for saving model names.

		'''

		# initializing the variables
		self.decreasing_loss_patience = decreasing_loss_patience
		self.loss_counter = 0
		self.training_counter = 0
		self.best_score = None
		self.early_stop = False
		self.best_epoch = None
		self.model_config=model_config

	def save_checkpoint(self, val_loss):
		'''
		Function to continually save the best model.

		Args:
			val_loss (float): the validation loss for the model
		'''

		# saving the model if the validation loss is less than the best loss
		self.best_loss = val_loss
		print('Saving checkpoint!')

		torch.save({'model': self.model.state_dict(),
					'optimizer':self.optimizer.state_dict(),
					'best_epoch':self.best_epoch,
					'finished_training':False,
					'model_config':self.model_config},
					model_file)

	def __call__(self, train_loss, val_loss, model, optimizer, epoch):
		'''
		Function to call the early stopping condition.

		Args:
			train_loss (float): the training loss for the model
			val_loss (float): the validation loss for the model
			model (object): the model to be saved
			epoch (int): the current epoch

		Returns:
			bool: whether the model should stop training or not
		'''

		# using the absolute value of the loss for negatively orientied loss functions
		# val_loss = abs(val_loss)

		# initializing the best score if it is not already
		self.model = model
		self.optimizer = optimizer
		if self.best_score is None:
			self.best_train_loss = train_loss
			self.best_score = val_loss
			self.best_loss = val_loss
			self.save_checkpoint(val_loss)
			self.best_epoch = epoch

		# if the validation loss greater than the best score add one to the loss counter
		elif val_loss >= self.best_score:
			self.loss_counter += 1

			# if the loss counter is greater than the patience, stop the model training
			if self.loss_counter >= self.decreasing_loss_patience:
				gc.collect()
				print(f'Engaging Early Stopping due to lack of improvement in validation loss. Best model saved at epoch {self.best_epoch} with a training loss of {self.best_train_loss} and a validation loss of {self.best_score}')
				return True

		# if the validation loss is less than the best score, reset the loss counter and use the new validation loss as the best score
		else:
			self.best_train_loss = train_loss
			self.best_score = val_loss
			self.best_epoch = epoch

			# saving the best model as a checkpoint
			self.save_checkpoint(val_loss)
			self.loss_counter = 0
			self.training_counter = 0

			return False


def resume_training(model, optimizer):
	'''
	Function to resume training of a model if it was interupted without completeing.

	Args:
		model (object): the model to be trained
		optimizer (object): the optimizer to be used
		pretraining (bool): whether the model is being pre-trained

	Returns:
		object: the model to be trained
		object: the optimizer to be used
		int: the epoch to resume training from
	'''

	try:
		checkpoint = torch.load(model_file)
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		epoch = checkpoint['best_epoch']
		finished_training = checkpoint['finished_training']
	except KeyError:
		model.load_state_dict(torch.load(model_file))
		optimizer = None
		epoch = 0
		finished_training = True

	return model, optimizer, epoch, finished_training


def fit_model(model, empty_model, train, val, val_loss_patience=25, num_epochs=500, bin_edges=None, bin_weights=None, model_config=None):

	'''
	_summary_: Function to train the swmag model.

	Args:
		model (object): the model to be trained
		train (torch.utils.data.DataLoader): the training data
		val (torch.utils.data.DataLoader): the validation data
		val_loss_patience (int): the number of epochs to wait before stopping the model
									if the validation loss does not decrease
		num_epochs (int): the number of epochs to train the model
		pretraining (bool): whether the model is being pre-trained

	Returns:
		object: the trained model
	'''

	bin_edges = bin_edges.to(DEVICE)
	bin_weights = bin_weights.to(DEVICE)
	criterion = WeightedCRPS(bin_edges=bin_edges, bin_weights=bin_weights)
	optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

	# checking if the model has already been trained, loading it if it exists
	if os.path.exists(model_file):
		model, optimizer, current_epoch, finished_training = resume_training(model=model, optimizer=optimizer)
	else:
		finished_training = False
		current_epoch = 0

	if current_epoch is None:
		current_epoch = 0

	# checking to see if the model was already trained or was interupted during training
	if not finished_training:

		# initializing the lists to hold the training and validation loss which will be used to plot the losses as a function of epoch
		train_loss_list, val_loss_list = [], []

		# moving the model to the available device
		model.to(DEVICE)

		# initalizing the early stopping class
		early_stopping = Early_Stopping(decreasing_loss_patience=val_loss_patience, model_config=model_config)

		# looping through the epochs
		while current_epoch < num_epochs:

			# starting the clock for the epoch
			stime = time.time()

			# setting the model to training mode
			model.train()

			# initializing the running loss
			running_training_loss, running_val_loss = 0.0, 0.0

			# using the training set to train the model
			for X, y in tqdm.tqdm(train):

				# moving the data to the available device
				X = X.to(DEVICE, dtype=torch.float)
				y = y.to(DEVICE, dtype=torch.float)
				# adding a channel dimension to the data
				X = X.unsqueeze(1)

				# forward pass
				output = model(X)
				# print(output.shape)
				output = output.squeeze()
				# print(output.shape)
				# calculating the loss

				loss = criterion(output, y)

				# backward pass
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				# emptying the cuda cache
				X = X.to('cpu')
				y = y.to('cpu')

				# adding the loss to the running training loss
				running_training_loss += loss.to('cpu').item()


			# setting the model to eval mode so the dropout layers are not used during validation and weights are not updated
			model.eval()

			# using validation set to check for overfitting
			# looping through the batches
			for X, y in tqdm.tqdm(val):

				# moving the data to the available device
				X = X.to(DEVICE, dtype=torch.float)
				y = y.to(DEVICE, dtype=torch.float)

				# adding a channel dimension to the data
				X = X.unsqueeze(1)

				# forward pass with no gradient calculation
				with torch.no_grad():

					output = model(X)
					# output = output.view(len(output),2)
					output = output.squeeze()

					val_loss = criterion(output, y)

					# emptying the cuda cache
					X = X.to('cpu')
					y = y.to('cpu')

					# adding the loss to the running val loss
					running_val_loss += val_loss.to('cpu').item()

			# getting the average loss for the epoch
			loss = running_training_loss/len(train)
			val_loss = running_val_loss/len(val)

			# adding the loss to the list
			train_loss_list.append(loss)
			val_loss_list.append(val_loss)

			# checking for early stopping or the end of the training epochs
			if (early_stopping(train_loss=loss, val_loss=val_loss, model=model, optimizer=optimizer, epoch=current_epoch)) or (current_epoch == num_epochs-1):

				# saving the final model
				gc.collect()

				# clearing the cuda cache
				torch.cuda.empty_cache()
				gc.collect()

				# clearing the model so the best one can be loaded without overwhelming the gpu memory
				model = None
				model = empty_model

				# loading the best model version
				final = torch.load(model_file)

				# setting the finished training flag to True
				final['finished_training'] = True

				# getting the best model state dict
				model.load_state_dict(final['model'])

				# saving the final model
				torch.save(final, model_file)

				# breaking the loop
				break

			# getting the time for the epoch
			epoch_time = time.time() - stime

			# printing the loss for the epoch
			print(f'Epoch [{current_epoch}/{num_epochs}], Loss: {loss:.4f} Validation Loss: {val_loss:.4f}' + f' Epoch Time: {epoch_time:.2f} seconds')

			# emptying the cuda cache
			torch.cuda.empty_cache()

			# updating the epoch
			current_epoch += 1

		# transforming the lists to a dataframe to be saved
		loss_tracker = pd.DataFrame({'train_loss':train_loss_list, 'val_loss':val_loss_list})

		if not os.path.exists(working_dir+'loss_tracker'):
			os.makedirs(working_dir+'loss_tracker')
		loss_tracker.to_feather(working_dir + f'loss_tracker/{CONFIG["version"]}_{CONFIG["eras"]}_loss_tracker.feather')

		gc.collect()

	else:
		# loading the model if it has already been trained.
		try:
			final = torch.load(model_file)
			model.load_state_dict(final['model'])
		except KeyError:
			model.load_state_dict(torch.load(model_file))

	return model


def evaluation(model, test, test_dict):
	'''
	Function using the trained models to make predictions with the testing data.

	Args:
		model (object): pre-trained model
		test_dict (dict): dictonary with the testing model inputs and the real data for comparison
		split (int): which split is being tested

	Returns:
		dict: test dict now containing columns in the dataframe with the model predictions for this split
	'''
	# creting an array to store the predictions
	output, xtest_list, ytest_list = [], [], []
	# setting the encoder and decoder into evaluation model
	model.eval()

	# creating a loss value
	running_loss = 0.0

	# making sure the model is on the correct device
	model.to(DEVICE, dtype=torch.float)

	with torch.no_grad():
		for x, y in tqdm.tqdm(test):

			x = x.to(DEVICE, dtype=torch.float)
			y = y.to(DEVICE, dtype=torch.float)

			x = x.unsqueeze(1)

			predicted = model(x)

			predicted = predicted.squeeze()
			# print(predicted.shape)
			predicted = predicted[:,:,:,1:-1]
			# getting shape of tensor
			loss = F.mse_loss(predicted[:,0,:,:], y) # this one is for CRPS
			# loss = F.mse_loss(predicted, y) # this one is for non-crps
			running_loss += loss.item()

			# making sure the predicted value is on the cpu
			if predicted.get_device() != -1:
				predicted = predicted.to('cpu')
			if x.get_device() != -1:
				x = x.to('cpu')
			if y.get_device() != -1:
				y = y.to('cpu')

			# adding the decoded result to the predicted list after removing the channel dimension
			predicted = torch.squeeze(predicted, dim=1).numpy()

			output.append(predicted)

			x = torch.squeeze(x, dim=1).numpy()

	output = np.concatenate(output,axis=0)
	print(output.shape)
	print(f'Evaluation Loss: {running_loss/len(test)}')

	# transforming the lists to arrays
	for pred, key in zip(output, test_dict.keys()):
		test_dict[key]['predicted'] = pred

	return test_dict


def main():
	'''
	Pulls all the above functions together. Outputs a saved file with the results.

	'''
	if not os.path.exists(working_dir+f'/outputs'):
		os.makedirs(working_dir+f'/outputs')
	if not os.path.exists(working_dir+f'/models'):
		os.makedirs(working_dir+f'/models')

	# loading all data and indicies
	print('Loading data...')
	PD = PreparingData()
	train_dict, val_dict, test_dict = PD()

	# for 2d outputs (ACORN)
	train_x, train_y = [train_dict[key]['input'] for key in train_dict.keys()], [train_dict[key]['ampere'] for key in train_dict.keys()]
	val_x, val_y = [val_dict[key]['input'] for key in val_dict.keys()], [val_dict[key]['ampere'] for key in val_dict.keys()]
	test_x, test_y = [test_dict[key]['input'] for key in test_dict.keys()], [test_dict[key]['ampere'] for key in test_dict.keys()]

	print(f'Y train shape: {train_y[0].shape}')
	bin_weights, hist, bin_edges = create_bin_weights(np.hstack(train_y), num_bins=[0,np.percentile(np.abs(np.hstack(train_y)),95), np.max(np.abs(np.hstack(train_y)))], range_min=None, range_max=None)
	bin_weights = torch.tensor(bin_weights)
	bin_edges = torch.tensor(bin_edges)
	# bin_weights, bin_edges = None, None

	train_y = np.array([np.concatenate((Y[:,-2:-1],Y,Y[:,0:1]), axis=1) for Y in train_y])
	val_y = np.array([np.concatenate((Y[:,-2:-1],Y,Y[:,0:1]), axis=1) for Y in val_y])

	print(f'Y train shape: {train_y[0].shape}')

	# creating the dataloaders
	train = DataLoader(list(zip(train_x, train_y)), batch_size=CONFIG['batch_size'], shuffle=True)
	val = DataLoader(list(zip(val_x, val_y)), batch_size=CONFIG['batch_size'], shuffle=True)
	test = DataLoader(list(zip(test_x, test_y)), batch_size=CONFIG['batch_size'], shuffle=False)

	# creating the model
	print('Creating model....')

	# setting random seed
	torch.manual_seed(CONFIG['random_seed'])
	torch.cuda.manual_seed(CONFIG['random_seed'])

	# for 1d CRPS output
	# output_size = (2*train_y[0].shape[0],)

	# for 1d or 2d mse or weighted MSE output
	output_size = train_y[0].shape

	# model_config = dict(in_channels=1,
	# 		out_channels=2,
	# 		base_channels=64,
	# 		depth=2,
	# 		num_res_blocks=3,
	# 		layers_per_block=3,
	# 		channel_mult=2.0,
	# 		cbam_reduction=8,
	# 		dropout_rate=0.3,
	# 		dropout_depth=0,
	# 		enc_kernel_size=3,
	# 		dec_kernel_size=2,
	# 		debug=False,
	# 		output_size=output_size,
	# 		use_cbam=True,
	# 		use_attention_gates=True,)
	model_config = CONFIG['model_config']
	model_config['output_size']=output_size
	model = ACORN(**model_config)

	print(model)

	# printing model summary
	model.to(DEVICE)

	# fitting the model
	print('Fitting model....')
	model = fit_model(model=model, empty_model=model, train=train, val=val, val_loss_patience=25,
						num_epochs=CONFIG['epochs'], bin_edges=bin_edges, bin_weights=bin_weights,
						model_config=model_config)
	# making predictions
	print('Making predictions....')
	test_dict = evaluation(model, test, test_dict)

	# saving the results
	print('Saving results....')
	with open(working_dir+f'/outputs/{CONFIG["model"]}_{CONFIG["version"]}_{CONFIG["eras"]}_storm_training_{CONFIG["extract_storms"]}_results.pkl', 'wb') as f:
		pickle.dump(test_dict, f)

	# clearing the session to prevent memory leaks
	gc.collect()


if __name__ == '__main__':

	main()

	print('It ran. Good job!')
