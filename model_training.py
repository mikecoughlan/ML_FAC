"""
model_training.py
=================

Trains an ACORN model end to end.

Pulls prepared data from data_prep.PreparingData, builds the network from
model_classes.ACORN, trains it against a loss from
custom_loss_functions.build_loss, and saves the trained weights plus a
results dictionary holding test-set predictions for later analysis.

Selecting a model
-----------------
	python model_training.py            # science model (config.json active_model)
	ACORN_MODEL=op python model_training.py   # operational model

CONFIG is read at module scope, so the choice is made from the
environment rather than parsed from argv. The same model name is handed
to PreparingData, so data preparation and training cannot drift apart.

Outputs
-------
	models/acorn_<model>.pt          trained weights
	outputs/results_<model>.pkl      test-set predictions
	loss_tracker/loss_<model>.feather per-epoch loss history
"""

import gc
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

import utils
from custom_loss_functions import build_loss, create_bin_weights
from data_prep import PreparingData
from model_classes import ACORN

# Which model to train: 'sci' or 'op'. See config.json.
MODEL_NAME = os.environ.get('ACORN_MODEL', None)

# Merged configuration for the selected model.
CONFIG = utils.load_config(MODEL_NAME)
MODEL_NAME = CONFIG['model_name']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

pd.options.mode.chained_assignment = None

if not os.path.exists(CONFIG["model_dir"]):
	os.makedirs(CONFIG["model_dir"])


model_file = utils.model_file(CONFIG)


class Early_Stopping():
	'''
	Class to create an early stopping condition for the model.

	'''

	def __init__(self, decreasing_loss_patience=25, model_config=None):
		'''
		Initializing the class.

		Args:
			decreasing_loss_patience (int): number of epochs to wait without an
				improvement in validation loss before signaling a stop.
			model_config (dict, optional): model configuration used when saving
				checkpoints. Falls back to CONFIG['model_config'] when omitted.
		'''

		self.model_config = model_config if model_config is not None else CONFIG['model_config']
		self.decreasing_loss_patience = decreasing_loss_patience

		# Epochs elapsed since the last improvement; compared against the patience.
		self.loss_counter = 0

		# Best validation loss so far. None until the first epoch establishes a
		# baseline, which distinguishes "no score yet" from a legitimately low one.
		self.best_score = None

		# Set once patience is exhausted; the training loop polls this to break.
		self.early_stop = False
		self.best_epoch = None

	def save_checkpoint(self, val_loss):
		'''
		Function to continually save the best model.

		Args:
			val_loss (float): the validation loss for the model at the current epoch
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
			optimizer (object): the optimizer state to be saved alongside the model
			epoch (int): the current epoch

		Returns:
			bool: whether the model should stop training or not
		'''

		self.model = model
		self.optimizer = optimizer

		# Using the absolute value as comparison in case the loss is negativly oriented
		val_score = abs(val_loss)

		# initializing the best score if it is not already
		if self.best_score is None:
			self.best_train_loss = train_loss
			self.best_score = val_score
			self.best_loss = val_loss
			self.save_checkpoint(val_loss)
			self.best_epoch = epoch

		# if the validation magnitude is no better than the best, add one to the loss counter
		elif val_score >= self.best_score:
			self.loss_counter += 1

			# if the loss counter is greater than the patience, stop the model training
			if self.loss_counter >= self.decreasing_loss_patience:
				gc.collect()
				print(f'Engaging Early Stopping due to lack of improvement in validation loss. '
					f'Best model saved at epoch {self.best_epoch} with a training loss of '
					f'{self.best_train_loss} and a validation loss of {self.best_loss}')
				return True

		# if the validation magnitude improved, reset the counter and record the new best
		else:
			self.best_train_loss = train_loss
			self.best_score = val_score
			self.best_loss = val_loss
			self.best_epoch = epoch

			# saving the best model as a checkpoint
			self.save_checkpoint(val_loss)
			self.loss_counter = 0

		return False


def resume_training(model, optimizer):
	'''
	Function to resume training of a model if it was interupted without completeing.

	Args:
		model (object): the model to be trained
		optimizer (object): the optimizer to be used

	Returns:
		object: the model with the checkpointed weights loaded
		object: the optimizer with its checkpointed state, or None if the file
			held bare weights rather than a full training checkpoint
		int: the epoch to resume training from, or 0 when there is no state to resume
		bool: whether training had already finished, in which case there is
			nothing to resume
	'''

	try:
		checkpoint = torch.load(model_file)
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		epoch = checkpoint['best_epoch']
		finished_training = checkpoint['finished_training']

	# A bare state_dict rather than a training checkpoint: the weights are
	# usable but there is no optimizer or epoch state, so treat it as done.
	except KeyError:
		model.load_state_dict(torch.load(model_file))
		optimizer = None
		epoch = 0
		finished_training = True

	return model, optimizer, epoch, finished_training


def fit_model(model, train, val, val_loss_patience=25, num_epochs=500, bin_edges=None, bin_weights=None, model_config=None):

	'''
	_summary_: Function to train the model.

	Args:
		model (object): the model to be trained
		train (torch.utils.data.DataLoader): the training data
		val (torch.utils.data.DataLoader): the validation data
		val_loss_patience (int): the number of epochs to wait before stopping the model
									if the validation loss does not decrease
		num_epochs (int): the number of epochs to train the model
		bin_edges (torch.Tensor): bin boundaries passed to the weighted loss
		bin_weights (torch.Tensor): per-bin weights passed to the weighted loss
		model_config (dict, optional): model configuration handed to Early_Stopping
										for checkpoint naming

	Returns:
		object: the trained model, with the best checkpointed weights loaded
		object: the loss function, carrying the is_probabilistic flag
	'''

	bin_edges = bin_edges.to(DEVICE)
	bin_weights = bin_weights.to(DEVICE)
	# Objective selected by the "loss" key in the config; see
	# LOSS_REGISTRY in custom_loss_functions.py for valid values.
	criterion = build_loss(
		CONFIG.get("loss", "weighted_crps"),
		bin_edges=bin_edges,
		bin_weights=bin_weights,
	)
	print(f'Training with loss: {type(criterion).__name__}')
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

				# calculating the loss
				batch_loss = criterion(output, y)

				# backward pass
				optimizer.zero_grad()
				batch_loss.backward()
				optimizer.step()

				# emptying the cuda cache
				X = X.to('cpu')
				y = y.to('cpu')

				# adding the loss to the running training loss
				running_training_loss += batch_loss.to('cpu').item()


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

					batch_val_loss = criterion(output, y)

					# emptying the cuda cache
					X = X.to('cpu')
					y = y.to('cpu')

					# adding the loss to the running val loss
					running_val_loss += batch_val_loss.to('cpu').item()

			# getting the average loss for the epoch. Named separately from the
			# per-batch tensors above so the two are not confused downstream.
			epoch_train_loss = running_training_loss/len(train)
			epoch_val_loss = running_val_loss/len(val)

			# adding the loss to the list
			train_loss_list.append(epoch_train_loss)
			val_loss_list.append(epoch_val_loss)

			# checking for early stopping or the end of the training epochs
			if (early_stopping(train_loss=epoch_train_loss, val_loss=epoch_val_loss, model=model, optimizer=optimizer, epoch=current_epoch)) or (current_epoch == num_epochs-1):

				# The checkpoint is written by Early_Stopping, which saves on its
				# first call, so it should exist by the time either exit condition
				# fires. Guarded anyway so a missing file surfaces as a clear error
				# rather than a load failure on the line below.
				if not os.path.exists(model_file):
					raise FileNotFoundError(
						f'Training ended at epoch {current_epoch} but no checkpoint '
						f'was written to {model_file}.'
					)

				# Adam carries two state tensors per parameter, so dropping the
				# optimizer frees more than swapping the model would.
				del optimizer
				torch.cuda.empty_cache()
				gc.collect()

				# Loading to cpu so the checkpoint tensors are not briefly resident
				# on the gpu alongside the model's own parameters. load_state_dict
				# copies into the existing tensors, so no new allocation happens.
				final = torch.load(model_file, map_location='cpu')

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
			print(f'Epoch [{current_epoch}/{num_epochs}], Loss: {epoch_train_loss:.4f} Validation Loss: {epoch_val_loss:.4f}' + f' Epoch Time: {epoch_time:.2f} seconds')

			# emptying the cuda cache
			torch.cuda.empty_cache()

			# updating the epoch
			current_epoch += 1

		# transforming the lists to a dataframe to be saved
		loss_tracker = pd.DataFrame({'train_loss':train_loss_list, 'val_loss':val_loss_list})

		loss_dir = os.path.dirname(utils.loss_file(CONFIG))
		os.makedirs(loss_dir, exist_ok=True)
		loss_tracker.to_feather(os.path.join(
			loss_dir, os.path.basename(utils.loss_file(CONFIG))
		))

		gc.collect()

	else:
		# loading the model if it has already been trained.
		try:
			final = torch.load(model_file)
			model.load_state_dict(final['model'])
		except KeyError:
			model.load_state_dict(torch.load(model_file))

	return model, criterion


def evaluation(model, test, test_dict, is_probabilistic=True):
	'''
	Function using the trained models to make predictions with the testing data.

	Args:
		model (object): pre-trained model
		test (torch.utils.data.DataLoader): batched testing inputs and targets
		test_dict (dict): dictonary with the testing model inputs and the real
							data for comparison. Keys are assumed to be in the
							same order as the samples yielded by `test`.
		is_probabilistic (bool): whether the model emits a distribution rather
							than a point estimate. Determines whether the output
							channel axis is kept. Available as
							`criterion.is_probabilistic` on any loss in
							LOSS_REGISTRY.

	Returns:
		dict: test dict now containing columns in the dataframe with the model
				predictions for this split
	'''

	# array to store the predictions
	output = []

	# setting the model into evaluation mode
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

			# Trim the wrap-padding columns added to the targets during
			# training, returning the prediction to the native 50 x 24
			# grid so it lines up with the unpadded test targets.
			predicted = predicted[:, :, :, 1:-1]

			# Guards against a config that says CRPS while the loaded checkpoint
			# was built with a single output channel. Without this the slicing
			# below silently produces the wrong shape rather than failing.
			if is_probabilistic and predicted.shape[1] < 2:
				raise ValueError(
					f'is_probabilistic=True but the model emits '
					f'{predicted.shape[1]} output channel(s).'
				)

			# Channel 0 is the mean under both conventions: the sole output of a
			# deterministic model, or the first of the two a probabilistic one
			# emits. Reported as plain MSE for reference, not the training
			# objective when that objective is CRPS.
			loss = F.mse_loss(predicted[:, 0, :, :], y)
			running_loss += loss.item()

			# bringing everything back to the cpu before converting to numpy
			predicted = predicted.cpu()
			x = x.cpu()
			y = y.cpu()

			# Probabilistic models keep the channel axis so the std survives
			# alongside the mean; deterministic ones drop it, leaving a bare
			# 50 x 24 field.
			if not is_probabilistic:
				predicted = predicted[:, 0, :, :]

			output.append(predicted.numpy())

	output = np.concatenate(output, axis=0)

	print(output.shape)
	print(f'Evaluation Loss: {running_loss/len(test)}')

	if len(output) != len(test_dict):
		raise ValueError(
			f'{len(output)} predictions do not match {len(test_dict)} entries in '
			f'test_dict; zip would silently drop the excess.'
		)

	# attaching each sample's prediction to its corresponding dict entry
	for pred, key in zip(output, test_dict.keys()):
		test_dict[key]['predicted'] = pred

	return test_dict


def main():
	'''
	Pulls all the above functions together. Outputs a saved file with the results.

	'''
	# Output directories, created beside the script if absent.
	# All output paths resolve against the current working directory, the
	# same base as data_dir, so a run's inputs and outputs stay together.
	os.makedirs(os.path.dirname(utils.results_file(CONFIG)), exist_ok=True)
	os.makedirs(os.path.dirname(utils.model_file(CONFIG)), exist_ok=True)

	# loading all data and indicies
	print('Loading data...')
	PD = PreparingData(MODEL_NAME)
	train_dict, val_dict, test_dict = PD()

	# for 2d outputs (ACORN)
	train_x, train_y = [train_dict[key]['input'] for key in train_dict.keys()], [train_dict[key]['ampere'] for key in train_dict.keys()]
	val_x, val_y = [val_dict[key]['input'] for key in val_dict.keys()], [val_dict[key]['ampere'] for key in val_dict.keys()]
	test_x, test_y = [test_dict[key]['input'] for key in test_dict.keys()], [test_dict[key]['ampere'] for key in test_dict.keys()]

	print(f'Y train shape: {train_y[0].shape}')
	bin_weights, hist, bin_edges = create_bin_weights(np.hstack(train_y), num_bins=[0,np.percentile(np.abs(np.hstack(train_y)),95), np.max(np.abs(np.hstack(train_y)))], range_min=None, range_max=None)
	bin_weights = torch.tensor(bin_weights)
	bin_edges = torch.tensor(bin_edges)

	# Pad the MLT axis with one wrapped column on each side, so the
	# network sees midnight as continuous rather than as a hard edge.
	# MLT is circular: column 23 precedes column 0 and column 0 follows
	# column 23, giving [23, 0..23, 0] and a 50 x 26 target.
	#
	# test_y is deliberately left unpadded: evaluation trims the model's two
	# wrap columns instead, so predictions land on the native 50 x 24 grid.
	# Padding test here would double-count the trim and misalign the targets.
	train_y = np.array([np.concatenate((Y[:, -1:], Y, Y[:, 0:1]), axis=1) for Y in train_y])
	val_y = np.array([np.concatenate((Y[:, -1:], Y, Y[:, 0:1]), axis=1) for Y in val_y])

	print(f'Y train shape: {train_y[0].shape}')

	# creating the dataloaders. test is unshuffled so its order matches the
	# insertion order of test_dict.
	train = DataLoader(list(zip(train_x, train_y)), batch_size=CONFIG['batch_size'], shuffle=True)
	val = DataLoader(list(zip(val_x, val_y)), batch_size=CONFIG['batch_size'], shuffle=True)
	test = DataLoader(list(zip(test_x, test_y)), batch_size=CONFIG['batch_size'], shuffle=False)

	# creating the model
	print('Creating model....')

	# setting random seed
	torch.manual_seed(CONFIG['random_seed'])
	torch.cuda.manual_seed(CONFIG['random_seed'])

	# Output grid the network is built against: the padded 50 x 26 target.
	output_size = train_y[0].shape

	model_config = CONFIG['model_config']
	model_config['output_size']=output_size
	model = ACORN(**model_config)

	print(model)

	# printing model summary
	model.to(DEVICE)

	# fitting the model
	print('Fitting model....')
	model, criterion = fit_model(model=model, train=train, val=val,
						val_loss_patience=CONFIG['early_stop_patience'],
						num_epochs=CONFIG['epochs'], bin_edges=bin_edges, bin_weights=bin_weights,
						model_config=model_config)

	# making predictions
	print('Making predictions....')
	test_dict = evaluation(model, test, test_dict, is_probabilistic=criterion.is_probabilistic)

	# saving the results
	print('Saving results....')
	results_path = utils.results_file(CONFIG)
	with open(results_path, 'wb') as f:
		pickle.dump(test_dict, f)

	# clearing the session to prevent memory leaks
	gc.collect()


if __name__ == '__main__':

	# Which config this run used, echoed so it appears in the log
	# alongside the results.
	print(f'Model: {MODEL_NAME}')

	main()

	print('It ran. Good job!')
