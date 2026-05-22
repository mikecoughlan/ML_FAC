# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import gc
import json
import os
import pickle
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import tqdm

import shap
from data_prep import PreparingData
from model_classes import *

pd.options.mode.chained_assignment = None

os.environ["CDF_LIB"] = "~/CDF/lib"

working_dir = os.path.dirname(os.path.abspath(__file__))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
# DEVICE = torch.device('cpu')
# print(f'Device: {DEVICE}')

# Loading CONFIG json file
with open('config.json', 'r') as f:
	CONFIG = json.load(f)

os.makedirs(CONFIG["model_dir"], exist_ok=True)

model_file = f'{CONFIG["model_dir"]}{CONFIG["model"]}_{CONFIG["version"]}_{CONFIG["eras"]}.pt'


def _free_gpu():
	'''Flush GPU memory after freeing a tensor.'''
	gc.collect()
	if DEVICE.type == 'cuda':
		torch.cuda.empty_cache()


def _make_predict_fn(model, input_shape):
	'''
	Wraps a PyTorch model into a numpy-in / numpy-out callable for KernelExplainer.

	KernelExplainer communicates entirely in flat 2D numpy arrays (n_samples, n_features).
	This wrapper:
	  1. Reshapes the flat array back to (n_samples, *input_shape)
	  2. Moves the batch to GPU only for the forward pass
	  3. Pulls the result back to CPU as numpy immediately after

	At any moment only one perturbation batch occupies GPU memory.

	Args:
		model (torch.nn.Module): trained model already in eval mode.
		input_shape (tuple): per-sample shape before flattening, e.g. (C, H, W).

	Returns:
		callable: prediction function suitable for shap.KernelExplainer.
	'''
	def predict_fn(x_flat):
		x = torch.tensor(x_flat, dtype=torch.float).reshape(-1, *input_shape).to(DEVICE)
		with torch.no_grad():
			out = model(x)
		result = out.cpu().numpy().reshape(x.shape[0], -1)
		del x, out
		_free_gpu()
		return result

	return predict_fn


def get_shap_values(model, model_name, training_data, testing_data,
					background_examples=1000, delimiter=100, explainer_type='deep'):
	'''
	Calculates SHAP values for the given model and test data.

	GPU memory strategy: everything is built and stored on CPU. Data is moved
	to GPU only at the moment it is consumed, then freed immediately. At peak,
	only the model weights + one batch (or the background for DeepExplainer
	initialisation) occupy GPU memory at any one time.

	Supports two explainer backends:

	  'deep'   -- shap.DeepExplainer. Fast, gradient-based, PyTorch-native.
	             Background is moved to GPU for init then freed before the
	             test loop begins. Each test batch is moved to GPU, explained,
	             then freed before the next batch.

	  'kernel' -- shap.KernelExplainer. Model-agnostic, perturbation-based.
	             Fully CPU/numpy-based. predict_fn moves each internal
	             perturbation batch to GPU for inference only.
	             Use smaller background_examples (e.g. 100) and delimiter (e.g. 5).

	Args:
		model (torch.nn.Module): trained neural network model.
		model_name (str): version string, used for labelling.
		training_data (list[Tensor] | np.ndarray | Tensor): source for background samples.
		testing_data (list[Tensor] | np.ndarray | Tensor): data to explain.
		background_examples (int): number of background samples. Defaults to 1000.
		delimiter (int): batch size for SHAP forward passes. Defaults to 100.
		explainer_type (str): 'deep' or 'kernel'. Defaults to 'deep'.

	Returns:
		list: SHAP value batches across the test set.
		expected_value: explainer baseline (scalar for DeepExplainer,
		                list per output for KernelExplainer).
	'''

	if explainer_type not in ('deep', 'kernel'):
		raise ValueError(f"explainer_type must be 'deep' or 'kernel', got '{explainer_type}'.")

	# ------------------------------------------------------------------
	# 1. Build background and testing tensors on CPU only.
	#    Neither is moved to GPU here — that happens lazily below.
	# ------------------------------------------------------------------
	if isinstance(training_data, list):
		print(f'Training data is a list of {len(training_data)} tensors, shape: {training_data[0].shape}')
		input_shape = training_data[0].shape  # per-sample shape, e.g. (C, H, W)

		random_indices = np.random.choice(len(training_data), background_examples, replace=False)
		# Stack on CPU in one vectorised call
		background_cpu = torch.stack(
			[training_data[i] for i in random_indices], dim=0
		).to('cpu', dtype=torch.float)

		# Stack test data on CPU — batches will be moved to GPU one at a time
		testing_cpu = torch.stack(testing_data, dim=0).to('cpu', dtype=torch.float)

	elif isinstance(training_data, (np.ndarray, torch.Tensor)):
		print('Training data is a numpy array / tensor....')
		input_shape = training_data[0].shape

		random_indices = np.random.choice(len(training_data), background_examples, replace=False)
		if isinstance(training_data, np.ndarray):
			background_cpu = torch.tensor(training_data[random_indices], dtype=torch.float)
			testing_cpu = torch.tensor(testing_data, dtype=torch.float)
		else:
			background_cpu = training_data[random_indices].to('cpu', dtype=torch.float)
			testing_cpu = testing_data.to('cpu', dtype=torch.float)

	else:
		raise ValueError('training_data must be a list of Tensors, a numpy array, or a Tensor.')

	print(f'Background shape: {background_cpu.shape}')
	print(f'Testing data shape: {testing_cpu.shape}')

	del training_data
	gc.collect()

	n_samples = testing_cpu.shape[0]

	# ------------------------------------------------------------------
	# 2. Build explainer.
	#    model.eval() is required regardless of explainer type — disables
	#    dropout and fixes batchnorm statistics for consistent explanations.
	# ------------------------------------------------------------------
	model.eval()

	if explainer_type == 'deep':
		# Move background to GPU only for DeepExplainer initialisation.
		# Once the explainer has ingested it, free it from GPU immediately
		# so the test loop starts with maximum free VRAM.
		background_gpu = background_cpu.to(DEVICE)
		del background_cpu
		gc.collect()

		with torch.no_grad():
			# explainer = shap.DeepExplainer(model=model, data=background_gpu)
			explainer = shap.GradientExplainer(model=model, data=background_gpu)

		del background_gpu
		_free_gpu()

		# ------------------------------------------------------------------
		# 3a. Calculate SHAP values — DeepExplainer
		#     Each batch is moved to GPU, explained, then freed before the
		#     next batch is loaded. Peak GPU = model + one batch.
		# ------------------------------------------------------------------
		print('Calculating SHAP values (DeepExplainer)....')
		shap_values = []
		for batch_start in tqdm.tqdm(range(0, n_samples, delimiter), desc='shap value calculations'):
			batch_end = min(batch_start + delimiter, n_samples)
			batch_gpu = testing_cpu[batch_start:batch_end].to(DEVICE)
			shap_values.append(
				explainer.shap_values(batch_gpu)
			)
			print(shap_values[0].shape)
			raise
			del batch_gpu
			_free_gpu()

	else:
		# ------------------------------------------------------------------
		# 3b. Calculate SHAP values — KernelExplainer
		#
		# KernelExplainer is fully CPU/numpy-based. Background and test data
		# are flattened to (n_samples, n_features) numpy arrays here.
		# predict_fn (see _make_predict_fn) moves each internal perturbation
		# batch to GPU only for the forward pass, then pulls back to CPU.
		# Peak GPU = model + one KernelExplainer perturbation batch.
		#
		# Note: KernelExplainer.shap_values() does not accept check_additivity.
		# It is significantly slower than DeepExplainer — keep delimiter small.
		# ------------------------------------------------------------------
		predict_fn = _make_predict_fn(model, input_shape)

		background_np = background_cpu.numpy().reshape(background_cpu.shape[0], -1)
		del background_cpu
		gc.collect()

		explainer = shap.KernelExplainer(predict_fn, background_np)
		del background_np
		gc.collect()

		testing_np = testing_cpu.numpy().reshape(n_samples, -1)
		del testing_cpu
		gc.collect()

		print('Calculating SHAP values (KernelExplainer)....')
		shap_values = []
		for batch_start in tqdm.tqdm(range(0, n_samples, delimiter), desc='shap value calculations'):
			batch_end = min(batch_start + delimiter, n_samples)
			shap_values.append(
				explainer.shap_values(testing_np[batch_start:batch_end])
			)

	return shap_values, '____'#explainer.expected_value


def converting_shap_to_percentages(shap_values, features):
	'''
	Converts raw SHAP values to percentage contributions per feature.

	Args:
		shap_values (list[np.ndarray] | np.ndarray): SHAP values to convert.
		features (list[str]): feature names for DataFrame columns.

	Returns:
		list[pd.DataFrame] | pd.DataFrame: percentage contributions.
	'''

	def _to_percentage(arr):
		# Sum across spatial/time axis, reshape to (samples, features), then normalise
		summed = np.sum(arr, axis=1).reshape(arr.shape[0], -1)
		df = pd.DataFrame(summed, columns=features)
		return df.div(df.abs().sum(axis=1), axis=0) * 100

	if len(shap_values) > 1:
		return [_to_percentage(sv) for sv in shap_values]
	else:
		return _to_percentage(shap_values[0])


def compare_state_dicts(checkpoint_path, model):
    """Print all shape mismatches between a checkpoint and a live model."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Unwrap if nested
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]

    model_sd = model.state_dict()

    ckpt_keys  = set(ckpt.keys())
    model_keys = set(model_sd.keys())

    missing   = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    mismatched = {
        k for k in ckpt_keys & model_keys
        if ckpt[k].shape != model_sd[k].shape
    }

    print(f"  Missing in checkpoint  : {len(missing)}")
    print(f"  Unexpected in checkpoint: {len(unexpected)}")
    print(f"  Shape mismatches       : {len(mismatched)}")

    if mismatched:
        print(f"\n  {'Key':<55} {'Checkpoint':>20} {'Model':>20}")
        print(f"  {'─'*55} {'─'*20} {'─'*20}")
        for k in sorted(mismatched):
            print(f"  {k:<55} {str(tuple(ckpt[k].shape)):>20} {str(tuple(model_sd[k].shape)):>20}")

    if missing:
        print(f"\n  Missing keys:\n  " + "\n  ".join(sorted(missing)))
    if unexpected:
        print(f"\n  Unexpected keys:\n  " + "\n  ".join(sorted(unexpected)))

def main():
	'''
	Pulls all the above functions together. Outputs a saved file with the results.
	'''
	os.makedirs(working_dir + '/outputs', exist_ok=True)
	os.makedirs(working_dir + '/models', exist_ok=True)
	os.makedirs(working_dir + '/shap', exist_ok=True)

	print('Loading data...')
	PD = PreparingData()
	train_dict, val_dict, test_dict = PD()

	# Gathering dates
	dates = [date for date in test_dict.keys()]

	may_start = pd.to_datetime('2023-05-04 00:00:00')
	may_end = pd.to_datetime('2023-05-09 00:00:00')

	# Build tensor lists on CPU — unsqueeze(0) adds the channel dim required by the conv layers
	train_x = [torch.tensor(train_dict[key]['input']).unsqueeze(0) for key in train_dict.keys()]
	# test_x  = [torch.tensor(test_dict[key]['input']).unsqueeze(0) for key in test_dict.keys() if pd.to_datetime(key) >= may_start and pd.to_datetime(key) <= may_end]
	test_x  = [torch.tensor(test_dict[key]['input']).unsqueeze(0) for key in test_dict.keys()]

	# Free dicts not needed for SHAP
	del train_dict, val_dict, test_dict
	gc.collect()

	print('Creating model....')
	torch.manual_seed(CONFIG['random_seed'])
	torch.cuda.manual_seed(CONFIG['random_seed'])

	output_size = (50,24)
	input_shape = train_x[0].squeeze(0).shape

	# model = BK_model(input_size=input_shape, output_size=output_size,
	# 				 num_channels=128, num_residual_blocks=3, crps=True)
	# model = ResidualUNet(
	# 	in_channels=1,
	# 	out_channels=2,
	# 	base_channels=64,
	# 	depth=4,
	# 	num_res_blocks=2,
	# 	layers_per_block=3,
	# 	channel_mult=2.0,
	# 	cbam_reduction=8,
	# 	output_size=output_size,
	# )

	# cfg = dict(use_cbam=True, use_attention_gates=True)

	# model_config = {"in_channels":1,
	# 				"out_channels":2,
	# 				"base_channels":64,
	# 				"depth":2,
	# 				"num_res_blocks":3,
	# 				"layers_per_block":3,
	# 				"channel_mult":2.0,
	# 				"cbam_reduction":8,
	# 				"dropout_rate":0.3,   # applied at bottleneck + dropout_depth levels
	# 				"dropout_depth":0,    # bottleneck + deepest enc/dec level
	# 				"output_size":output_size}

	model = ACORN(in_channels=1,
            out_channels=2,
            base_channels=64,
            depth=2,
            num_res_blocks=3,
            layers_per_block=3,
            channel_mult=2.0,
			cbam_reduction=8,
            dropout_rate=0.3,
			dropout_depth=0,
            debug=False,
			output_size=output_size,
            use_cbam=True,
			use_attention_gates=True,
    )
	compare_state_dicts(model_file, model)
	print(model)
	# map_location loads the checkpoint directly to the target device,
	# skipping a redundant CPU intermediate
	model.to(DEVICE)
	checkpoint = torch.load(model_file, map_location=DEVICE)
	model.load_state_dict(checkpoint['model'])

	# Set explainer_type to 'deep' or 'kernel'
	# 'deep'   -- faster, gradient-based, recommended for differentiable models
	# 'kernel' -- slower, model-agnostic; use smaller background_examples + delimiter
	EXPLAINER_TYPE = 'deep'

	print(f'Getting SHAP values using {EXPLAINER_TYPE} explainer....')
	shap_values, expectation_values = get_shap_values(
		model=model,
		model_name=CONFIG["version"],
		training_data=train_x,
		testing_data=test_x,
		background_examples=1000,
		delimiter=1,
		explainer_type=EXPLAINER_TYPE,
	)
	# with open('shap/ACORN_0_0_deep.pkl', 'rb') as f:
	# 	shap = pickle.load(f)

	print('Saving results...')
	with open(working_dir + f'/shap/{CONFIG["version"]}_{EXPLAINER_TYPE}.pkl', 'wb') as f:
		pickle.dump({'shap_values': shap['shap_values'],
					'expectation_values': shap['expectation_values'],
					'dates':dates}, f)

	gc.collect()


if __name__ == '__main__':

	main()

	print('It ran. Good job!')
