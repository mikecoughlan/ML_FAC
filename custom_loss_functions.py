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
# from spacepy import pycdf
from torch.utils.data import DataLoader, Dataset, TensorDataset

import utils

# from torchsummary import summary
# from torchvision.models.feature_extraction import (create_feature_extractor,
                                                #    get_graph_node_names)

pd.options.mode.chained_assignment = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')


class CRPS(nn.Module):
    '''
    Defining the CRPS loss function for model training.
    '''

    def __init__(self):
        super(CRPS, self).__init__()

    def forward(self, y_pred, y_true):

        # splitting the y_pred tensor into mean and std

        mean, std = torch.unbind(y_pred, dim=1)

        # making the arrays the right dimensions
        mean = mean.unsqueeze(-1)
        std = std.unsqueeze(-1)
        y_true = y_true.unsqueeze(-1)

        # calculating the error
        crps = torch.mean(self.calculate_crps(self.epsilon_error(y_true, mean), std))

        return crps

    def epsilon_error(self, y, u):

        epsilon = torch.abs(y - u)

        return epsilon

    def calculate_crps(self, epsilon, sig):

        crps = torch.mul(sig, (torch.add(torch.mul(torch.div(epsilon, sig), torch.erf(torch.div(epsilon, torch.mul(np.sqrt(2), sig)))), \
								torch.sub(torch.mul(torch.sqrt(torch.div(2, np.pi)), torch.exp(torch.div(torch.mul(-1, torch.pow(epsilon, 2)), \
								(torch.mul(2, torch.pow(sig, 2)))))), torch.div(1, torch.sqrt(torch.tensor(np.pi)))))))


        return crps


def create_bin_weights(y_train, num_bins=50, range_min=None, range_max=None):
    """
    Create weights based on the inverse frequency of samples in bins.
    """
    # Histogram of values
    if range_min == None:
        range_min = min(np.abs(y_train.flatten()))
    if range_max == None:
        range_max = max(np.abs(y_train.flatten()))
    hist, bin_edges = np.histogram(np.abs(y_train), bins=num_bins, range=(range_min, range_max), density=False)
    hist = hist/np.sum(hist)

    if not isinstance(num_bins, int):
        num_bins=len(num_bins)-1

    bin_width = (bin_edges[-1] - bin_edges[0]) / num_bins
    print(f'Hist results: {hist}')
    # def get_weight(value):
    #     bin_idx = int((value - bin_edges[0]) / bin_width)
    #     bin_idx = min(bin_idx, num_bins - 1)
    #     bin_idx = max(bin_idx, 0)
    #     print(f'Value: {value}, bin_idx: {bin_idx}')
    #     print(f'1/hist[bin_idx]: {1/hist[bin_idx]}')
    #     raise
    #     return 1/hist[bin_idx]

    # weights = np.array([get_weight(val) for val in hist])
    inverse_weights = pd.Series(np.where(hist==0, np.nan, hist)).interpolate(method='linear').to_numpy()
    weights = 1/inverse_weights

    # print(f'y_train shape: {y_train.shape}')
    # print(f'weights shape: {weights.shape}')
    # print(f'weights: {weights}')
    # print(f'bin_edges shape: {bin_edges.shape}')
    # print(f'bin edges: {bin_edges}')

    return weights, hist, bin_edges


class WeightedCRPS(nn.Module):
    """
    Custom loss function that applies weights based on the target value's frequency.

    This loss gives higher importance to rare samples
    while maintaining reasonable performance for common samples.
    """
    def __init__(self, bin_edges, bin_weights):
        super(WeightedCRPS, self).__init__()

        # Store bin info as buffers so they move with .to(device)
        self.register_buffer(
            "bin_edges",
            torch.tensor(bin_edges[:-1], dtype=torch.float32)
        )
        self.register_buffer(
            "bin_weights",
            torch.tensor(bin_weights, dtype=torch.float32)
        )

        self.num_bins = len(bin_weights)
        self.range_min = bin_edges[0]
        self.range_max = bin_edges[-1]
        self.bin_width = (self.range_max - self.range_min) / self.num_bins

    def epsilon_error(self, y, u):

        epsilon = torch.abs(y - u)

        return epsilon

    def calculate_crps(self, epsilon, sig):

        sig = torch.add(sig,1e-6)
        crps = torch.mul(sig, (torch.add(torch.mul(torch.div(epsilon, sig), torch.erf(torch.div(epsilon, torch.mul(np.sqrt(2), sig)))), \
								torch.sub(torch.mul(torch.sqrt(torch.div(2, np.pi)), torch.exp(torch.div(torch.mul(-1, torch.pow(epsilon, 2)), \
								(torch.mul(2, torch.pow(sig, 2)))))), torch.div(1, torch.sqrt(torch.tensor(np.pi)))))))


        return crps

    def forward(self, y_pred, y_true):
        """
        Args:
            y_true (Tensor): target values
            y_pred (Tensor): predicted values
        """
        # splitting the y_pred tensor into mean and std
        if len(y_pred.shape)==3:
            mean, std = torch.unbind(y_pred, dim=2) # for linear model output
        elif len(y_pred.shape)==4:
            mean, std = torch.unbind(y_pred, dim=1) # for 2d conv model output
        else:
            print(f'y_pred shape: {y_pred.shape}')
            raise ValueError('Output is of an incorrect dimension. Check.')
        # making the arrays the right dimensions
        mean = mean.unsqueeze(-1)
        std = std.unsqueeze(-1)
        y_true = y_true.unsqueeze(-1)

        # flattening the tensors
        y_true = y_true.flatten()
        mean = mean.flatten()
        std = std.flatten()

        # assigning weights based on y_true
        bindices = torch.searchsorted(self.bin_edges, torch.abs(y_true), right=False) - 1
        bindices = torch.clamp(bindices, 0, len(self.bin_weights) - 1)

        weights = self.bin_weights[bindices]
        weights = weights.to(DEVICE)

        # Weighted CRPS: calculating the error
        crps = self.calculate_crps(self.epsilon_error(y_true, mean), std)
        weighted_errors = torch.mul(weights, crps)

        return torch.mean(weighted_errors)


class WeightedMeanSquaredError(nn.Module):
    """
    Custom loss function that applies weights based on the target value's frequency.

    This loss gives higher importance to rare samples
    while maintaining reasonable performance for common samples.
    """
    def __init__(self, bin_edges, bin_weights):
        super(WeightedMeanSquaredError, self).__init__()

        # Store bin info as buffers so they move with .to(device)
        self.register_buffer(
            "bin_edges",
            torch.tensor(bin_edges[:-1], dtype=torch.float32)
        )
        self.register_buffer(
            "bin_weights",
            torch.tensor(bin_weights, dtype=torch.float32)
        )

        self.num_bins = len(bin_weights)
        self.range_min = bin_edges[0]
        self.range_max = bin_edges[-1]
        self.bin_width = (self.range_max - self.range_min) / self.num_bins

    def forward(self, y_pred, y_true):
        """
        Args:
            y_true (Tensor): target values
            y_pred (Tensor): predicted values
        """
        y_true = y_true.unsqueeze(-1)
        y_pred = y_pred.unsqueeze(-1)

		# flattening the tensors
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)

        # print(f'y_true size: {y_true.shape}')
        # print(f'y_pred size: {y_pred.shape}')

        # Compute bin indices
        # bin_indices = torch.floor(
        #     (y_true - self.range_min) / self.bin_width
        #     ).long()
        # print(f'bin indices shape: {bin_indices.shape}')
        # print(bin_indices[:5])
        # # Clip to valid range
        # bin_indices = torch.clamp(
        #     bin_indices, min=0, max=self.num_bins - 1
        #     )
        bindices = torch.searchsorted(self.bin_edges, torch.abs(y_true), right=False) - 1
        bindices = torch.clamp(bindices, 0, len(self.bin_weights) - 1)

        weights = self.bin_weights[bindices]
        # print(f'bin indicies size: {bin_indices.shape}')
        # print(f'bin weights pre transform shape: {self.bin_weights.shape}')
        # bin_indices = bin_indices.to(DEVICE)
        weights = weights.to(DEVICE)
        # print(f'weights: {weights}')
        # print(f'y_true: {y_true}')
        # print(f'y_pred: {y_pred}')
        # print(f'bindices: {bindices}')

        # Weighted MSE
        # calculating the error
        mean_squared = torch.pow(torch.sub(y_pred,y_true),2)
        weighted_errors = torch.mul(weights,mean_squared)
        # print(f'mean squared: {mean_squared}')
        # print(f'weighted_errors: {weighted_errors}')
        # raise

        return torch.mean(weighted_errors)

