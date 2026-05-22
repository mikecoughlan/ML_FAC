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
from torch.utils.data import DataLoader, Dataset, TensorDataset
# from torchsummary import summary
# from torchvision.models.feature_extraction import (create_feature_extractor,
#                                                    get_graph_node_names)
import utils


class ResNet(nn.Module):
	def __init__(self, input_size, output_size, num_channels, num_residual_blocks):
		super(ResNet, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.num_channels = num_channels
		self.num_residual_blocks = num_residual_blocks
		self.num_nodes = 128

		# Initial convolution layer
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_channels, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(num_channels)

		# Residual blocks
		self.residual_blocks = nn.ModuleList([
			nn.Sequential(
				nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
				nn.BatchNorm2d(num_channels),
				nn.ReLU(),
				nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
				nn.BatchNorm2d(num_channels),
				nn.ReLU(),
				nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
				nn.BatchNorm2d(num_channels)
			) for _ in range(num_residual_blocks)
		])

		# Calculate the size after the convolutional layers to define the first fully connected layer
		residual_output_size = (num_channels, input_size[0], input_size[1])  # Assuming padding keeps the spatial dimensions the same

		# Final layers
		self.fc1 = nn.Linear(residual_output_size[0]*residual_output_size[1]*residual_output_size[2], self.num_nodes)
		self.fc2 = nn.Linear(self.num_nodes, output_size[0])

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))

		for block in self.residual_blocks:
			residual = x
			x = block(x)
			x += residual
			x = F.relu(x)

		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		# x = x.view(x.size(0), self.output_size[0], self.output_size[1], self.output_size[2])
		return x