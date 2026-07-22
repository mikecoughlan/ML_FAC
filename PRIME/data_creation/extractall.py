import os

import numpy as np
import torch

# --- 1. Load Your Trained Model and Dataset ---
# Replace 'SWRegressor' with your exact LightningModule class name
from prime_torch import SWRegressor
from torch.utils.data import DataLoader

from data import SWDataModule  # Replace with your actual data module file

# Initialize data module with your base configuration
data_dir = "../data/prime/"
dm = SWDataModule(data_dir=data_dir, batch_size=256)
dm.setup(stage="fit")  # Builds internal dataset structures for all splits

# Combine your split datasets into a single unified tracking sequence
# (Adapt these internal attributes if your DataModule names its datasets differently)
full_dataset = torch.utils.data.ConcatDataset([dm.train_dataset, dm.val_dataset, dm.test_dataset])

# Create a clean sequential DataLoader (CRITICAL: shuffle must be False)
inference_loader = DataLoader(full_dataset, batch_size=256, shuffle=False, num_workers=4, drop_last=False)

# Load your best trained model checkpoint weights
checkpoint_path = "../data/prime/checkpoints/final_model.ckpt"
model = SWRegressor.load_from_checkpoint(checkpoint_path)
model.eval()  # Disables dropout, batch norm updates, etc.

# Move model execution safely to hardware acceleration if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- 2. Run Inference Loop over Entire Dataset ---
all_preds = []
all_targets = []
all_times = []

# Optional Space Tracking variables if you are plotting Figure 6 trajectories
all_gse_x = []
all_gse_y = []
all_gse_z = []

print(f"Starting inference across the entire dataset ({len(full_dataset)} total samples)...")

with torch.no_grad():  # Disable gradient calculation to save memory/speed up loops
    for batch in inference_loader:
        # Unpack based on your custom batch structure
        # (e.g., timeseries, position, target, times = batch)
        timeseries, position, target, times = batch

        # Move tensors to active hardware accelerator device
        timeseries = timeseries.to(device)
        position = position.to(device)

        # Forward pass through model architecture
        y_hat = model(timeseries, position)  # Shape: [Batch, 18] (Interleaved Mean/Std)

        # Append outputs to collection lists
        all_preds.append(y_hat.cpu().numpy())
        all_targets.append(target.numpy())
        all_times.append(np.array(times))

        # If position contains your GSE coordinates [X, Y, Z], collect them here:
        # Assuming position shape is [Batch, 3] or encoded inside metadata:
        all_gse_x.append(position[:, 0].cpu().numpy())
        all_gse_y.append(position[:, 1].cpu().numpy())
        all_gse_z.append(position[:, 2].cpu().numpy())

# --- 3. Consolidate and Save Unified Arrays ---
print("Consolidating output tensors...")
preds_final = np.concatenate(all_preds, axis=0)
targets_final = np.concatenate(all_targets, axis=0)
times_final = np.concatenate(all_times, axis=0)

gse_x_final = np.concatenate(all_gse_x, axis=0)
gse_y_final = np.concatenate(all_gse_y, axis=0)
gse_z_final = np.concatenate(all_gse_z, axis=0)

# Write out unified file matrix to disk
save_dir = "../data/prime/outputs"
os.makedirs(save_dir, exist_ok=True)
filepath = os.path.join(save_dir, "entire_dataset_predictions.npz")

np.savez(filepath, preds=preds_final, targets=targets_final, times=times_final, gse_x=gse_x_final, gse_y=gse_y_final, gse_z=gse_z_final)

print(f"Successfully saved complete inference mapping to: {filepath}")
