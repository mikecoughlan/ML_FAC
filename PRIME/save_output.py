import numpy as np
import pandas as pd
import spacepy.coordinates as spc
import spacepy.time as spt

data = np.load("data/prime/outputs/test_predictions.npz")

times = data["times"]  # Shape is likely (N,) or (N, 1)
preds = data["preds"]  # Shape is (N, 18)

base_cols = [
    "Vx",  # Vx
    "Vy",  # Vy
    "Vz",  # Vz
    "proton_density",  # proton_density
    "T_p",
    "T",  # T
    "BX_GSE",  # BX_GSE
    "BY_GSE",  # BY_GSE not GSM
    "BZ_GSE",  # BZ_GSE
]

pred_cols = []
for col in base_cols:
    pred_cols.append(col)
    pred_cols.append(f"{col}_std")

all_column_names = ["Epoch"] + pred_cols

# Ensure 'times' is 2D and stack horizontally
if times.ndim == 1:
    times = times[:, np.newaxis]

combined_array = np.hstack((times, preds))

df = pd.DataFrame(combined_array, columns=all_column_names)

df = df.sort_values("Epoch").reset_index(drop=True)
df["Epoch"] = pd.to_datetime(df["Epoch"], format="%Y%m%d %H:%M:%S")

# GSE -> GSM

b_gse = np.column_stack([df["BX_GSE"], df["BY_GSE"], df["BZ_GSE"]])
c_gse = spc.Coords(b_gse, "GSE", "car")
c_gse.ticks = spt.Ticktock(df["Epoch"].tolist(), "UTC")
c_gsm = c_gse.convert("GSM", "car")

df["BY_GSM"] = c_gsm.data[:, 1]
df["BZ_GSM"] = c_gsm.data[:, 2]

dummy_y_gse = np.zeros_like(b_gse)
dummy_y_gse[:, 1] = 1.0
c_dummy = spc.Coords(dummy_y_gse, "GSE", "car")
c_dummy.ticks = c_gse.ticks
c_dummy_gsm = c_dummy.convert("GSM", "car")

# Extract the cosine and sine of the dipole tilt angle
cos_theta = c_dummy_gsm.data[:, 1]
sin_theta = c_dummy_gsm.data[:, 2]

# Calculate variances (Standard Deviation squared)
var_y_gse = df["BY_GSE_std"] ** 2
var_z_gse = df["BZ_GSE_std"] ** 2

# Apply variance rotation (assuming 0 covariance) and take the square root
df["BY_GSM_std"] = np.sqrt(cos_theta**2 * var_y_gse + sin_theta**2 * var_z_gse)
df["BZ_GSM_std"] = np.sqrt(sin_theta**2 * var_y_gse + cos_theta**2 * var_z_gse)

# Drop all GSE columns so the output is strictly GSM
gse_columns_to_drop = ["BY_GSE", "BZ_GSE", "BX_GSE_std", "BY_GSE_std", "BZ_GSE_std"]
df = df.drop(columns=gse_columns_to_drop)

# Convert Epoch back to string format
df["Epoch"] = df["Epoch"].dt.strftime("%Y-%m-%d %H:%M:%S")

df.to_feather("data/prep/prime_predictions.feather")
print("DataFrame saved to 'data/prep/prime_predictions.feather' with columns:")
print(df.columns.tolist())
print(df.head())
