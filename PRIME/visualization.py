import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

# --- 1. Load Data ---
data = np.load("../data/prime/outputs/test_predictions.npz")
raw_preds = data["preds"]  # Shape: (19362, 18) -> Alternating Mean/Std
all_targets = data["targets"]  # Shape: (19362, 9)
plt.close("all")
# Feature naming setup matching the original paper schema
target_features = ["B_x GSM", "B_y GSM", "B_z GSM", "V_x GSE", "V_y GSE", "V_z GSE", "n", "P_dyn"]
hex_colors = ["#2c5263", "#0e6b70", "#1ca394", "#a2b36c", "#fcd075", "#fba257", "#ff7b4d", "#ff6b57"]

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6.2, 7.0), gridspec_kw={"height_ratios": [2.5, 1]})
plt.subplots_adjust(hspace=0.0)

# Ideal calibration reference baselines
ax1.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.2)
ax2.axhline(0, linestyle="--", color="black", linewidth=1.2)

# --- 2. Process Camporeale et al. Methodology ---
for idx, label_text in enumerate(target_features):
    if idx >= all_targets.shape[1]:
        break

    y_obs = all_targets[:, idx]
    mu = raw_preds[:, idx * 2]  # Extract alternating mean channel (\mu)
    sigma = raw_preds[:, idx * 2 + 1]  # Extract alternating standard deviation channel (\sigma)

    # Prevent potential divide-by-zero errors in case sigma is perfectly zero
    sigma = np.clip(sigma, 1e-6, None)

    # Step 1: Calculate standardized errors: \eta_i = (y_obs - \mu) / (\sqrt{2} * \sigma)
    eta = (y_obs - mu) / (np.sqrt(2.0) * sigma)

    # Step 2: Compute Gaussian forecast probability: \Phi_i = 0.5 * [erf(\eta_i) + 1]
    phi = 0.5 * (erf(eta) + 1.0)

    # Step 3: Compute continuous empirical CDF (Observed vs Predicted Frequency)
    # Sorting \Phi values automatically replicates the Heaviside step summation C(\phi)
    # and matches an evaluation grid spanning 1/N to 1.0 sequentially without binning.
    predicted_frequency = np.sort(phi)
    observed_frequency = np.linspace(1.0 / len(phi), 1.0, len(phi))

    color = hex_colors[idx % len(hex_colors)]

    # Format text strings to LaTeX syntax (e.g., B_x GSM -> $B_x$ GSM)
    if "_" in label_text:
        parts = label_text.split(" ")
        base = parts[0].split("_")
        latex_label = f"${base[0]}_{{{base[1]}}}$"
        if len(parts) > 1:
            latex_label += f" {' '.join(parts[1:])}"
        label_text = latex_label

    # Plot perfectly continuous unbinned trajectories
    ax1.plot(predicted_frequency, observed_frequency, color=color, linewidth=1.5, label=label_text)
    ax2.plot(predicted_frequency, observed_frequency - predicted_frequency, color=color, linewidth=1.5)

# --- 3. Polish Aesthetics & Limits ---
ax1.set_title("Reliability Diagram", fontsize=14, pad=12)
ax1.set_ylabel("Observed Frequency", fontsize=11)
ax1.set_xlim(0.0, 1.0)
ax1.set_ylim(0.0, 1.0)
ax1.legend(loc="lower right", frameon=False, fontsize=10, labelspacing=0.35)

ax2.set_xlabel("Predicted Frequency", fontsize=11)
ax2.set_ylabel("Under/Over-\nEstimation", fontsize=11)
ax2.set_ylim(-0.15, 0.15)  # Matches standard tight visual threshold spacing from report

ax1.tick_params(axis="x", which="both", bottom=True, labelbottom=False)
ax2.tick_params(axis="x", which="both", bottom=True)

plt.tight_layout()
plt.savefig("./reliability_diagram_paper_exact.png", dpi=300, bbox_inches="tight")
