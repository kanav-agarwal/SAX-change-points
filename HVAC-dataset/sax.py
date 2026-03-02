import pandas as pd
import numpy as np
import csv
import string
from pathlib import Path
import matplotlib.pyplot as plt
from pyts.approximation import SymbolicAggregateApproximation

project_root = Path(__file__).parent

time_col = "time (s)"
temp_col = "AHU Mixed Air Temperature (°K)"

# ==============================================================================
# Load Z-Normalized Data
# ==============================================================================

data = pd.read_csv(
    project_root / "processed" / "z_normalized_mixed_air_temp.csv"
)

# Ensure numeric
data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
data[temp_col] = pd.to_numeric(data[temp_col], errors="coerce")

data = data.dropna(subset=[time_col, temp_col])

scaled_values = data[[temp_col]].values  # shape (n_samples, 1)

# Determine time range for plotting
time_min = data[time_col].min()
time_max = data[time_col].max()
padding = (time_max - time_min) * 0.02

# ==============================================================================
# Apply SAX
# ==============================================================================

sax = SymbolicAggregateApproximation(
    n_bins=8,
    strategy="quantile"
)

# For univariate signal: reshape to (n_samples, n_timestamps)
sax_symbols = sax.fit_transform(scaled_values.T).T  # back to (n_samples, 1)

letters = list(string.ascii_lowercase[:sax.n_bins])

try:
    numeric_sax = np.array(
        [[letters.index(val) for val in row] for row in sax_symbols]
    )
except ValueError:
    numeric_sax = sax_symbols.astype(int)

# ==============================================================================
# Plot SAX Representation
# ==============================================================================

plt.figure(figsize=(12, 6))
plt.step(
    data[time_col],
    numeric_sax[:, 0],
    where="post",
    zorder=1
)
plt.title("SAX Symbolic Representation - AHU Mixed Air Temperature")
plt.xlabel("Time (s)")
plt.ylabel("Symbol Index")
plt.yticks(range(8), letters)
plt.xlim(time_min - padding, time_max + padding)
plt.tight_layout()
plt.savefig(project_root / "figures" / "sax_transformed" / "mixed_air_temp.png")
plt.close()

# ==============================================================================
# Detect Change Points from SAX and Plot Results
# ==============================================================================

series = numeric_sax[:, 0]
letters = list(string.ascii_lowercase[:sax.n_bins])

change_point_records = []

for delta in range(1, 8):

    cp_indices = np.where(np.abs(np.diff(series)) >= delta)[0] + 1

    # ---- Store detailed information ----
    if delta == 1:
        for idx in cp_indices:
            change_point_records.append({
                "variable": "AHU Mixed Air Temperature (°K)",
                "index": int(idx),
                "time_s": float(data[time_col].iloc[idx]),
                "sax_before": int(series[idx - 1]),
                "sax_after": int(series[idx]),
                "symbol_before": letters[series[idx - 1]],
                "symbol_after": letters[series[idx]],
                "symbol_jump": int(abs(series[idx] - series[idx - 1]))
            })

    # ---- Plot change points (same behavior as before) ----
    save_dir = project_root / "figures" / "sax_change_points"

    plt.figure(figsize=(12, 6))
    plt.scatter(
        data[time_col].iloc[cp_indices],
        numeric_sax[cp_indices, 0],
        color="red",
        label=f"delta = {delta}"
    )

    plt.title(f"SAX Change Points - AHU Mixed Air Temperature (delta={delta})")
    plt.xlabel("Time (s)")
    plt.ylabel("SAX Symbol Index")
    plt.yticks(range(len(letters)), letters)
    plt.xlim(time_min - padding, time_max + padding)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"delta_{delta}.png")
    plt.close()

# ==============================================================================
# Save Detailed Change Point Data
# ==============================================================================

cp_df = pd.DataFrame(change_point_records)

cp_df.to_csv(
    project_root / "processed" / "mixed_air_temp_change_points_detailed.csv",
    index=False
)

print("SAX algorithm complete. Graphs and detailed change point data saved.")