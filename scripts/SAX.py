import csv
import pandas as pd
import numpy as np
import string
import os
from pathlib import Path
import matplotlib.pyplot as plt
import ruptures as rpt
from ruptures.exceptions import BadSegmentationParameters
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pyts.approximation import SymbolicAggregateApproximation

selected = ['CO(GT)', 'NO2(GT)', 'C6H6(GT)']

# Get the project root directory (sax_alogrithm)
project_root = Path(__file__).parent.parent

data = pd.read_csv(
    project_root / "processed" / "z_normalized_data.csv",
    parse_dates=["Date_Time"]
)

# Extract just the numeric values for SAX
scaled_values = data[selected].values

# 
# 
#
# Apply SAX and Plot Results
#
# 
# 

# --- 4) SAX Transformation ---
sax = SymbolicAggregateApproximation(
    n_bins=8,
    strategy="quantile"
)

# SAX expects shape (n_samples, n_timestamps)
sax_symbols = sax.fit_transform(scaled_values.T).T

# Reorder symbols for better visualization
letters = list(string.ascii_lowercase[:sax.n_bins])

try:
    # If sax_symbols are strings
    numeric_sax = np.array([[letters.index(val) for val in row] for row in sax_symbols])
except ValueError:
    # If sax_symbols are already numeric
    numeric_sax = sax_symbols.astype(int)

for i, col in enumerate(selected):
    plt.figure(figsize=(12, 6))
    plt.step(
        data["Date_Time"],
        numeric_sax[:, i],
        where="post",
        label=col,
        zorder=1
    )
    plt.title(f"SAX Symbolic Representation - {col}")
    plt.xlabel("Time")
    plt.ylabel("Symbol Index")
    plt.yticks(range(8), letters)
    plt.legend()
    plt.tight_layout()
    plt.savefig(project_root / "figures" / "sax_transformed" / f"{col}.png")
    plt.close()

# 
# 
#
# Detect change points from SAX and Plot Results
#
# 
# 

# --- 5) Detect change points from SAX ---
all_change_points = {}
change_points_counts = {}
for i, col in enumerate(selected):
    series = numeric_sax[:, i]  # SAX symbols for this series
    change_points = {}

    for delta in range(1, 8):
        # Detect change points for this delta
        cp_indices = np.where(np.abs(np.diff(series)) >= delta)[0] + 1
        change_points[delta] = cp_indices.tolist()
        print(f"Processed {col} for delta={delta}")

    all_change_points[col] = change_points
    change_points_counts[col] = {delta: len(cps) for delta, cps in change_points.items()}

# --- Plot SAX change points (one plot per series and delta) ---
for col, delta_dict in all_change_points.items():
    for delta, cp in delta_dict.items():
        if len(cp) == 0:
            continue  # Skip plotting if no change points

        save_dir = project_root / "figures" / "sax_change_points" / col

        plt.figure(figsize=(12, 6))
        plt.scatter(
            data['Date_Time'].iloc[cp],
            numeric_sax[cp, selected.index(col)],
            label=f"{col} - delta {delta}",
            color='red'
        )
        plt.title(f"SAX Change Points - {col} (delta={delta})")
        plt.xlabel("Time")
        plt.ylabel("SAX Symbol Index")
        plt.yticks(range(len(letters)), letters)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{delta}.png")
        plt.close()

print(change_points_counts)