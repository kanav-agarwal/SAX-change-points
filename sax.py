import pandas as pd
import numpy as np
import string
from pathlib import Path
import matplotlib.pyplot as plt
from pyts.approximation import SymbolicAggregateApproximation

selected = ['CO(GT)', 'NO2(GT)', 'C6H6(GT)']

project_root = Path(__file__).parent

data = pd.read_csv(
    project_root / "processed" / "z_normalized_data.csv",
    parse_dates=["Date_Time"]
)

scaled_values = data[selected].values

# ==============================================================================
# Apply SAX and Plot Results
# ==============================================================================

# Initialize SAX with 8 bins
sax = SymbolicAggregateApproximation(
    n_bins=8,
    strategy="quantile"
)

# Reshape for SAX (requires n_timestamps x n_samples)
sax_symbols = sax.fit_transform(scaled_values.T).T

# Convert symbols to numeric indices for visualization
letters = list(string.ascii_lowercase[:sax.n_bins])

try:
    numeric_sax = np.array([[letters.index(val) for val in row] for row in sax_symbols])
except ValueError:
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

# ==============================================================================
# Detect Change Points from SAX and Plot Results
# ==============================================================================

# Extract change points for each series and delta value
all_change_points = {}
change_points_counts = {}
for i, col in enumerate(selected):
    series = numeric_sax[:, i]
    change_points = {}

    for delta in range(1, 8):
        cp_indices = np.where(np.abs(np.diff(series)) >= delta)[0] + 1
        change_points[delta] = cp_indices.tolist()
        print(f"Processed {col} for delta={delta}")

    all_change_points[col] = change_points
    change_points_counts[col] = {delta: len(cps) for delta, cps in change_points.items()}

# Plot change points for each series and delta
for col, delta_dict in all_change_points.items():
    for delta, cp in delta_dict.items():
        if len(cp) == 0:
            continue

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
        plt.savefig(save_dir / f"{delta}.png")
        plt.close()

print(change_points_counts)