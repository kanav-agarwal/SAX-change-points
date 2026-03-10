import pandas as pd
import numpy as np
import csv
import string
from pathlib import Path
import matplotlib.pyplot as plt
from pyts.approximation import SymbolicAggregateApproximation

project_root = Path(__file__).parent

time_col = "time (s)"

# List of variables to process
variables = [
    "Cooling Coil Heat Transfer Rate (W)", 
    "Cooling Coil Discharge Air Temperature (°K)",
    "AHU Supply Air Temperature (°K)",
    "Control Signal for AHU Cooling Coil Valve from BAS (fraction, 0: valve should be fully closed to 1: valve should be fully open)",
    "Chilled Water Flow Rate of the Chiller (kg/s)",
    "Power Consumption of the Chiller (W)",
    "Cooling Tower Fan Speed Control Signal (fraction, 0: fan speed should be 0% to 1:fan speed should be 100%)",
    "Flow Rate of the Condenser Water Loop (kg/s)"
]

for var_col in variables:
    sanitized = (var_col.replace(" ", "_")
                 .replace("(", "")
                 .replace(")", "")
                 .replace("°", "deg")
                 .replace("/", "_")
                 .replace(",", "")
                 .replace(":", "")
                 .replace("?", "")
                 .replace("*", "")
                 .replace("<", "")
                 .replace(">", "")
                 .replace("|", "")
                 .replace('"', "")
                 .lower())

    # ==============================================================================
    # Load Z-Normalized Data
    # ==============================================================================

    data = pd.read_csv(
        project_root / "processed" / "z_normalized" / f"z_normalized_{sanitized}.csv"
    )

    # Ensure numeric
    data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
    data[var_col] = pd.to_numeric(data[var_col], errors="coerce")

    data = data.dropna(subset=[time_col, var_col])

    scaled_values = data[[var_col]].values  # shape (n_samples, 1)

    # Determine time range for plotting
    time_min = data[time_col].min()
    time_max = data[time_col].max()
    padding = (time_max - time_min) * 0.02

    # ==============================================================================
    # Apply SAX
    # ==============================================================================

    sax = SymbolicAggregateApproximation(
        n_bins=4,
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
    plt.title(f"SAX Symbolic Representation - {var_col}")
    plt.xlabel("Time (s)")
    plt.ylabel("Symbol Index")
    plt.yticks(range(4), letters)
    plt.xlim(time_min - padding, time_max + padding)
    plt.tight_layout()
    plt.savefig(project_root / "figures" / "sax_transformed" / f"{sanitized}.png")
    plt.close()

    # ==============================================================================
    # Detect Change Points from SAX and Plot Results
    # ==============================================================================

    series = numeric_sax[:, 0]
    letters = list(string.ascii_lowercase[:sax.n_bins])

    change_point_records = []

    for delta in range(1, 4):

        cp_indices = np.where(np.abs(np.diff(series)) >= delta)[0] + 1

        # ---- Store detailed information ----
        if delta == 1:
            for idx in cp_indices:
                change_point_records.append({
                    "variable": var_col,
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

        # Plot z-normalized signal
        plt.plot(
            data[time_col],
            data[var_col],
            label="Z-Normalized Value",
            zorder=1
        )

        # Overlay change points
        plt.scatter(
            data[time_col].iloc[cp_indices],
            data[var_col].iloc[cp_indices],
            color="red",
            label=f"Change Points (delta={delta})",
            zorder=2
        )

        plt.title(f"SAX Change Points on Z-Normalized Data - {var_col} (delta={delta})")
        plt.xlabel("Time (s)")
        plt.ylabel("Z-Normalized Value")
        plt.xlim(time_min - padding, time_max + padding)
        plt.legend()
        plt.tight_layout()

        (save_dir / sanitized).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / sanitized / f"delta_{delta}.png")
        plt.close()

    # ==============================================================================
    # Save Detailed Change Point Data
    # ==============================================================================

    cp_df = pd.DataFrame(change_point_records)

    cp_df.to_csv(
        project_root / "processed" / "sax_change_points" / f"change_points_{sanitized}_4bins.csv",
        index=False
    )

    print(f"SAX algorithm complete for {var_col}. Graphs and detailed change point data saved.")