import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os

project_root = Path(__file__).parent

time_col = "time (s)"

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

    # -----------------------------
    # Sanitize (same as SAX)
    # -----------------------------
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

    # -----------------------------
    # Load Z-normalized data
    # -----------------------------
    data = pd.read_csv(
        project_root / "processed" / "z_normalized" / f"z_normalized_{sanitized}.csv"
    )

    data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
    data[var_col] = pd.to_numeric(data[var_col], errors="coerce")
    data = data.dropna(subset=[time_col, var_col])

    # Time bounds
    time_min = data[time_col].min()
    time_max = data[time_col].max()
    padding = (time_max - time_min) * 0.02

    # -----------------------------
    # Load Ruptures Change Points
    # -----------------------------
    cp_file = (
        project_root
        / "processed"
        / "ruptures_change_points"
        / f"change_points_{sanitized}_ruptures.csv"
    )

    try:
        cp_df = pd.read_csv(cp_file)
    except FileNotFoundError:
        print(f"Missing ruptures file for {var_col}")
        continue

    # Use indices (preferred for alignment)
    cp_indices = cp_df["index"].astype(int).values

    # -----------------------------
    # Plot (same style as SAX)
    # -----------------------------
    save_dir = project_root / "figures" / "ruptures_change_points"
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))

    # Plot signal
    plt.plot(
        data[time_col],
        data[var_col],
        label="Z-Normalized Value",
        zorder=1
    )

    # Plot change points
    plt.scatter(
        data[time_col].iloc[cp_indices],
        data[var_col].iloc[cp_indices],
        color="red",
        label="Ruptures Change Points",
        zorder=2
    )

    plt.title(f"Ruptures Change Points on Z-Normalized Data - {var_col}")
    plt.xlabel("Time (s)")
    plt.ylabel("Z-Normalized Value")
    plt.xlim(time_min - padding, time_max + padding)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_dir / f"{sanitized}.png")
    plt.close()

    print(f"Plotted ruptures change points for {var_col}")