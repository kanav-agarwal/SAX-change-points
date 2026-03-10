from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ==============================================================================
# Load, Preprocess, and Visualize HVAC Dataset
# ==============================================================================

project_root = Path(__file__).parent

# Load CSV (standard comma-separated)
df = pd.read_csv(project_root.parent / "Public_ScientificData_AHUFaults" / "06_G36-Degrad" / "BaselineSystem.csv")

# Clean column names (remove accidental whitespace)
df.columns = df.columns.str.strip()

# Select time column
time_col = "time (s)"

# Ensure numeric
df[time_col] = pd.to_numeric(df[time_col], errors="coerce")

# Drop rows where time is missing
df = df.dropna(subset=[time_col])

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
    # Ensure variable is numeric
    df[var_col] = pd.to_numeric(df[var_col], errors="coerce")

    data = df[[time_col, var_col]].copy()

    # ------------------------------------------------------------------------------
    # Impute Missing Values (mean strategy)
    # ------------------------------------------------------------------------------

    imputer = SimpleImputer(strategy="mean")
    data[[var_col]] = imputer.fit_transform(data[[var_col]])

    # ------------------------------------------------------------------------------
    # Plot Raw Time Series
    # ------------------------------------------------------------------------------

    time_min = data[time_col].min()
    time_max = data[time_col].max()
    padding = (time_max - time_min) * 0.02  # 2% padding

    plt.figure(figsize=(12, 6))
    plt.plot(data[time_col], data[var_col])
    plt.title(f"Raw {var_col}")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.xlim(time_min - padding, time_max + padding)
    plt.tight_layout()
    
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
    plt.savefig(project_root / "figures" / "raw" / f"raw_{sanitized}.png")
    plt.close()

    # ==============================================================================
    # Z-Normalization
    # ==============================================================================

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(data[[var_col]])

    scaled = pd.DataFrame(
        scaled_values,
        columns=[var_col]
    )

    scaled[time_col] = data[time_col].values
    scaled = scaled[[time_col, var_col]]

    # ------------------------------------------------------------------------------
    # Plot Z-Normalized Series
    # ------------------------------------------------------------------------------

    plt.figure(figsize=(12, 6))
    plt.plot(scaled[time_col], scaled[var_col])
    plt.title(f"Z-Normalized {var_col}")
    plt.xlabel("Time (s)")
    plt.ylabel("Z-Score")
    plt.xlim(time_min - padding, time_max + padding)
    plt.tight_layout()
    plt.savefig(project_root / "figures" / "z_normalized" / f"z_normalized_{sanitized}.png")
    plt.close()

    # ------------------------------------------------------------------------------
    # Save Processed Data
    # ------------------------------------------------------------------------------

    scaled.to_csv(
        project_root / "processed" / "z_normalized" / f"z_normalized_{sanitized}.csv",
        index=False
    )

    print(f"Preprocessing complete for {var_col}. Data and plots saved.")