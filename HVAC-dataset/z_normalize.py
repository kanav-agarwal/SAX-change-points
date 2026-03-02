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
df = pd.read_csv(project_root / "Public_ScientificData_AHUFaults" / "08_G36-HIL" / "BaselineSystem.csv")

# Clean column names (remove accidental whitespace)
df.columns = df.columns.str.strip()

# Select time and target variable
time_col = "time (s)"
temp_col = "AHU Mixed Air Temperature (°K)"

# Ensure numeric
df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")

# Drop rows where time is missing
df = df.dropna(subset=[time_col])

data = df[[time_col, temp_col]].copy()

# ------------------------------------------------------------------------------
# Impute Missing Values (mean strategy)
# ------------------------------------------------------------------------------

imputer = SimpleImputer(strategy="mean")
data[[temp_col]] = imputer.fit_transform(data[[temp_col]])

# ------------------------------------------------------------------------------
# Plot Raw Time Series
# ------------------------------------------------------------------------------

time_min = data[time_col].min()
time_max = data[time_col].max()
padding = (time_max - time_min) * 0.02  # 2% padding

plt.figure(figsize=(12, 6))
plt.plot(data[time_col], data[temp_col])
plt.title("Raw AHU Mixed Air Temperature Time Series")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (°K)")
plt.xlim(time_min - padding, time_max + padding)
plt.tight_layout()
plt.savefig(project_root / "figures" / "raw_mixed_air_temp.png")
plt.close()

# ==============================================================================
# Z-Normalization
# ==============================================================================

scaler = StandardScaler()
scaled_values = scaler.fit_transform(data[[temp_col]])

scaled = pd.DataFrame(
    scaled_values,
    columns=[temp_col]
)

scaled[time_col] = data[time_col].values
scaled = scaled[[time_col, temp_col]]

# ------------------------------------------------------------------------------
# Plot Z-Normalized Series
# ------------------------------------------------------------------------------

plt.figure(figsize=(12, 6))
plt.plot(scaled[time_col], scaled[temp_col])
plt.title("Z-Normalized AHU Mixed Air Temperature")
plt.xlabel("Time (s)")
plt.ylabel("Z-Score")
plt.xlim(time_min - padding, time_max + padding)
plt.tight_layout()
plt.savefig(project_root / "figures" / "z_normalized_mixed_air_temp.png")
plt.close()

# ------------------------------------------------------------------------------
# Save Processed Data
# ------------------------------------------------------------------------------

scaled.to_csv(
    project_root / "processed" / "z_normalized_mixed_air_temp.csv",
    index=False
)

print("Preprocessing complete. Data and plots saved.")