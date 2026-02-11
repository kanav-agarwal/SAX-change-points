from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ==============================================================================
# Load, Preprocess, and Visualize Air Quality Data
# ==============================================================================

project_root = Path(__file__).parent

df = pd.read_csv(
    project_root / "raw_data" / "AirQualityUCI.csv",
    sep=";",
    decimal=","
)

# clean empty columns
df = df.dropna(axis=1, how="all")

# Fix time format (18.00.00 -> 18:00:00)
df["Time"] = df["Time"].str.replace(".", ":", regex=False)

# Combine date and time with explicit format
df["Date_Time"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce"
)

df = df.dropna(subset=["Date_Time"])

# choose 3 variables
selected = ['CO(GT)', 'NO2(GT)', 'C6H6(GT)']
data_selected = df[['Date_Time'] + selected].copy()

# Impute missing values using mean strategy
imputer = SimpleImputer(missing_values=-200, strategy='mean')
imputed_values = imputer.fit_transform(data_selected[selected])
data_imputed = pd.DataFrame(imputed_values, columns=selected)
data_imputed['Date_Time'] = data_selected['Date_Time'].values
data = data_imputed

# Plot raw time series
plt.figure(figsize=(12,6))
for col in selected:
    plt.plot(data['Date_Time'], data[col], label=col)
plt.title("Raw Time Series")
plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.savefig(project_root / "figures" / "raw_time_series.png")
plt.close()

# ==============================================================================
# Normalize and Plot Time Series
# ============================================================================== 

# Apply Z-normalization
scaler = StandardScaler()
scaled_values = scaler.fit_transform(data[selected])
scaled = pd.DataFrame(scaled_values, columns=selected)

plt.figure(figsize=(12,6))
for i,col in enumerate(selected):
    plt.plot(data['Date_Time'], scaled[col], label=col)
plt.title("Z-Normalized Time Series")
plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.savefig(project_root / "figures" / "z_normalized.png")
plt.close()

# Save Z-normalized data with timestamps
scaled_with_time = scaled.copy()
scaled_with_time["Date_Time"] = data["Date_Time"].values
scaled_with_time = scaled_with_time[["Date_Time"] + selected]

# Save to CSV
scaled_with_time.to_csv(
    project_root / "processed" / "z_normalized_data.csv",
    index=False
)

print("Preprocessing complete. Data and plots saved.")