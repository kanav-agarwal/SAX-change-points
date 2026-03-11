from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt

project_root = Path(__file__).parent

# Load data
data = pd.read_csv(
    project_root / "processed" / "z_normalized_mixed_air_temp.csv"
)

time = data["time (s)"]
signal = data["AHU Mixed Air Temperature (°K)"].values.reshape(-1, 1)

L = len(signal)

# Fixed number of change points
n_bkps = 8

# Dynp parameters
min_size = max(5, L // 200)
jump = 1

# Fit ruptures model
algo = rpt.Dynp(model="l2", min_size=min_size, jump=jump).fit(signal)

# Predict change points
cp_indices = algo.predict(n_bkps=n_bkps)

# Remove final index if present
cp_indices = [cp for cp in cp_indices if cp < L]

# Create dataframe of detected change points
cp_records = []

for cp in cp_indices:
    cp_records.append({
        "variable": "AHU Mixed Air Temperature (°K)",
        "index": cp,
        "time_s": time.iloc[cp]
    })

cp_df = pd.DataFrame(cp_records)

# Save CSV
output_path = project_root / "processed" / "ruptures_change_points.csv"
cp_df.to_csv(output_path, index=False)

print(f"Change points saved to {output_path}")

# Save directory
save_dir = project_root / "figures" / "ruptures_change_points"
save_dir.mkdir(parents=True, exist_ok=True)

# Plot
plt.figure(figsize=(12,6))

plt.plot(time, signal, label="Mixed Air Temperature")

plt.scatter(
    time.iloc[cp_indices],
    signal[cp_indices, 0],
    color="red",
    label="Change Points"
)

plt.xlabel("Time (s)")
plt.ylabel("Z-Score")
plt.title("Ruptures Change Points - AHU Mixed Air Temperature")
plt.legend()

plt.tight_layout()
plt.savefig(save_dir / "ruptures_mixed_air_temp.png")
plt.close()

print("Done. Change point plot saved.")