from pathlib import Path
import pandas as pd
import os
import matplotlib.pyplot as plt
import ruptures as rpt

# -----------------------------
# Configuration
# -----------------------------
selected = ['CO(GT)', 'NO2(GT)', 'C6H6(GT)']
model = "l2"
jump = 2

project_root = Path(__file__).parent

# -----------------------------
# Load data
# -----------------------------
data = pd.read_csv(
    project_root / "processed" / "z_normalized_data.csv",
    parse_dates=["Date_Time"]
)

cp_df = pd.read_csv(
    project_root / "processed" / "change_point_counts.csv"
)

change_points_counts = (
    cp_df
    .groupby("variable")[["delta", "count"]]
    .apply(lambda x: dict(zip(x["delta"], x["count"])))
    .to_dict()
)

# -----------------------------
# Binseg using SAX counts
# -----------------------------
binseg_change_points = {}

for col in selected:
    signal = data[col].values.reshape(-1, 1)
    L = len(signal)

    binseg_change_points[col] = {}

    save_dir = project_root / "figures" / "binseg_change_points" / col
    os.makedirs(save_dir, exist_ok=True)

    for delta, cp_sax in change_points_counts[col].items():
        # Only plot for a specific delta value
        # if delta != 5:
        #     continue

        # Skip if SAX found nothing
        # if cp_sax == 0:
        #     binseg_change_points[col][delta] = []
        #     continue

        # Cap breakpoints to something sane
        max_bkps = L // 5
        n_bkps = min(cp_sax, max_bkps)

        algo = rpt.Binseg(
            model=model,
            jump=jump
        ).fit(signal)

        try:
            cp_indices = algo.predict(n_bkps=n_bkps)
        except Exception as e:
            print(f"{col}, delta={delta}: Binseg failed ({e})")
            binseg_change_points[col][delta] = []
            continue

        # Drop last point (= end of signal)
        cp_indices = [cp for cp in cp_indices if cp < L]
        binseg_change_points[col][delta] = cp_indices

        # -----------------------------
        # Plot
        # -----------------------------
        plt.figure(figsize=(12, 6))
        # plt.plot(data["Date_Time"], signal, label=f"{col} Signal")

        plt.scatter(
            data["Date_Time"].iloc[cp_indices],
            signal[cp_indices, 0],
            color="red",
            label=f"Binseg (n_bkps={n_bkps}, delta={delta})"
        )

        plt.title(f"Binseg Change Points - {col} (delta={delta})")
        plt.xlabel("Time")
        plt.ylabel("Z-Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f"{delta}.png")
        plt.close()

print("Done. Plots saved: binseg.")
