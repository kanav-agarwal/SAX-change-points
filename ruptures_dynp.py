from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
from ruptures.exceptions import BadSegmentationParameters

selected = ['CO(GT)', 'NO2(GT)', 'C6H6(GT)']

project_root = Path(__file__).parent

data = pd.read_csv(
    project_root / "processed" / "z_normalized_data.csv",
    parse_dates=["Date_Time"]
)

# Load change point counts from SAX analysis
cp_df = pd.read_csv(project_root / "processed" / "change_point_counts.csv")
change_points_counts = (
    cp_df
    .groupby("variable")[["delta", "count"]]
    .apply(lambda x: dict(zip(x["delta"], x["count"])))
    .to_dict()
)

# ==============================================================================
# Detect Change Points using Ruptures (Dynp Algorithm)
# ==============================================================================

ruptures_change_points = {}

for col in selected:
    signal = data[col].values.reshape(-1, 1)
    ruptures_change_points[col] = {}

    L = len(signal)
    min_size = max(5, L // 200)
    jump = 2
    max_bkps = max(1, L // min_size - 1)

    for delta, cp_sax in change_points_counts[col].items():
        # Specify for a single delta value
        if delta != 7:
            continue

        # skip if no SAX change points
        if cp_sax == 0:
            ruptures_change_points[col][delta] = []
            continue

        # cap n_bkps to what Dynp can handle
        # n_bkps = min(cp_sax, max_bkps)
        n_bkps = cp_sax

        if n_bkps <= 0:
            ruptures_change_points[col][delta] = []
            print(f"{col}, delta={delta}: n_bkps too small, skipping Dynp")
            continue

        # Initialize Dynp
        algo = rpt.Dynp(model="l2", min_size=min_size, jump=jump).fit(signal)

        try:
            cp_indices = algo.predict(n_bkps=n_bkps)
        except BadSegmentationParameters:
            print(f"{col}, delta={delta}: BadSegmentationParameters, skipping.")
            ruptures_change_points[col][delta] = []
            continue
        except Exception as e:
            print(f"{col}, delta={delta}: Dynp failed ({e}), skipping.")
            ruptures_change_points[col][delta] = []
            continue

        cp_indices = [cp for cp in cp_indices if cp < L]
        ruptures_change_points[col][delta] = cp_indices

        # Plot change points
        save_dir = project_root / "figures" / "dynp_change_points" / col
        save_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        plt.scatter(
            data['Date_Time'].iloc[cp_indices],
            signal[cp_indices, 0],
            label=f"{col} - delta {delta}",
            color='red'
        )
        plt.title(f"Ruptures Dynp Change Points - {col} (delta={delta})")
        plt.xlabel("Time")
        plt.ylabel("Z-Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f"{delta}.png")
        plt.close()


print("Done. Ruptures change point plots saved.")