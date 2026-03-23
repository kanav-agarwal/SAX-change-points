import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent

time_col = "time (s)"
ground_truth_col = "Cooling Mode Control Signal (1: free cooling , 2: partial mechanical, 3: full mechanical, 4: off)"

# --------------------------------------------------
# Parameters
# --------------------------------------------------

# tolerance window for matching change points (seconds)
tolerance = 600

# list of variables you analyzed
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

# --------------------------------------------------
# Load original dataset to get ground truth changes
# --------------------------------------------------

data = pd.read_csv(project_root.parent / "Public_ScientificData_AHUFaults" / "06_G36-Degrad" / "BaselineSystem.csv")
data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
data[ground_truth_col] = pd.to_numeric(data[ground_truth_col], errors="coerce")

data = data.dropna(subset=[time_col, ground_truth_col])

gt_series = data[ground_truth_col].values

gt_indices = np.where(np.diff(gt_series) != 0)[0] + 1
gt_times = data[time_col].iloc[gt_indices].values

# --------------------------------------------------
# Evaluation Function
# --------------------------------------------------

def evaluate_change_points(pred_times, gt_times, tolerance):

    matched_gt = set()
    matched_pred = set()
    time_errors = []

    for i, p in enumerate(pred_times):
        diffs = np.abs(gt_times - p)

        if len(diffs) == 0:
            continue

        min_idx = np.argmin(diffs)

        if diffs[min_idx] <= tolerance and min_idx not in matched_gt:
            matched_gt.add(min_idx)
            matched_pred.add(i)
            time_errors.append(diffs[min_idx])

    TP = len(matched_pred)
    FP = len(pred_times) - TP
    FN = len(gt_times) - len(matched_gt)

    accuracy = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    avg_error = np.mean(time_errors) if time_errors else None
    med_error = np.median(time_errors) if time_errors else None

    return TP, FP, FN, accuracy, precision, recall, f1, avg_error, med_error


# --------------------------------------------------
# Run evaluation for each variable
# --------------------------------------------------

results = []

for var in variables:

    filename = (var.replace(" ", "_")
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
    cp_file = project_root / "processed" / "ruptures_change_points" / f"change_points_{filename}_ruptures.csv"

    try:
        cp_df = pd.read_csv(cp_file)
    except FileNotFoundError:
        print(f"Missing file for {var}")
        continue

    pred_times = cp_df["time_s"].values

    TP, FP, FN, accuracy, precision, recall, f1, avg_err, med_err = evaluate_change_points(
        pred_times, gt_times, tolerance
    )

    results.append({
        "Variable": var,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Avg Time Error (s)": avg_err,
        "Median Time Error (s)": med_err
    })


# --------------------------------------------------
# Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    project_root / "processed" / "ruptures_evaluation.csv",
    index=False
)

print(results_df)
print("\nEvaluation complete. Results saved.")