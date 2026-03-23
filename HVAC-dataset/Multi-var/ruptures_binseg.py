import pandas as pd
import numpy as np
from pathlib import Path
import ruptures as rpt

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

model = "l2"
jump = 2

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

    signal = data[[var_col]].values
    L = len(signal)

    # -----------------------------
    # Load SAX change points
    # -----------------------------
    sax_cp_path = (
        project_root
        / "processed"
        / "sax_change_points"
        / f"change_points_{sanitized}_4bins.csv"
    )

    sax_cp_df = pd.read_csv(sax_cp_path)

    # Only delta = 1
    sax_cp_df = sax_cp_df[sax_cp_df["symbol_jump"] >= 1]

    n_bkps = len(sax_cp_df)

    # Safety cap
    max_bkps = L // 3
    n_bkps = min(n_bkps, max_bkps)

    print(f"{var_col}: using n_bkps = {n_bkps}")

    # -----------------------------
    # Run Ruptures
    # -----------------------------
    algo = rpt.Binseg(model=model, jump=jump).fit(signal)

    try:
        cp_indices = algo.predict(n_bkps=n_bkps)
    except Exception as e:
        print(f"{var_col}: ruptures failed ({e})")
        continue

    cp_indices = [cp for cp in cp_indices if cp < L]

    # -----------------------------
    # Store (SAX-like format)
    # -----------------------------
    change_point_records = []

    for idx in cp_indices:
        if idx == 0:
            continue

        change_point_records.append({
            "variable": var_col,
            "index": int(idx),
            "time_s": float(data[time_col].iloc[idx]),
            "value_before": float(signal[idx - 1, 0]),
            "value_after": float(signal[idx, 0]),
            "value_jump": float(abs(signal[idx, 0] - signal[idx - 1, 0]))
        })

    cp_out = pd.DataFrame(change_point_records)

    # -----------------------------
    # Save CSV
    # -----------------------------
    output_path = (
        project_root
        / "processed"
        / "ruptures_change_points"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    cp_out.to_csv(
        output_path / f"change_points_{sanitized}_ruptures.csv",
        index=False
    )

    print(f"Saved ruptures change points for {var_col}")