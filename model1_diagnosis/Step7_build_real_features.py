"""
STEP 7 - build_real_features.py
===============================
Runs every REAL NASA discharge cycle through extract_features() (step 5) -
the same function the synthetic training curves went through - and writes
one table:

    real_features.csv   one row per discharge cycle:
                        cell, cycle, measured capacity, + the 26 features

Step 8 feeds this table to the trained model to diagnose the real cells.

RUN:   python step7_build_real_features.py
"""

import os
import sys
import csv
import glob
import importlib
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _name in ("step5_feature_extractor", "Step5_feature_extractor", "feature_extractor"):
    try:
        _fx = importlib.import_module(_name)
        break
    except ModuleNotFoundError:
        continue
else:
    raise SystemExit("Cannot find the feature extractor (step 5) in this folder.")
extract_features, features_to_row, FEATURE_NAMES = (_fx.extract_features, _fx.features_to_row,
                                                    _fx.FEATURE_NAMES)

# ------------------------------------------------------------------ config
DATA_DIR = r"C:\Users\99017994\OneDrive - LTTS\Documents\1. BatteryAgingARC-FY08Q4"
CELLS = ["B0005", "B0006", "B0007", "B0018"]
OUT_DIR = r"C:\Users\99017994\battery_out"
OUT_CSV = "real_features.csv"
MIN_CURRENT_A = 0.1      # below this the battery is resting, not discharging


def find_mat(cell):
    path = os.path.join(DATA_DIR, cell + ".mat")
    if os.path.isfile(path):
        return path
    hits = glob.glob(os.path.join(DATA_DIR, "**", cell + ".mat"), recursive=True)
    return hits[0] if hits else None


def load_cycles(path, cell):
    mat = loadmat(path)
    key = cell if cell in mat else next(k for k in mat if not k.startswith("__"))
    cyc = mat[key][0, 0]["cycle"]
    return cyc[0] if cyc.ndim == 2 else cyc


def discharge_vq(c):
    """(voltage, capacity, measured capacity) of the REAL discharge only -
    same cleaning as step 2: no rest at the start, no bounce-back at the end."""
    d = c["data"]
    d = d[0, 0] if d.ndim == 2 else d
    v = np.squeeze(d["Voltage_measured"]).astype(float)
    i = np.squeeze(d["Current_measured"]).astype(float)
    t = np.squeeze(d["Time"]).astype(float)
    if v.size < 20 or v.size != t.size:
        return None
    q = np.concatenate([[0.0], np.cumsum(np.abs(i[:-1]) * np.diff(t))]) / 3600.0
    k = np.argmin(v)
    on = np.abs(i[:k + 1]) > MIN_CURRENT_A
    if on.sum() < 20:
        return None
    return v[:k + 1][on], q[:k + 1][on], float(np.squeeze(d["Capacity"]))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print("STEP 7 - Features of every real NASA discharge cycle")
    print("=" * 70)

    rows, skipped = [], 0
    for cell in CELLS:
        path = find_mat(cell)
        if path is None:
            print(f"  {cell}: not found, skipping")
            continue
        cyc_no = used = 0
        for c in load_cycles(path, cell):
            if str(np.squeeze(c["type"])) != "discharge":
                continue
            cyc_no += 1
            res = discharge_vq(c)
            feats = extract_features(*res[:2]) if res else None
            if feats is None or not np.isfinite(feats["capacity_Ah"]):
                skipped += 1
                continue
            rows.append([cell, cyc_no, res[2]] + features_to_row(feats))
            used += 1
        print(f"  {cell}: {used} discharge cycles -> features")

    if not rows:
        print("\nNo cycles processed. Check DATA_DIR at the top of this file.")
        return 1

    out_path = os.path.join(OUT_DIR, OUT_CSV)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cell", "cycle", "measured_capacity_Ah"] + FEATURE_NAMES)
        w.writerows(rows)
    print(f"\nRows written: {len(rows)}   features: {len(FEATURE_NAMES)}   skipped: {skipped}")
    print(f"Saved: {out_path}")

    # sanity check on the first cell found: capacity and PE/NE ratio should both fall
    first = [r for r in rows if r[0] == rows[0][0]]
    r_col = 3 + FEATURE_NAMES.index("ratio_high_low")
    print(f"\n{rows[0][0]} sanity check:")
    print(f"  capacity     {first[0][2]:.3f} -> {first[-1][2]:.3f} Ah "
          f"({(1 - first[-1][2] / first[0][2]) * 100:.1f}% fade)")
    print(f"  PE/NE ratio  {first[0][r_col]:.3f} -> {first[-1][r_col]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())