"""
MODEL 2 - STEP 5a: Export the real NASA discharge curves to simple CSV files
============================================================================
Reads B0005.mat, B0006.mat, B0007.mat and B0018.mat (the same NASA files
Model 1 uses) and saves every discharge cycle as plain numbers:
    cycle, time_s, voltage_V, current_A, temp_C, capacity_Ah

Output: b0005_discharge_curves.csv, b0006_..., b0007_..., b0018_...  (next to this file)

Only thing to check: DATA_DIR below must point to the folder with B0005.mat
(the same folder as DATA_DIR in model1_diagnosis).
"""

import glob
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat

DATA_DIR = r"C:\Users\99017994\OneDrive - LTTS\Documents\1. BatteryAgingARC-FY08Q4"
CELLS = ["B0005", "B0006", "B0007", "B0018"]
HERE = Path(__file__).parent


def find_mat_file(cell):
    path = os.path.join(DATA_DIR, cell + ".mat")
    if os.path.isfile(path):
        return path
    hits = glob.glob(os.path.join(DATA_DIR, "**", cell + ".mat"), recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(f"{cell}.mat not found in {DATA_DIR} - fix DATA_DIR at the top")


def export(cell):
    mat = loadmat(find_mat_file(cell))
    key = cell if cell in mat else next(k for k in mat if not k.startswith("__"))
    cycles = mat[key][0, 0]["cycle"]
    cycles = cycles[0] if cycles.ndim == 2 else cycles

    parts, n = [], 0
    for c in cycles:
        if str(np.squeeze(c["type"])) != "discharge":
            continue
        n += 1                                         # discharge number (kept even if skipped)
        d = c["data"]
        d = d[0, 0] if d.ndim == 2 else d
        v = np.squeeze(d["Voltage_measured"]).astype(float)
        i = np.squeeze(d["Current_measured"]).astype(float)
        T = np.squeeze(d["Temperature_measured"]).astype(float)
        t = np.squeeze(d["Time"]).astype(float)
        if v.size < 20 or not (v.size == i.size == T.size == t.size):
            continue                                   # broken recording
        # capacity = integral of |current| over time, in Ah
        q = np.concatenate([[0.0], np.cumsum(np.abs(i[:-1]) * np.diff(t))]) / 3600.0
        parts.append(pd.DataFrame({"cycle": n, "time_s": t, "voltage_V": v,
                                   "current_A": i, "temp_C": T, "capacity_Ah": q}))

    df = pd.concat(parts, ignore_index=True)
    out = HERE / f"{cell.lower()}_discharge_curves.csv"
    df.to_csv(out, index=False)
    caps = df.groupby("cycle")["capacity_Ah"].max()
    print(f"{cell}: {len(parts)} discharge cycles, capacity {caps.iloc[0]:.3f} -> {caps.iloc[-1]:.3f} Ah "
          f"({(1 - caps.iloc[-1] / caps.iloc[0]) * 100:.1f} % fade)  ->  {out.name}")


if __name__ == "__main__":
    for cell in CELLS:
        try:
            export(cell)
        except FileNotFoundError as e:
            print(f"{cell}: skipped - {e}")