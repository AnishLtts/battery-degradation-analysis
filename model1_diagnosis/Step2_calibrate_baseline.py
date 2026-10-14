"""
STEP 2 - calibrate_baseline.py
==============================
Measures the HEALTHY (fresh) cell's ICA shape from the real NASA data,
so the synthetic generator (step 4) starts from the real battery and not
from invented peaks.

    1. Take the first N_FRESH_CYCLES discharge cycles of each cell
    2. Compute their ICA (same function the rest of the model uses)
    3. Average them into one fresh-cell curve
    4. Fit N_PEAKS Gaussian peaks + a straight-line background
    5. Save the result to baseline_fit.json (read by step 4) + a picture

RUN:   python step2_calibrate_baseline.py
"""

import os
import sys
import json
import glob
import importlib
import numpy as np
from scipy.io import loadmat
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ config
DATA_DIR = r"C:\Users\99017994\OneDrive - LTTS\Documents\1. BatteryAgingARC-FY08Q4"
CELLS = ["B0005", "B0006", "B0007", "B0018"]
OUT_DIR = r"C:\Users\99017994\battery_out"
OUT_JSON = "baseline_fit.json"

N_FRESH_CYCLES = 10      # first N discharge cycles count as "fresh"
N_PEAKS = 3              # Gaussians to fit
MIN_CURRENT_A = 0.1      # below this the battery is resting, not discharging

# ---- load step 5 (feature extractor) whatever its file is called ----------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _name in ("step5_feature_extractor", "Step5_feature_extractor", "feature_extractor"):
    try:
        _fx = importlib.import_module(_name)
        break
    except ModuleNotFoundError:
        continue
else:
    raise SystemExit("Cannot find the feature extractor (step 5) in this folder.")
compute_ica_dva, V_GRID, V_SPLIT = _fx.compute_ica_dva, _fx.V_GRID, _fx.V_SPLIT


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
    """Voltage and capacity of the REAL discharge only.
    Removes the rest at the start (current ~0) and the voltage bounce-back
    after the cut-off - both create fake ICA peaks."""
    d = c["data"]
    d = d[0, 0] if d.ndim == 2 else d
    v = np.squeeze(d["Voltage_measured"]).astype(float)
    i = np.squeeze(d["Current_measured"]).astype(float)
    t = np.squeeze(d["Time"]).astype(float)
    if v.size < 20 or v.size != t.size:
        return None
    q = np.concatenate([[0.0], np.cumsum(np.abs(i[:-1]) * np.diff(t))]) / 3600.0
    k = np.argmin(v)                              # end of discharge
    on = np.abs(i[:k + 1]) > MIN_CURRENT_A        # current really flowing
    if on.sum() < 20:
        return None
    return v[:k + 1][on], q[:k + 1][on]


def multi_gauss(v, *p):
    """N_PEAKS Gaussians plus a straight-line background."""
    out = p[3 * N_PEAKS] + p[3 * N_PEAKS + 1] * v
    for k in range(N_PEAKS):
        c, a, w = p[3 * k:3 * k + 3]
        out = out + a * np.exp(-((v - c) / w) ** 2)
    return out


def fresh_curve():
    """Average ICA of the first N_FRESH_CYCLES discharges of every cell."""
    stack = []
    for cell in CELLS:
        path = find_mat(cell)
        if path is None:
            print(f"  {cell}: not found, skipping")
            continue
        used = 0
        for c in load_cycles(path, cell):
            if str(np.squeeze(c["type"])) != "discharge":
                continue
            vq = discharge_vq(c)
            if vq is None:
                continue
            _, ica, _, valid = compute_ica_dva(*vq)
            if ica is None:
                continue
            stack.append(np.where(valid, ica, np.nan))
            used += 1
            if used == N_FRESH_CYCLES:
                break
        print(f"  {cell}: {used} fresh cycles used")
    return np.vstack(stack) if stack else None


def fit_peaks(v, y):
    """Start guesses from the biggest bumps, then least-squares fit."""
    pk, props = find_peaks(y, prominence=y.max() * 0.02, distance=10)
    pk = pk[np.argsort(props["prominences"])[::-1][:N_PEAKS]]
    # not enough bumps found -> spread the missing guesses evenly
    extra = np.linspace(0, len(y) - 1, N_PEAKS + 2)[1:-1].astype(int)
    pk = np.sort(np.concatenate([pk, extra[:N_PEAKS - len(pk)]]))

    p0, lo, hi = [], [], []
    for j in pk:
        p0 += [v[j], max(y[j], 1e-3), 0.10]
        lo += [v.min(), 0.0, 0.01]
        hi += [v.max(), y.max() * 3, 0.60]
    p0 += [float(y.min()), 0.0]
    lo += [-y.max(), -y.max()]
    hi += [2 * y.max(), 2 * y.max()]

    try:
        popt, _ = curve_fit(multi_gauss, v, y, p0=p0, bounds=(lo, hi), maxfev=40000)
    except RuntimeError as e:
        print(f"\nFit failed ({e}); using the start guesses instead.")
        popt = np.array(p0)
    return popt


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print("STEP 2 - Measuring the fresh-cell ICA shape from real cells")
    print("=" * 70)

    arr = fresh_curve()
    if arr is None:
        print("\nNo cycles found. Check DATA_DIR at the top of this file.")
        return 1

    n_ok = np.isfinite(arr).sum(axis=0)
    good = n_ok > 0                               # voltages covered by at least one cycle
    v = V_GRID[good]
    y = np.nansum(arr[:, good], axis=0) / n_ok[good]
    print(f"\nAveraged {arr.shape[0]} fresh cycles into one curve.")

    popt = fit_peaks(v, y)
    fit = multi_gauss(v, *popt)
    r2 = 1.0 - float(np.sum((y - fit) ** 2) / np.sum((y - y.mean()) ** 2))

    peaks = sorted(({"pos": float(popt[3 * k]),
                     "height": float(popt[3 * k + 1]),
                     "width": float(popt[3 * k + 2])} for k in range(N_PEAKS)),
                   key=lambda d: d["pos"])

    print(f"\nFit quality R^2 = {r2:.4f}\n")
    print("Fresh-cell peaks:")
    for k, p in enumerate(peaks, 1):
        side = "NE side" if p["pos"] < V_SPLIT else "PE side"
        print(f"  peak{k}: {p['pos']:.3f} V   height {p['height']:.4f}   "
              f"width {p['width']:.4f}   ({side})")

    # PE-side area / NE-side area of the measured curve
    lo_m, hi_m = v < V_SPLIT, v >= V_SPLIT
    a_lo = float(np.trapezoid(y[lo_m], v[lo_m])) if lo_m.sum() > 2 else 0.0
    a_hi = float(np.trapezoid(y[hi_m], v[hi_m])) if hi_m.sum() > 2 else 0.0
    ratio = a_hi / a_lo if a_lo > 1e-12 else float("nan")
    print(f"\nratio_high_low (PE area / NE area) = {ratio:.3f}")

    out = {"peaks": peaks,
           "background": [float(popt[-2]), float(popt[-1])],
           "v_range": [float(v.min()), float(v.max())],   # voltages the fit is valid for
           "ratio_high_low": ratio,
           "r2": r2,
           "n_cycles_averaged": int(arr.shape[0]),
           "cells": CELLS}
    json_path = os.path.join(OUT_DIR, OUT_JSON)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {json_path}")

    plt.figure(figsize=(9, 5))
    plt.plot(v, y, lw=2, color="#2a78d6", label="measured fresh cells")
    plt.plot(v, fit, lw=1.6, ls="--", color="#eb6834",
             label=f"fitted baseline (R²={r2:.3f})")
    for k, p in enumerate(peaks, 1):
        plt.axvline(p["pos"], color="#86857f", lw=0.8, ls=":")
        plt.text(p["pos"], y.max() * 0.95, f" p{k}", fontsize=8, color="#52514e")
    plt.xlabel("Voltage (V)")
    plt.ylabel("dQ/dV")
    plt.title("Real fresh-cell baseline vs fitted model")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    png = os.path.join(OUT_DIR, "baseline_fit.png")
    plt.savefig(png, dpi=130)
    print(f"Saved: {png}")
    print("\nNext: run step 6 (train_model) and step 8 (predict_degradation).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())