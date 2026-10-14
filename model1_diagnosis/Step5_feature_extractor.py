"""
STEP 5 - feature_extractor.py
=============================
The bridge between real and synthetic data. BOTH call extract_features():

    real NASA cycle  (step 7) --\
                                 >-- extract_features(V, Q) --> same features, same order
    synthetic curve  (step 4) --/

    ICA = dQ/dV vs voltage   -> peaks, NE/PE areas and their ratio
    DVA = dV/dQ vs capacity  -> central peak position (Ah) and height

FEATURE_NAMES is a fixed list - same length, same order for every curve -
which is what the classifier (step 6) needs.

USAGE
    from step5_feature_extractor import extract_features, features_to_row
    row = features_to_row(extract_features(voltage, capacity))
"""

import numpy as np
from scipy.signal import savgol_filter, find_peaks, peak_widths

# ---------------------------------------------------------------- settings
# Shared by every step. Do not override per caller.
V_MIN, V_MAX, V_POINTS = 3.2, 4.1, 300      # useful discharge window, 3 mV steps
V_GRID = np.linspace(V_MIN, V_MAX, V_POINTS)
DV = float(V_GRID[1] - V_GRID[0])
V_SPLIT = 3.6                                # below = NE-side peaks, above = PE-side

SMOOTH_WINDOW = 21                           # Savitzky-Golay window (63 mV on V_GRID)
SMOOTH_ORDER = 3
N_PEAK_SLOTS = 4                             # fixed peak slots, zero-padded

Q_POINTS = 200                               # DVA capacity grid
DVA_SOC = (0.2, 0.8)                         # look for the central DVA peak here


def _names():
    n = ["capacity_Ah", "ica_max", "ica_max_V", "ica_centroid_V",
         "area_low", "area_high", "ratio_high_low",
         "dva_peak", "dva_peak_Q", "n_peaks_found"]
    for i in range(1, N_PEAK_SLOTS + 1):
        n += [f"peak{i}_V", f"peak{i}_height", f"peak{i}_fwhm", f"peak{i}_area"]
    return n


FEATURE_NAMES = _names()
N_FEATURES = len(FEATURE_NAMES)


# ------------------------------------------------------------------ helpers
def _clean(voltage, capacity):
    """
    Keep only the real discharge, with voltage strictly falling.
    Drops the voltage bounce-back after cut-off and repeated/noisy points -
    sorting all points by voltage instead creates fake ICA peaks.
    Returns (v, q) with v INCREASING (what np.interp needs), or (None, None).
    """
    v = np.asarray(voltage, dtype=float).ravel()
    q = np.asarray(capacity, dtype=float).ravel()
    ok = np.isfinite(v) & np.isfinite(q)
    v, q = v[ok], q[ok]
    if v.size < 10:
        return None, None
    k = int(np.argmin(v))                          # end of discharge
    v, q = v[:k + 1], q[:k + 1]
    vm = np.minimum.accumulate(v)
    keep = np.r_[True, np.diff(vm) < 0]            # strictly falling
    v, q = vm[keep][::-1], q[keep][::-1]
    return (v, q) if v.size >= 10 else (None, None)


def _smooth(y):
    win = min(SMOOTH_WINDOW, (y.size // 2) * 2 - 1)
    return savgol_filter(y, win, SMOOTH_ORDER) if win > SMOOTH_ORDER + 1 else y


def _ica(v, q):
    """dQ/dV on V_GRID. Returns (ica, valid) - valid = grid inside the data."""
    valid = (V_GRID >= v[0]) & (V_GRID <= v[-1])
    if valid.sum() < 20:
        return None, None
    q_grid = np.interp(V_GRID, v, q)
    ica = _smooth(-np.gradient(q_grid, V_GRID))    # Q falls as V rises -> minus sign
    return np.maximum(ica, 0.0), valid


def _dva(v, q):
    """dV/dQ on a capacity grid (the true DVA). Returns (q_grid, dva)."""
    qg = np.linspace(q.min(), q.max(), Q_POINTS)
    order = np.argsort(q)
    q_s, idx = np.unique(q[order], return_index=True)
    v_on_q = np.interp(qg, q_s, v[order][idx])
    return qg, _smooth(np.abs(np.gradient(v_on_q, qg)))


def compute_ica_dva(voltage, capacity):
    """
    ICA and DVA of a raw curve, both on V_GRID.
    Returns (V_GRID, ica, dva, valid) or (None, None, None, None).
    (dva here = dV/dQ at each voltage, i.e. 1/ICA - handy for plots.
     The DVA FEATURES use the proper capacity-grid version in _dva.)
    """
    v, q = _clean(voltage, capacity)
    if v is None:
        return None, None, None, None
    ica, valid = _ica(v, q)
    if ica is None:
        return None, None, None, None
    dva = 1.0 / np.maximum(ica, max(ica.max() * 1e-3, 1e-9))
    return V_GRID, ica, dva, valid


# -------------------------------------------------------------------- main
def extract_features(voltage, capacity):
    """
    Diagnostic features of one raw voltage-capacity curve.
    Always returns a dict with exactly FEATURE_NAMES as keys.
    Unusable curve -> NaN everywhere (peak slots 0). Never raises.
    """
    # empty peak slots are 0, not NaN: "no 4th peak" is information
    out = {k: (0.0 if k.startswith("peak") else float("nan")) for k in FEATURE_NAMES}

    v, q = _clean(voltage, capacity)
    if v is None:
        return out
    ica, valid = _ica(v, q)
    if ica is None:
        return out
    y = np.where(valid, ica, 0.0)

    # ---- whole-curve shape ------------------------------------------------
    out["capacity_Ah"] = float(q.max() - q.min())
    j = int(np.argmax(y))
    out["ica_max"], out["ica_max_V"] = float(y[j]), float(V_GRID[j])
    out["ica_centroid_V"] = float((V_GRID * y).sum() / y.sum()) if y.sum() > 0 else float("nan")

    # ---- NE vs PE side: LAM-NE eats the low side, LAM-PE the high side ----
    low, high = (V_GRID < V_SPLIT) & valid, (V_GRID >= V_SPLIT) & valid
    a_low = float(np.trapezoid(y[low], V_GRID[low])) if low.sum() > 2 else 0.0
    a_high = float(np.trapezoid(y[high], V_GRID[high])) if high.sum() > 2 else 0.0
    out["area_low"], out["area_high"] = a_low, a_high
    out["ratio_high_low"] = a_high / a_low if a_low > 1e-12 else float("nan")

    # ---- DVA: central peak (graphite stage change) in the middle of the curve
    qg, dva = _dva(v, q)
    frac = (qg - qg[0]) / max(qg[-1] - qg[0], 1e-12)
    mid = (frac >= DVA_SOC[0]) & (frac <= DVA_SOC[1])
    if mid.any():
        k = int(np.argmax(np.where(mid, dva, -np.inf)))
        out["dva_peak"] = float(dva[k])
        out["dva_peak_Q"] = float(qg[k] - qg[0])

    # ---- ICA peaks, slot 1 = lowest voltage --------------------------------
    peaks, props = (find_peaks(y, prominence=y.max() * 0.03, distance=int(0.04 / DV))
                    if y.max() > 0 else (np.array([], dtype=int), {}))
    out["n_peaks_found"] = float(len(peaks))
    if len(peaks):
        keep = np.sort(peaks[np.argsort(props["prominences"])[::-1][:N_PEAK_SLOTS]])
        widths, _, lo, hi = peak_widths(y, keep, rel_height=0.5)
        for slot, (pk, wid, l, h) in enumerate(zip(keep, widths, lo, hi), start=1):
            li, hi_i = max(int(np.floor(l)), 0), min(int(np.ceil(h)) + 1, y.size)
            out[f"peak{slot}_V"] = float(V_GRID[pk])
            out[f"peak{slot}_height"] = float(y[pk])
            out[f"peak{slot}_fwhm"] = float(wid * DV)
            out[f"peak{slot}_area"] = (float(np.trapezoid(y[li:hi_i], V_GRID[li:hi_i]))
                                       if hi_i - li > 2 else 0.0)
    return out


def features_to_row(feats):
    """dict -> fixed-order list, ready for sklearn."""
    return [feats[k] for k in FEATURE_NAMES]


if __name__ == "__main__":
    print(f"step 5 - {N_FEATURES} features on {V_MIN}-{V_MAX} V:")
    for i, n in enumerate(FEATURE_NAMES, 1):
        print(f"  {i:2d}. {n}")