"""
STEP 4 - synthetic_degradation_generator.py
===========================================
Makes the labelled training data. Real NASA cells have no labels, so we
build many virtual cells whose degradation mode we KNOW:

    1. start from the fresh-cell peaks measured in step 2
       (every virtual cell gets slightly different peaks - real cells differ)
    2. age it with ONE mode (LLI, LAM-PE, LAM-NE, Resistance) or keep it Healthy,
       using the rules in step 3 (ica_curve)
    3. turn the dQ/dV curve into a voltage-capacity curve, the same two arrays
       a real NASA discharge gives: ~275 points evenly spaced in capacity
       (constant current) plus ~0.3 mV voltage-sensor noise

Each curve carries a cell_id, so training (step 6) can keep all curves of
one virtual cell on the same side of the train/test split.

RUN:   python step4_synthetic_generator.py      (after steps 2 and 3)
"""

import os
import sys
import importlib
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load(*names):
    """Import a step module whatever its file is called."""
    for n in names:
        try:
            return importlib.import_module(n)
        except ModuleNotFoundError:
            continue
    raise SystemExit(f"Cannot find {names[-1]}.py in this folder.")


_eq = _load("step3_degradation_equations", "Step3_degradation_equations", "degradation_equations")

CLASSES = ["Healthy", "LLI", "LAM-PE", "LAM-NE", "Resistance"]
PEAKS, BG = _eq.BASELINE_PEAKS, _eq.BASELINE_BG       # measured in step 2
ELECTRODE = [p["electrode"] for p in PEAKS]

V_START, V_CUTOFF = 4.2, 2.7                    # NASA B0005 discharge window
N_SAMPLES = 275                                 # NASA B0005 logs 170-355 points (median 275),
                                                # evenly in time = evenly in capacity
NOISE_V = 0.0003                                # voltage-sensor noise (V), measured on B0005
MAX_SEVERITY = 0.5                              # damage reached at the last cycle
MODE_TO_ARG = {"LLI": "lli", "LAM-PE": "lam_pe", "LAM-NE": "lam_ne",
               "Resistance": "resistance"}


def sample_cell(rng):
    """One virtual cell's own fresh shape - real cells are not identical."""
    return [{"pos": p["pos"] + rng.normal(0, 0.015),
             "height": p["height"] * (1 + rng.normal(0, 0.06)),
             "width": max(p["width"] * (1 + rng.normal(0, 0.06)), 0.02)}
            for p in PEAKS]


def generate_degradation_vq_curve(
    cycle_num: int = 0,
    degradation_mode: str = "Healthy",
    severity: float = 0.0,
    noise_level: float = NOISE_V,
    cell_params: List[Dict] = None,
    rng=None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    One synthetic discharge: returns (voltage, capacity, metadata), the same
    arrays read_battery (step 1) gives for a real NASA cycle.
    """
    rng = np.random.default_rng() if rng is None else rng
    cell_params = sample_cell(rng) if cell_params is None else cell_params
    s = float(np.clip(severity, 0.0, 1.0))

    v_fine = np.linspace(V_START, V_CUTOFF, 3000)              # discharge: high -> low
    modes = {MODE_TO_ARG[degradation_mode]: s} if degradation_mode in MODE_TO_ARG else {}
    dq_dv = _eq.ica_curve(v_fine, cell_params, BG, **modes)

    # capacity = running area under dQ/dV, starting from 0 at the top voltage
    q_fine = np.concatenate([[0.0], np.cumsum(
        0.5 * (dq_dv[1:] + dq_dv[:-1]) * np.abs(np.diff(v_fine)))])

    # sample it the way the NASA tester does: evenly in capacity, + sensor noise
    capacity = np.linspace(0.0, q_fine[-1], N_SAMPLES)
    voltage = np.interp(capacity, q_fine, v_fine) + rng.normal(0, noise_level, N_SAMPLES)

    meta = {"cycle": cycle_num, "degradation_mode": degradation_mode, "severity": s}
    return voltage, capacity, meta


def generate_curves(n_cells_per_mode: int = 40, cycles: List[int] = None, seed: int = 0):
    """All training curves: n_cells_per_mode virtual cells per class, each aged over cycles."""
    cycles = cycles or [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]
    rng = np.random.default_rng(seed)
    last = max(cycles) or 1
    out = []
    for mode in CLASSES:
        for cell in range(n_cells_per_mode):
            params = sample_cell(rng)
            for cyc in cycles:
                if mode == "Healthy":
                    sev = 0.0
                else:
                    sev = float(np.clip(MAX_SEVERITY * cyc / last + rng.normal(0, 0.03), 0, 1))
                v, q, _ = generate_degradation_vq_curve(
                    cycle_num=cyc, degradation_mode=mode, severity=sev,
                    cell_params=params, rng=rng)
                out.append({"cell_id": f"{mode}_{cell:03d}", "label": mode, "cycle": cyc,
                            "severity": sev, "voltage": v, "capacity": q})
    return out


if __name__ == "__main__":
    print(f"peaks in use ({len(PEAKS)}), from baseline_fit.json:")
    for i, (p, e) in enumerate(zip(PEAKS, ELECTRODE), 1):
        print(f"  peak{i}: {p['pos']:.3f} V  height {p['height']:.4f}  "
              f"width {p['width']:.4f}  -> {e}")

    curves = generate_curves(n_cells_per_mode=3, cycles=[0, 70, 140])
    print(f"\n{len(curves)} test curves made. Capacity at the last cycle:")
    for c in curves:
        if c["cycle"] == 140 and c["cell_id"].endswith("000"):
            print(f"  {c['label']:<11} severity {c['severity']:.2f}   Q max {c['capacity'][-1]:.3f} Ah")