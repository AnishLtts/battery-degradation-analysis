"""
STEP 3 - degradation_equations.py
=================================
Takes the fresh-cell ICA curve measured in step 2 and bends it four ways,
one per degradation mode. The function ica_curve() below holds the rules;
the synthetic generator (step 4) imports it to make labelled training data,
so the picture this file draws is exactly what the model is trained on.

    LLI         less lithium      -> ALL peaks shrink together and slide left
    LAM-PE      PE material lost  -> only the PE-side peaks shrink
    LAM-NE      NE material lost  -> only the NE-side peaks shrink
    RESISTANCE  higher I*R        -> curve slides left and smears out,
                                     area (capacity) kept

Uniform vs selective shrinking is what lets the classifier tell LLI, LAM-PE
and LAM-NE apart; "area kept" is what separates resistance from the rest.

Note: in a real full cell every ICA peak has contributions from both
electrodes. Calling a peak "PE" or "NE" (by V_SPLIT) is the usual working
simplification for this kind of screening.

RUN:   python step3_degradation_equations.py     (after step 2)
"""

import os
import sys
import json
import importlib
import numpy as np
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ config
OUT_DIR = r"C:\Users\99017994\battery_out"
BASELINE_JSON = os.path.join(OUT_DIR, "baseline_fit.json")   # written by step 2

LLI_SHIFT = 0.06      # V the peaks slide left at 100 % LLI
R_SHIFT = 0.10        # V the curve slides left at 100 % resistance
R_BROADEN = 0.09      # V of extra smearing at 100 % resistance

# ---- voltage grid + PE/NE split come from step 5, so all steps agree -------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _name in ("step5_feature_extractor", "Step5_feature_extractor", "feature_extractor"):
    try:
        _fx = importlib.import_module(_name)
        break
    except ModuleNotFoundError:
        continue
else:
    raise SystemExit("Cannot find the feature extractor (step 5) in this folder.")
V, V_SPLIT = _fx.V_GRID, _fx.V_SPLIT
DV = float(V[1] - V[0])


# ---------------------------------------------------------------- baseline
def load_baseline():
    """Fresh-cell peaks + background measured by step 2, each peak tagged PE or NE."""
    if not os.path.isfile(BASELINE_JSON):
        raise SystemExit(f"{BASELINE_JSON} not found - run step 2 (calibrate_baseline) first.")
    with open(BASELINE_JSON, encoding="utf-8") as f:
        fit = json.load(f)
    peaks = sorted(fit["peaks"], key=lambda p: p["pos"])
    for p in peaks:
        p["electrode"] = "PE" if p["pos"] >= V_SPLIT else "NE"
    return peaks, fit["background"] + fit.get("v_range", [float(V.min()), float(V.max())])


BASELINE_PEAKS, BASELINE_BG = load_baseline()     # BG = [offset, slope, v_lo, v_hi]


def capacity_of(ica, v=None):
    """Area under dQ/dV = capacity."""
    v = V if v is None else v
    return float(abs(np.trapezoid(ica, v)))


def ica_curve(v, peaks, background=(0.0, 0.0, -np.inf, np.inf),
              lli=0.0, lam_pe=0.0, lam_ne=0.0, resistance=0.0):
    """
    THE degradation rules - used by this demo AND by the generator (step 4).
    Any combination of the four modes (0 = none, 1 = total) on any voltage grid.
    Physical order: lithium and material are lost first, then resistance
    distorts whatever is left.
    """
    lli, lam_pe, lam_ne, resistance = (float(np.clip(x, 0, 1))
                                       for x in (lli, lam_pe, lam_ne, resistance))
    shift = LLI_SHIFT * lli + R_SHIFT * resistance      # everything slides left
    vs = v + shift                                      # evaluate the curve "shifted"

    # straight-line background, only trusted inside the fitted voltage range;
    # outside it fades out (0.05 V) instead of being extrapolated
    b0, b1, lo, hi = background
    vc = np.clip(vs, lo, hi)
    fade = np.exp(-np.abs(vs - vc) / 0.05)
    ica = (b0 + b1 * vc) * fade * (1 - lli)
    for p in peaks:
        side = "PE" if p["pos"] >= V_SPLIT else "NE"
        lam = lam_pe if side == "PE" else lam_ne
        ica = ica + p["height"] * (1 - lli) * (1 - lam) * np.exp(-((vs - p["pos"]) / p["width"]) ** 2)
    ica = np.maximum(ica, 0.0)                          # dQ/dV can never be negative

    if resistance > 0:                                  # smear, keep the area
        dv = abs(float(v[1] - v[0]))
        before = capacity_of(ica, v)
        ica = gaussian_filter1d(ica, sigma=R_BROADEN * resistance / dv, mode="nearest")
        after = capacity_of(ica, v)
        if after > 0:
            ica *= before / after
    return ica


def apply_mix(**modes):
    """Measured fresh cell + chosen modes, on the step-5 grid (peaks only)."""
    return ica_curve(V, BASELINE_PEAKS, **modes)


BASELINE = apply_mix()
BASELINE_CAP = capacity_of(BASELINE)


def apply_lli(s):
    return apply_mix(lli=s)


def apply_lam(s, electrode):
    return apply_mix(lam_pe=s) if electrode == "PE" else apply_mix(lam_ne=s)


def apply_resistance(s):
    return apply_mix(resistance=s)


# -------------------------------------------------------------------- demo
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pe = ", ".join(f"{p['pos']:.2f}" for p in BASELINE_PEAKS if p["electrode"] == "PE")
    ne = ", ".join(f"{p['pos']:.2f}" for p in BASELINE_PEAKS if p["electrode"] == "NE")

    mechanisms = [
        ("LLI", apply_lli, "all peaks shrink together + slide left"),
        ("LAM-PE", lambda s: apply_lam(s, "PE"), f"only PE peaks ({pe} V) shrink"),
        ("LAM-NE", lambda s: apply_lam(s, "NE"), f"only NE peaks ({ne} V) shrink"),
        ("Resistance", apply_resistance, "peaks slide + smear, area kept"),
    ]
    levels = [0.0, 0.15, 0.30, 0.45]
    cols = plt.cm.plasma(np.linspace(0.05, 0.8, len(levels)))

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle("Step 3 - The four degradation modes applied to the measured fresh cell",
                 fontsize=14, fontweight="bold")
    print("=" * 68)
    print(f"Baseline capacity (area under dQ/dV): {BASELINE_CAP:.4f}")
    print("=" * 68)

    def plot_panel(ax, title, curves, colours):
        for c, (label, ica) in zip(colours, curves):
            ax.plot(V, ica, color=c, lw=1.7, label=label)
        ax.set(title=title, xlabel="Voltage (V)", ylabel="dQ/dV")
        ax.title.set_fontsize(10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # panels 1-4: one mode at a time
    for ax, (name, fn, blurb) in zip(axes.flat, mechanisms):
        print(f"\n{name}  -  {blurb}")
        curves = []
        for s in levels:
            ica = fn(s)
            cap = capacity_of(ica)
            print(f"   severity {s*100:4.0f}%   capacity {cap:.4f}   ({(cap/BASELINE_CAP-1)*100:+.1f}%)")
            curves.append((f"severity {s*100:.0f}%", ica))
        plot_panel(ax, f"{name}\n{blurb}", curves, cols)

    # panel 5: realistic mixes
    mixes = [("fresh", {}),
             ("LLI-dominant", dict(lli=.30, lam_pe=.05, lam_ne=.05, resistance=.15)),
             ("PE-dominant", dict(lli=.10, lam_pe=.35, lam_ne=.05, resistance=.10)),
             ("heavily aged", dict(lli=.35, lam_pe=.25, lam_ne=.20, resistance=.40))]
    print("\nMixed cases (what a real aged cell looks like)")
    curves = []
    for label, kw in mixes:
        ica = apply_mix(**kw)
        cap = capacity_of(ica)
        print(f"   {label:<14} capacity {cap:.4f}  ({(cap/BASELINE_CAP-1)*100:+.1f}%)")
        curves.append((label, ica))
    plot_panel(axes[1, 1], "Mixed modes\nreal cells are always a blend",
               curves, plt.cm.viridis(np.linspace(0, .85, len(mixes))))

    # panel 6: capacity vs severity
    ax = axes[1, 2]
    sev = np.linspace(0, 0.6, 25)
    for name, fn, _ in mechanisms:
        ax.plot(sev * 100, [capacity_of(fn(s)) / BASELINE_CAP * 100 for s in sev],
                lw=1.8, label=name)
    ax.set(title="Capacity vs severity\nresistance is the flat one",
           xlabel="Severity (%)", ylabel="Capacity retained (%)")
    ax.title.set_fontsize(10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    png = os.path.join(OUT_DIR, "degradation_mechanisms.png")
    plt.savefig(png, dpi=130)
    print(f"\nSaved plot: {png}")

    header, rows = ["voltage_V", "baseline"], [V, BASELINE]
    for name, fn, _ in mechanisms:
        for s in levels[1:]:
            header.append(f"{name}_{int(s*100)}pct")
            rows.append(fn(s))
    csv = os.path.join(OUT_DIR, "degradation_curves.csv")
    np.savetxt(csv, np.column_stack(rows), delimiter=",",
               header=",".join(header), comments="", fmt="%.6f")
    print(f"Saved data: {csv}")


if __name__ == "__main__":
    main()