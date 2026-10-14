"""
MODEL 2 - STEP 1: Build a fresh battery cell from its two electrodes
=====================================================================
A battery = positive electrode (PE) + negative electrode (NE).
Cell voltage while discharging =
        PE voltage  -  NE voltage  -  current x resistance  -  polarisation

  * Electrode curves are published curves: PE = lithium cobalt oxide (LCO),
    NE = graphite (MCMB) - the "Dualfoil" curves (Doyle & Newman, also in PyBaMM).
  * Both curves are slightly blurred, because in a real cell not every
    particle is identical - this is what makes real ICA peaks broad.
  * The cell numbers and the two blur widths were FITTED (step5b_calibrate.py)
    so this simulated fresh cell matches real B0005 cycle 1.

Used by the later steps:  FRESH_CELL, discharge_curve(), ica_curve(),
                          pe_voltage(), ne_voltage(), V_MAX, V_MIN
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter

HERE = Path(__file__).parent   # pictures are saved next to this file


# ---------------------------------------------------------------
# 1. Electrode voltage curves  (fill = 0 empty ... 1 full of lithium)
# ---------------------------------------------------------------
def graphite_raw(s):
    """Graphite NE voltage vs Li (Doyle/Dualfoil MCMB curve)."""
    return (0.194 + 1.5 * np.exp(-120.0 * s)
            + 0.0351 * np.tanh((s - 0.286) / 0.083) - 0.0045 * np.tanh((s - 0.849) / 0.119)
            - 0.035 * np.tanh((s - 0.9233) / 0.05) - 0.0147 * np.tanh((s - 0.5) / 0.034)
            - 0.102 * np.tanh((s - 0.194) / 0.142) - 0.022 * np.tanh((s - 0.9) / 0.0164)
            - 0.011 * np.tanh((s - 0.124) / 0.0226) + 0.0155 * np.tanh((s - 0.105) / 0.029))


def lco_raw(s):
    """LCO PE voltage vs Li (Doyle/Dualfoil LiCoO2 curve)."""
    s = 1.062 * s
    return (2.16216 + 0.07645 * np.tanh(30.834 - 54.4806 * s)
            + 2.1581 * np.tanh(52.294 - 50.294 * s) - 0.14169 * np.tanh(11.0923 - 19.8543 * s)
            + 0.2051 * np.tanh(1.4684 - 5.4888 * s) + 0.2531 * np.tanh((-s + 0.56478) / 0.1316)
            - 0.02167 * np.tanh((s - 0.525) / 0.006))


def blurred(curve, width):
    """Blur an electrode curve by `width` (real particles are not all identical)."""
    s = np.linspace(-0.3, 1.3, 3201)
    step = s[1] - s[0]
    u = curve(np.clip(s, 1e-4, 1 - 1e-4))
    kernel = np.exp(-0.5 * (np.arange(-400, 401) * step / width) ** 2)
    kernel /= kernel.sum()
    u_blur = np.convolve(np.pad(u, 400, mode="edge"), kernel, mode="valid")
    return lambda fill: np.interp(fill, s, u_blur)


NE_BLUR, PE_BLUR = 0.047, 0.056          # fitted to B0005
ne_voltage = blurred(graphite_raw, NE_BLUR)
pe_voltage = blurred(lco_raw, PE_BLUR)


# ---------------------------------------------------------------
# 2. The fresh cell (numbers fitted to real B0005 cycle 1)
# ---------------------------------------------------------------
FRESH_CELL = {
    "ne_size_Ah":   2.015,   # how much lithium the NE can hold
    "pe_size_Ah":   4.249,   # how much lithium the PE can hold
    "lithium_Ah":   3.732,   # total movable lithium in the cell
    "resistance":   0.118,   # ohm  (instant voltage drop = current x this)
    "polarisation": 0.0128,  # V per Ah (extra drop that builds up during discharge)
    "current_A":    2.0,     # discharge current (NASA B0005 uses 2 A)
}
V_MAX, V_MIN = 4.2, 2.7      # charge limit and discharge cut-off


def discharge_curve(cell):
    """Return (capacity, voltage) for one full discharge of the cell."""
    ne_fill = np.linspace(0.9999, 0.0001, 5000)                                         # NE empties...
    pe_fill = (cell["lithium_Ah"] - ne_fill * cell["ne_size_Ah"]) / cell["pe_size_Ah"]  # ...PE fills
    ok = (pe_fill > 0) & (pe_fill < 1)
    ne_fill, pe_fill = ne_fill[ok], pe_fill[ok]
    if len(ne_fill) < 2:                                           # cell too damaged to work
        return np.array([0.0]), np.array([V_MIN])

    v_rest = pe_voltage(pe_fill) - ne_voltage(ne_fill)             # voltage with no current
    if not (v_rest <= V_MAX).any():                                # never drops below 4.2 V
        return np.array([0.0]), np.array([V_MIN])
    start = np.argmax(v_rest <= V_MAX)                             # fully charged point
    capacity = (ne_fill[start] - ne_fill[start:]) * cell["ne_size_Ah"]
    v_load = (v_rest[start:]
              - cell["current_A"] * cell["resistance"]             # instant drop
              - cell["polarisation"] * capacity)                   # slow build-up
    below = v_load < V_MIN
    end = (np.argmax(below) if below.any() else len(v_load) - 1) + 1
    return capacity[:end], v_load[:end]


def ica_curve(capacity, voltage):
    """dQ/dV: peaks show where each electrode has a plateau.
    Works on simulated AND on noisy real data."""
    v = np.minimum.accumulate(voltage)                 # remove small upward noise wiggles
    keep = np.r_[True, np.diff(v) < 0]                 # drop repeated voltages (interp needs them unique)
    v, q_raw = v[keep], np.asarray(capacity)[keep]
    grid = np.linspace(max(3.2, v.min() + 0.01), min(4.05, v.max() - 0.01), 200)
    q = np.interp(grid, v[::-1], q_raw[::-1])
    dqdv = -np.gradient(q, grid)
    return grid, savgol_filter(dqdv, 21, 3)            # smooth


# ---------------------------------------------------------------
# 3. Run it
# ---------------------------------------------------------------
if __name__ == "__main__":
    Q, V = discharge_curve(FRESH_CELL)
    print(f"Fresh cell capacity: {Q.max():.3f} Ah   (real B0005 cycle 1: 1.851 Ah)")
    print(f"Voltage range: {V.max():.2f} V -> {V.min():.2f} V")

    g, d = ica_curve(Q, V)
    peaks = [i for i in range(1, len(d) - 1) if d[i] > d[i-1] and d[i] > d[i+1] and d[i] > 0.3 * d.max()]
    print("ICA peaks at:", ", ".join(f"{g[i]:.3f} V" for i in peaks), "  (real B0005 main peak: 3.484 V)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(Q, V); ax[0].set(xlabel="Capacity (Ah)", ylabel="Voltage (V)", title="Fresh cell discharge")
    ax[1].plot(g, d); ax[1].set(xlabel="Voltage (V)", ylabel="dQ/dV (Ah/V)", title="ICA of fresh cell")
    for a in ax: a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(HERE / "step1_fresh_cell.png", dpi=120)
    print("Saved picture:", HERE / "step1_fresh_cell.png")