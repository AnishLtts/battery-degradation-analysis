"""
MODEL 2 - STEP 5b: Calibrate the simulated cell to the REAL B0005 cell
=====================================================================
Input : b0005_discharge_curves.csv  (made by Step 5a)
Output: step5b_calibration.png

Part A - fresh cell check
  Compares real B0005 cycle 1 with the simulated fresh cell from Step 1.
  With REFIT = True it searches again for the 7 numbers that make them match
  (ne_size, pe_size, lithium, resistance, polarisation, 2 blur widths)
  and prints them - these are the numbers written into step1_fresh_cell.py.

Part B - aged cell check
  Can our 4 mechanisms reproduce the LAST real cycle (168)?
  We search for the LLI / LAM-PE / LAM-NE / resistance amounts that fit it.
  If the fit error is small (tens of mV), the simulation is realistic enough
  to train the model on.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, differential_evolution
import step1_fresh_cell as s1
from step2_damage_one_at_a_time import damage

REFIT = False          # True = search the fresh-cell numbers again (only needed if Step 1 changes)
LAST_CYCLE = 168

data = pd.read_csv(s1.HERE / "b0005_discharge_curves.csv")


def real_curve(cycle):
    c = data[(data.cycle == cycle) & (data.current_A < -1.0)]   # only points under load
    Q = c.capacity_Ah.values
    return Q - Q[0], c.voltage_V.values


def voltage_diff(Q_sim, V_sim, Q_real, V_real):
    """Simulated minus real voltage at 60 equally spaced capacities (V)."""
    q = np.linspace(0, Q_real.max(), 60)
    return np.interp(q, Q_sim, V_sim, right=2.0) - np.interp(q, Q_real, V_real)


def error_mV(Q_sim, V_sim, Q_real, V_real):
    return 1000 * np.sqrt(np.mean(voltage_diff(Q_sim, V_sim, Q_real, V_real) ** 2))


def set_blur(bn, bp):
    s1.ne_voltage = s1.blurred(s1.graphite_raw, bn)
    s1.pe_voltage = s1.blurred(s1.lco_raw, bp)


def refit_fresh_cell(Q_real, V_real):
    """Search the 7 fresh-cell numbers that best match real cycle 1."""
    names = ["ne_size_Ah", "pe_size_Ah", "lithium_Ah", "resistance", "polarisation"]

    def residual(p):
        set_blur(p[5], p[6])
        Q, V = s1.discharge_curve(dict(s1.FRESH_CELL, **dict(zip(names, p[:5]))))
        return voltage_diff(Q, V, Q_real, V_real)

    r = least_squares(residual, [2.0, 4.2, 3.7, 0.12, 0.013, 0.05, 0.05], diff_step=1e-3,
                      bounds=([1.5, 1.5, 1.5, 0, 0, 0.005, 0.005], [4, 6, 6, 0.4, 0.5, 0.15, 0.15]))
    print("Fitted numbers (put these into step1_fresh_cell.py):")
    for n, v in zip(names + ["NE_BLUR", "PE_BLUR"], r.x):
        print(f"   {n:<14}= {v:.4f}")
    set_blur(r.x[5], r.x[6])                                  # use them for the rest of this script
    s1.FRESH_CELL.update(zip(names, r.x[:5]))


if __name__ == "__main__":
    # ---------------- Part A
    Qr, Vr = real_curve(1)
    if REFIT:
        refit_fresh_cell(Qr, Vr)
    Qs, Vs = s1.discharge_curve(s1.FRESH_CELL)
    print("Part A - fresh cell vs real cycle 1")
    print(f"   capacity  real {Qr.max():.3f} Ah   simulated {Qs.max():.3f} Ah")
    print(f"   voltage error: {error_mV(Qs, Vs, Qr, Vr):.1f} mV")

    # ---------------- Part B
    QL, VL = real_curve(LAST_CYCLE)
    print(f"\nPart B - searching the damage that reproduces real cycle {LAST_CYCLE} ... (a few seconds)")

    def aged(a):
        return s1.discharge_curve(damage(s1.FRESH_CELL, lli=a[0], lam_pe=a[1], lam_ne=a[2], r_growth=a[3]))

    def cost(a):
        Q, V = aged(a)
        return 9999.0 if len(Q) < 10 else error_mV(Q, V, QL, VL)   # 9999 = broken cell

    best = differential_evolution(cost, [(0, .45), (0, .45), (0, .45), (0, 2)],
                                  seed=0, maxiter=80, popsize=15, tol=1e-7, polish=False)
    lli, lam_pe, lam_ne, rg = best.x
    Qa, Va = aged(best.x)
    print(f"   best fit: LLI {lli*100:.0f} %, LAM-PE {lam_pe*100:.0f} %, LAM-NE {lam_ne*100:.0f} %, "
          f"resistance +{rg*100:.0f} %")
    print(f"   capacity  real {QL.max():.3f} Ah   simulated {Qa.max():.3f} Ah")
    print(f"   voltage error: {best.fun:.1f} mV")

    # ---------------- picture
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for (Q, V, st, lab) in [(Qr, Vr, "k-", "real cycle 1"), (Qs, Vs, "k--", "simulated fresh"),
                            (QL, VL, "r-", f"real cycle {LAST_CYCLE}"), (Qa, Va, "r--", "simulated aged")]:
        ax[0].plot(Q, V, st, label=lab)
        g, d = s1.ica_curve(Q, V)
        ax[1].plot(g, d, st, label=lab)
    ax[0].set(xlabel="Capacity (Ah)", ylabel="Voltage (V)", title="Real (solid) vs simulated (dashed)")
    ax[1].set(xlabel="Voltage (V)", ylabel="dQ/dV (Ah/V)", title="ICA: real vs simulated")
    for a in ax:
        a.legend(fontsize=8); a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(s1.HERE / "step5b_calibration.png", dpi=110)
    print("\nSaved picture:", s1.HERE / "step5b_calibration.png")