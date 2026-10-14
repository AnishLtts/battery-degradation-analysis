"""
MODEL 2 - STEP 2: Damage the fresh cell in 4 ways, one at a time
=================================================================
Each degradation mechanism = one number of the cell changing:

    LLI         lithium_Ah   goes DOWN   (lithium gets trapped, e.g. in SEI)
    LAM-PE      pe_size_Ah   goes DOWN   (positive electrode material lost)
    LAM-NE      ne_size_Ah   goes DOWN   (negative electrode material lost)
    Resistance  resistance   goes UP     (bigger voltage drop under current)

We apply each one ALONE and look at how the curves change.
Every mechanism leaves a different "fingerprint" on the ICA curve -
that is what the ML model will learn to recognise later.

Needs step1_fresh_cell.py in the SAME folder.
"""

import matplotlib.pyplot as plt
from step1_fresh_cell import FRESH_CELL, discharge_curve, ica_curve, HERE


def damage(cell, lli=0.0, lam_pe=0.0, lam_ne=0.0, r_growth=0.0):
    """
    Return a damaged copy of the cell.
    lli, lam_pe, lam_ne : fraction lost      (0.10 = 10 % lost)
    r_growth            : fraction increase  (1.00 = resistance doubled)
    """
    aged = dict(cell)                                   # copy, never change the fresh cell
    aged["lithium_Ah"] = cell["lithium_Ah"] * (1 - lli)
    aged["pe_size_Ah"] = cell["pe_size_Ah"] * (1 - lam_pe)
    aged["ne_size_Ah"] = cell["ne_size_Ah"] * (1 - lam_ne)
    aged["resistance"] = cell["resistance"] * (1 + r_growth)
    aged["polarisation"] = cell["polarisation"] * (1 + r_growth)   # grows with resistance
    return aged


# The 4 test cases: 10 % loss for the material mechanisms, resistance doubled
CASES = [
    ("LLI 10 %",          dict(lli=0.10),      "tab:red"),
    ("LAM-PE 10 %",       dict(lam_pe=0.10),   "tab:blue"),
    ("LAM-NE 10 %",       dict(lam_ne=0.10),   "tab:green"),
    ("Resistance x2",     dict(r_growth=1.0),  "tab:orange"),
]


if __name__ == "__main__":
    Q0, V0 = discharge_curve(FRESH_CELL)
    g0, d0 = ica_curve(Q0, V0)
    fresh_cap = Q0.max()
    print(f"Fresh cell capacity: {fresh_cap:.3f} Ah\n")
    print(f"{'Mechanism':<16}{'Capacity (Ah)':>14}{'Loss (%)':>10}")
    print("-" * 40)

    fig, ax = plt.subplots(2, 4, figsize=(16, 7))

    for col, (name, amount, colour) in enumerate(CASES):
        aged = damage(FRESH_CELL, **amount)
        Q, V = discharge_curve(aged)
        g, d = ica_curve(Q, V)
        loss = (1 - Q.max() / fresh_cap) * 100
        print(f"{name:<16}{Q.max():>14.3f}{loss:>10.1f}")

        # top row: voltage vs capacity
        ax[0, col].plot(Q0, V0, color="grey", label="fresh")
        ax[0, col].plot(Q, V, color=colour, label=name)
        ax[0, col].set(title=f"{name}  (-{loss:.1f} % capacity)",
                       xlabel="Capacity (Ah)", ylabel="Voltage (V)")
        # bottom row: ICA
        ax[1, col].plot(g0, d0, color="grey", label="fresh")
        ax[1, col].plot(g, d, color=colour, label=name)
        ax[1, col].set(xlabel="Voltage (V)", ylabel="dQ/dV (Ah/V)")
        for a in ax[:, col]:
            a.legend(fontsize=8); a.grid(alpha=0.3)

    plt.suptitle("Step 2: each mechanism changes the curves in its own way", fontsize=13)
    plt.tight_layout()
    out = HERE / "step2_damage_one_at_a_time.png"
    plt.savefig(out, dpi=110)
    print("\nSaved picture:", out)