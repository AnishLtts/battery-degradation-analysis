"""
MODEL 2 - STEP 3: Make thousands of training curves with ALL 4 mechanisms mixed
===============================================================================
VERSION 2 - uses the B0005-calibrated cell and makes the data look more real.

For each training example:
  1. Make a slightly different fresh cell (real cells are never identical: +-2 %)
  2. Pick RANDOM amounts of LLI, LAM-PE, LAM-NE and resistance growth
  3. Damage the cell with all of them at once  (damage() from Step 2)
  4. Make the curve look like a real measurement: same point spacing as the
     NASA tester, plus 1 mV of measurement noise
  5. Turn the curve into NUMBERS (features)
  6. Work out the LABEL: what % of the capacity loss came from each mechanism

Output:  training_data.csv   (one row = one simulated aged cell)

How the % label is worked out (fair sharing):
  Mechanisms interact, so "loss with only LLI" + "loss with only LAM-PE" ...
  does not add up to the real total loss. We use the fair-share rule
  (Shapley value): try adding the 4 mechanisms in every possible order and
  average how much extra capacity each one removes. The 4 shares then add
  up EXACTLY to the total loss, so the percentages always sum to 100 %.

Needs step1_fresh_cell.py and step2_damage_one_at_a_time.py in the SAME folder.
"""

import itertools
import math
import numpy as np
import pandas as pd
from step1_fresh_cell import FRESH_CELL, discharge_curve, ica_curve, HERE
from step2_damage_one_at_a_time import damage

N_CELLS = 4000                                  # how many training examples
MECHANISMS = ["LLI", "LAM_PE", "LAM_NE", "RES"]
RANGES = {                                      # random amount limits
    "LLI":    (0.00, 0.40),                     # up to 40 % lithium lost
    "LAM_PE": (0.00, 0.40),                     # up to 40 % PE material lost
    "LAM_NE": (0.00, 0.40),                     # up to 40 % NE material lost
    "RES":    (0.00, 1.50),                     # resistance up to 2.5x fresh
}
# (B0005 at its last cycle needs roughly LLI 33 %, LAM-PE 40 %, LAM-NE 15 %,
#  resistance +36 % - see Step 5b - so these ranges cover its whole life.)
ARG = {"LLI": "lli", "LAM_PE": "lam_pe", "LAM_NE": "lam_ne", "RES": "r_growth"}

CELL_SPREAD = 0.02                              # fresh cells differ by about 2 %
NOISE_V = 0.001                                 # 1 mV measurement noise

ICA_GRID = np.linspace(3.30, 3.95, 30)          # voltages where we read dQ/dV
FEATURE_NAMES = (["capacity_Ah", "v_start", "v_mean"]
                 + [f"ica_{v:.3f}V" for v in ICA_GRID])


# ------------------------------------------------------------------ features
def make_features(Q, V):
    """Turn one discharge curve into a fixed list of numbers.
    The SAME function is used on the real NASA data in Step 6."""
    g, d = ica_curve(Q, V)
    ica_on_grid = np.interp(ICA_GRID, g, d, left=0.0, right=0.0)
    return [Q.max(), V[0], V.mean()] + list(ica_on_grid)


def like_real_measurement(Q, V, rng):
    """Re-sample the smooth simulated curve like the NASA tester does, add noise."""
    step = rng.uniform(0.005, 0.010)                  # NASA logs every ~0.005-0.01 Ah
    q = np.arange(0, Q.max(), step)
    v = np.interp(q, Q, V) + rng.normal(0, NOISE_V, len(q))
    return q, v


# ------------------------------------------------------------------ labels
def capacity_with(fresh, amounts, which):
    """Capacity of the cell when only the mechanisms in `which` are applied."""
    kwargs = {ARG[m]: amounts[m] for m in which}
    Q, _ = discharge_curve(damage(fresh, **kwargs))
    return Q.max()


def fair_shares(fresh, amounts):
    """Shapley value: each mechanism's fair share of the capacity loss (Ah)."""
    n = len(MECHANISMS)
    cap = {}                                              # cache: 16 combinations
    for r in range(n + 1):
        for combo in itertools.combinations(MECHANISMS, r):
            cap[frozenset(combo)] = capacity_with(fresh, amounts, combo)

    shares = {}
    for m in MECHANISMS:
        others = [x for x in MECHANISMS if x != m]
        total = 0.0
        for r in range(n):
            for combo in itertools.combinations(others, r):
                S = frozenset(combo)
                weight = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                total += weight * (cap[S] - cap[S | {m}])     # extra loss m causes
        shares[m] = total
    return shares, cap[frozenset()] - cap[frozenset(MECHANISMS)]


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    rows = []
    print(f"Making {N_CELLS} simulated aged cells ... (takes about 15-60 seconds)")

    while len(rows) < N_CELLS:
        # 1. a slightly different fresh cell each time
        fresh = dict(FRESH_CELL)
        for key in ["ne_size_Ah", "pe_size_Ah", "lithium_Ah", "resistance", "polarisation"]:
            fresh[key] *= 1 + rng.normal(0, CELL_SPREAD)

        # 2-3. random damage + fair shares
        amounts = {m: rng.uniform(*RANGES[m]) for m in MECHANISMS}
        shares, total_loss = fair_shares(fresh, amounts)
        if total_loss < 0.02:                 # skip almost-healthy cells (< ~1 %)
            continue

        # 4-5. realistic curve -> features
        Q, V = discharge_curve(damage(fresh, **{ARG[m]: amounts[m] for m in MECHANISMS}))
        if Q.max() < 1.0:                     # skip cells far more worn than B0005 ever gets (1.29 Ah)
            continue
        q, v = like_real_measurement(Q, V, rng)
        feats = make_features(q, v)

        # 6. labels. A share can come out slightly below 0 when one mechanism is
        # 'hidden' by another - that mechanism then gets 0 % of the blame.
        pct = [max(shares[m], 0.0) for m in MECHANISMS]
        pct = [100 * p / sum(pct) for p in pct]           # shares -> % (sum = 100)

        rows.append(feats
                    + [amounts[m] for m in MECHANISMS]    # the true damage amounts
                    + [total_loss]
                    + pct)                                # the LABELS
        if len(rows) % 500 == 0:
            print(f"  {len(rows)} done")

    columns = (FEATURE_NAMES
               + [f"amount_{m}" for m in MECHANISMS]
               + ["total_loss_Ah"]
               + [f"pct_{m}" for m in MECHANISMS])
    df = pd.DataFrame(rows, columns=columns)
    out = HERE / "training_data.csv"
    df.to_csv(out, index=False)

    print(f"\nSaved {len(df)} rows x {len(columns)} columns to: {out}")
    print("\nFirst 5 cells - how their capacity loss is shared (%):")
    print(df[[f"pct_{m}" for m in MECHANISMS]].head().round(1).to_string())
    print("\nAverage share over all cells (%):")
    print(df[[f"pct_{m}" for m in MECHANISMS]].mean().round(1).to_string())