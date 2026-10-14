"""
MODEL 2 - STEP 6: Use the trained model on the REAL NASA cells
==============================================================
Input : b0005/b0006/b0007/b0018_discharge_curves.csv  (made by Step 5a)
        contribution_model.pkl                         (made by Step 4)
Output: per cell   b000X_contributions.csv  and  step6_b000X_contributions.png
        all cells  step6_all_cells.png  (side-by-side comparison)

The model was calibrated on B0005 only. B0006, B0007 and B0018 are a test:
they were NEVER used to build the simulation or train the model.

For every real discharge cycle of every cell:
  1. keep only the points where the cell is really discharging (current about -2 A),
     and stop at 2.7 V. (NASA discharged B0005 to 2.7 V, B0006 to 2.5 V, B0007 to 2.2 V
     and B0018 to 2.5 V. Cutting all of them at 2.7 V - the same limit as the
     simulation - makes the curves comparable.)
  2. turn the curve into the SAME 33 features as the training data (make_features)
  3. QUANTIFY: ask the model what % of the capacity loss is LLI / LAM-PE / LAM-NE / resistance
  4. CLASSIFY:  name the dominant mechanism of that cycle
       - "Too early"  : less than 2 % capacity lost - too little damage to judge
       - "Mixed (A+B)": the top two are within 10 points of each other
       - otherwise    : the mechanism with the biggest share
     label_confidence = "high" when the winner's range does not overlap the runner-up's
  5. CONFIDENCE: the model is 300 decision trees, each giving its own answer.
     Where the trees disagree, the model is less sure. We turn that spread into a
     "+-" range and CALIBRATE it on simulated cells the model never saw, so that
     the true answer falls inside the range 80 % of the time:
       - 400 unseen cells: find how much to stretch/shrink the tree spread
       - other 400 unseen cells: check that the range really holds 80 % of the time
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from step1_fresh_cell import HERE
from step3_make_training_data import make_features
from step4_train_model import to_percent, LABELS

CELLS = ["B0005", "B0006", "B0007", "B0018"]
V_CUTOFF = 2.7          # common end-of-discharge voltage for every cell
MIN_LOSS_PCT = 2.0      # below this, the cell is too healthy to say what is damaging it
MIXED_GAP = 10.0        # if the top two shares are closer than this, call it "Mixed"
TARGET = 0.80           # the +- range should contain the true answer 80 % of the time
SMOOTH = 10             # cycles averaged together (single cycles are noisy)
COLOURS = ["tab:red", "tab:blue", "tab:green", "tab:orange"]


def classify(shares, loss_pct, names):
    """Turn the 4 percentages of one cycle into a label."""
    if loss_pct < MIN_LOSS_PCT:
        return "Too early"
    first, second = np.argsort(shares)[::-1][:2]         # biggest two
    if shares[first] - shares[second] < MIXED_GAP:
        a, b = sorted([first, second])                   # same order every time
        return f"Mixed ({names[a]}+{names[b]})"
    return names[first]


def predict_with_range(model, X):
    """Mean answer + 10th / 90th percentile over all trees of the Random Forest."""
    per_tree = np.stack([to_percent(tree.predict(X)) for tree in model.estimators_])  # trees x cells x 4
    return (to_percent(model.predict(X)),
            np.percentile(per_tree, 10, axis=0),
            np.percentile(per_tree, 90, axis=0))


def cut_at(Q, V, vmin=V_CUTOFF):
    """Keep the curve only down to vmin (with the exact crossing point added)."""
    below = np.where(V < vmin)[0]
    if len(below) == 0 or below[0] == 0:
        return Q, V
    k = below[0]
    q_end = Q[k - 1] + (Q[k] - Q[k - 1]) * (V[k - 1] - vmin) / (V[k - 1] - V[k])
    return np.r_[Q[:k], q_end], np.r_[V[:k], vmin]


def calibrate_range(model, feature_cols):
    """Use unseen SIMULATED cells (true answer known) to make the +- range honest."""
    df = pd.read_csv(HERE / "training_data.csv")
    _, X_test, _, y_test = train_test_split(df[feature_cols].values, df[LABELS].values,
                                            test_size=0.20, random_state=0)   # same split as Step 4
    mid, lo, hi = predict_with_range(model, X_test)
    spread = np.maximum((hi - lo) / 2, 0.5)                    # tree spread, at least 0.5 point
    ratio = np.abs(y_test - mid) / spread                      # how many "spreads" off was it?
    half = len(y_test) // 2
    stretch = np.quantile(ratio[:half], TARGET, axis=0)        # calibrate on the first half ...
    inside = np.abs(y_test[half:] - mid[half:]) <= stretch * spread[half:]
    coverage = inside.mean(axis=0) * 100                       # ... check on the second half
    width = (stretch * spread[half:]).mean(axis=0)
    return stretch, coverage, width


def rolling(x):
    return pd.Series(np.asarray(x)).rolling(SMOOTH, center=True, min_periods=1).mean().round(1).values


def analyse(cell, model, mechanisms, features, stretch):
    """Quantify + classify every cycle of one real cell."""
    data = pd.read_csv(HERE / f"{cell.lower()}_discharge_curves.csv")
    rows = []
    for cycle, cyc in data.groupby("cycle"):
        on_load = cyc[cyc["current_A"] < -1.0]              # skip rest points
        if len(on_load) < 30:
            continue
        Q = on_load["capacity_Ah"].values
        Q, V = cut_at(Q - Q[0], on_load["voltage_V"].values)
        rows.append([cycle, Q.max()] + make_features(Q, V))

    table = pd.DataFrame(rows, columns=["cycle", "measured_capacity_Ah"] + features)
    pct, low, high = predict_with_range(model, table[features].values)
    pm = stretch * np.maximum((high - low) / 2, 0.5)           # calibrated +- per cycle and mechanism

    out = pd.DataFrame({"cycle": table["cycle"],
                        "capacity_Ah": table["measured_capacity_Ah"].round(4),
                        "capacity_loss_pct": (100 * (1 - table["measured_capacity_Ah"]
                                                     / table["measured_capacity_Ah"].iloc[0])).round(2)})
    for i, m in enumerate(mechanisms):
        out[f"{m}_pct"] = pct[:, i].round(1)
        out[f"{m}_low"] = np.clip(pct[:, i] - pm[:, i], 0, 100).round(1)    # bottom of the range
        out[f"{m}_high"] = np.clip(pct[:, i] + pm[:, i], 0, 100).round(1)   # top of the range
    for i, m in enumerate(mechanisms):                                       # 10-cycle averages
        out[f"{m}_pct_smooth"] = rolling(out[f"{m}_pct"])
        out[f"{m}_low_smooth"] = rolling(out[f"{m}_low"])
        out[f"{m}_high_smooth"] = rolling(out[f"{m}_high"])
        out[f"{m}_pm"] = rolling(pm[:, i])                                   # the +- value
    # capacity jumps up a little after rest periods (it "recovers"), so judge "too early" on
    # the smoothed loss, and once a cell has passed 2 % it stays past it (running maximum)
    out["capacity_loss_pct_smooth"] = (out.capacity_loss_pct.rolling(SMOOTH, center=True, min_periods=1)
                                       .mean().cummax().round(2))
    smooth = out[[f"{m}_pct_smooth" for m in mechanisms]].values
    lo_s = out[[f"{m}_low_smooth" for m in mechanisms]].values
    hi_s = out[[f"{m}_high_smooth" for m in mechanisms]].values
    labels, conf = [], []
    for k in range(len(out)):
        lab = classify(smooth[k], out.capacity_loss_pct_smooth.iloc[k], mechanisms)
        labels.append(lab)
        if lab == "Too early" or lab.startswith("Mixed"):
            conf.append("-")
        else:   # "high" = the winner's range does not overlap the runner-up's range
            first, second = np.argsort(smooth[k])[::-1][:2]
            conf.append("high" if lo_s[k, first] > hi_s[k, second] else "low")
    out["dominant_mechanism"] = labels
    out["label_confidence"] = conf
    out.to_csv(HERE / f"{cell.lower()}_contributions.csv", index=False)

    # ---- print a short report
    print(f"\n{'#' * 70}\n# {cell}" + ("   (calibration cell)" if cell == "B0005" else "   (TEST cell - never seen before)")
          + f"\n{'#' * 70}")
    print(f"Real {cell} cycles analysed: {len(out)}")
    print(f"Capacity {out.capacity_Ah.iloc[0]:.3f} -> {out.capacity_Ah.iloc[-1]:.3f} Ah "
          f"({out.capacity_loss_pct.iloc[-1]:.1f} % lost)\n")
    judged = out[out.capacity_loss_pct_smooth >= MIN_LOSS_PCT]   # same rule as the classification
    if len(judged):
        print("Share of the capacity loss (10-cycle average, +- = 80 % confidence range):")
        print(f"{'cycle':>6}{'loss %':>8}" + "".join(f"{m:>14}" for m in mechanisms) + "   dominant")
        for k in sorted(set(range(0, len(judged), max(1, len(judged) // 8))) | {len(judged) - 1}):
            r = judged.iloc[k]
            print(f"{int(r.cycle):>6}{r.capacity_loss_pct:>8.1f}"
                  + "".join(f"{r[m + '_pct_smooth']:>7.1f}% +-{r[m + '_pm']:>4.1f}" for m in mechanisms)
                  + f"   {r.dominant_mechanism}")
    else:
        print("Capacity loss never reached 2 % - too little damage to split.")

    # ---- classification summary: which mechanism dominates which part of the life
    print("\nClassification - dominant mechanism over the life of the cell:")
    block = (out.dominant_mechanism != out.dominant_mechanism.shift()).cumsum()
    for _, part in out.groupby(block):
        label, start, end = part.dominant_mechanism.iloc[0], part.cycle.iloc[0], part.cycle.iloc[-1]
        note = ("" if label == "Too early" or label.startswith("Mixed")
                else f"   (confidence high in {(part.label_confidence == 'high').mean() * 100:.0f} % of these cycles)")
        print(f"   cycles {start:>3} - {end:>3}  ({end - start + 1:>3} cycles) : {label:<22}{note}")
    print("\n   Cycles per class: " + ", ".join(f"{n}: {c}" for n, c in out.dominant_mechanism.value_counts().items()))

    # ---- picture
    first_judged = judged.cycle.iloc[0] if len(judged) else out.cycle.iloc[-1]
    fig, ax = plt.subplots(4, 1, figsize=(10, 11), sharex=True,
                           gridspec_kw={"height_ratios": [2.5, 3, 3, 0.8]})
    ax[0].plot(out.cycle, out.capacity_Ah, "k")
    ax[0].set(ylabel="Capacity (Ah)", title=f"Real NASA {cell}: capacity and what caused the loss")
    ax[0].grid(alpha=0.3)
    ax[1].stackplot(out.cycle, *[out[f"{m}_pct_smooth"] for m in mechanisms],
                    labels=mechanisms, colors=COLOURS, alpha=0.8)
    ax[1].set(ylabel="Share of capacity loss (%)", ylim=(0, 100))
    ax[1].legend(loc="upper right")
    ax[1].text(out.cycle.iloc[0] + 1, 50, "too little\nloss to split", fontsize=8)
    for m, c in zip(mechanisms, COLOURS):                  # each mechanism with its confidence band
        ax[2].plot(out.cycle, out[f"{m}_pct_smooth"], color=c, label=m)
        ax[2].fill_between(out.cycle, out[f"{m}_low_smooth"], out[f"{m}_high_smooth"], color=c, alpha=0.2)
    ax[2].set(ylabel="Share (%)", ylim=(0, 100), title="Each mechanism with its 80 % confidence band")
    ax[2].grid(alpha=0.3); ax[2].legend(loc="upper right", fontsize=8)
    for a in ax[1:3]:
        a.axvspan(out.cycle.iloc[0], first_judged, color="white", alpha=0.7)
    class_colour = dict(zip(mechanisms, COLOURS))           # one coloured block per cycle
    for cyc, lab in zip(out.cycle, out.dominant_mechanism):
        ax[3].axvspan(cyc - 0.5, cyc + 0.5, color=class_colour.get(lab, "lightgrey" if lab == "Too early" else "plum"))
    ax[3].set(yticks=[], xlabel="Cycle", title="Classification: dominant mechanism "
              "(grey = too early, purple = mixed)")
    plt.tight_layout()
    plt.savefig(HERE / f"step6_{cell.lower()}_contributions.png", dpi=110)
    plt.close(fig)
    print(f"\nSaved: {cell.lower()}_contributions.csv  and  step6_{cell.lower()}_contributions.png")
    return out


if __name__ == "__main__":
    saved = joblib.load(HERE / "contribution_model.pkl")
    model, mechanisms, features = saved["model"], saved["mechanisms"], saved["features"]
    stretch, coverage, width = calibrate_range(model, features)

    results = {}
    for cell in CELLS:
        if not (HERE / f"{cell.lower()}_discharge_curves.csv").exists():
            print(f"\n{cell}: no CSV found - run Step 5a first. Skipped.")
            continue
        results[cell] = analyse(cell, model, mechanisms, features, stretch)

    # ---- how trustworthy is the +- range?
    print("\nConfidence check on 400 unseen SIMULATED cells (true answer known):")
    for i, m in enumerate(mechanisms):
        print(f"   {m:<7}: true value inside the +- range in {coverage[i]:4.0f} % of cells "
              f"(target 80 %, average range +-{width[i]:.1f} points)")
    print("   Note: on REAL data the range covers the model's uncertainty only, not\n"
          "   differences between the simulation and the real cell (see Step 5b).")

    # ---- summary of all cells
    print(f"\n{'=' * 70}\nSUMMARY - all cells, last cycle (10-cycle average)\n{'=' * 70}")
    print(f"{'cell':<7}{'cycles':>7}{'loss %':>8}" + "".join(f"{m:>14}" for m in mechanisms) + "   dominant")
    for cell, out in results.items():
        r = out.iloc[-1]
        print(f"{cell:<7}{len(out):>7}{r.capacity_loss_pct:>8.1f}"
              + "".join(f"{r[m + '_pct_smooth']:>7.1f}% +-{r[m + '_pm']:>4.1f}" for m in mechanisms)
              + f"   {r.dominant_mechanism}")

    # ---- comparison picture: share of each mechanism vs capacity lost, all cells
    if results:
        styles = {"B0005": "-", "B0006": "--", "B0007": "-.", "B0018": ":"}
        fig, ax = plt.subplots(1, len(mechanisms), figsize=(16, 4.2), sharey=True)
        for j, (m, c) in enumerate(zip(mechanisms, COLOURS)):
            for cell, out in results.items():
                ok = out.capacity_loss_pct_smooth >= MIN_LOSS_PCT
                ax[j].plot(out.capacity_loss_pct_smooth[ok], out[f"{m}_pct_smooth"][ok],
                           styles.get(cell, "-"), color=c, label=cell)
            ax[j].set(title=m, xlabel="Capacity lost (%)", ylim=(0, 100))
            ax[j].grid(alpha=0.3)
            ax[j].legend(fontsize=8)
        ax[0].set_ylabel("Share of capacity loss (%)")
        plt.suptitle("All NASA cells: does each mechanism follow the same path as the cell wears out?")
        plt.tight_layout()
        plt.savefig(HERE / "step6_all_cells.png", dpi=110)
        print("\nSaved: step6_all_cells.png")