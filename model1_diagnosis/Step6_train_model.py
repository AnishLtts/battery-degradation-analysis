"""
STEP 6 - train_model.py
=======================
Trains the degradation-mode classifier on the synthetic cells (step 4).

    synthetic V-Q curves (step 4)
        -> extract_features() (step 5, the SAME function the real data uses)
        -> 26 features
        -> Random Forest
        -> degradation_model.pkl  (used by step 8)

Two rules that keep the score honest:
  * Split by CELL, not by curve. Each virtual cell gives ~13 curves; if they
    land on both sides of the split the model has already "seen" the test cell.
  * A curve with almost no damage is labelled Healthy, whatever mode its cell
    will develop later - at cycle 0 an "LLI cell" looks exactly like a healthy
    one, and teaching the model otherwise is just label noise.

RUN:   python step6_train_model.py      (after steps 2-5)
"""

import os
import sys
import pickle
import importlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load(*names):
    """Import a step module whatever its file is called."""
    for n in names:
        try:
            return importlib.import_module(n)
        except ModuleNotFoundError:
            continue
    raise SystemExit(f"Cannot find {names[-1]}.py in this folder.")


_fx = _load("step5_feature_extractor", "Step5_feature_extractor", "feature_extractor")
_gen = _load("step4_synthetic_generator", "Step4_synthetic_generator",
             "Step4_synthetic_degradation_generator", "synthetic_degradation_generator")
extract_features, features_to_row, FEATURE_NAMES = (_fx.extract_features, _fx.features_to_row,
                                                    _fx.FEATURE_NAMES)
generate_curves, CLASSES = _gen.generate_curves, _gen.CLASSES

# ------------------------------------------------------------------ config
OUT_DIR = r"C:\Users\99017994\battery_out"
MODEL_FILE = "degradation_model.pkl"
N_CELLS_PER_MODE = 40
TEST_FRACTION = 0.25
HEALTHY_BELOW = 0.05      # severity under 5 % -> the curve is still Healthy (step 8 uses this too)
SEED = 0


def build_table(curves):
    """Every curve -> one feature row (via step 5) + its label."""
    X, y, cells, sev = [], [], [], []
    for c in curves:
        f = extract_features(c["voltage"], c["capacity"])
        if not np.isfinite(f["capacity_Ah"]):          # curve unusable
            continue
        X.append(np.nan_to_num(features_to_row(f), nan=0.0, posinf=0.0, neginf=0.0))
        y.append("Healthy" if c["severity"] < HEALTHY_BELOW else c["label"])
        cells.append(c["cell_id"])
        sev.append(c["severity"])
    return np.array(X), np.array(y), np.array(cells), np.array(sev), len(curves) - len(X)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print("STEP 6 - Training the degradation classifier")
    print("=" * 70)

    curves = generate_curves(n_cells_per_mode=N_CELLS_PER_MODE, seed=SEED)
    print(f"\n{len(curves)} synthetic curves ({N_CELLS_PER_MODE} cells x {len(CLASSES)} modes)")
    X, y, cells, sev, bad = build_table(curves)
    print(f"{X.shape[0]} usable rows x {X.shape[1]} features"
          + (f"  ({bad} unusable curves dropped)" if bad else ""))

    # ---- split by CELL ----------------------------------------------------
    uniq = np.unique(cells)
    np.random.default_rng(SEED).shuffle(uniq)
    test_cells = set(uniq[:int(len(uniq) * TEST_FRACTION)])
    is_test = np.isin(cells, list(test_cells))
    Xtr, ytr, Xte, yte = X[~is_test], y[~is_test], X[is_test], y[is_test]
    print(f"\nSplit by cell: train {len(ytr)} curves / {len(uniq) - len(test_cells)} cells, "
          f"test {len(yte)} curves / {len(test_cells)} cells (never seen in training)")

    # ---- train ------------------------------------------------------------
    clf = RandomForestClassifier(n_estimators=300, random_state=SEED,
                                 class_weight="balanced_subsample", n_jobs=-1)
    clf.fit(Xtr, ytr)                         # trees need no feature scaling

    # ---- evaluate on the unseen cells -------------------------------------
    pred = clf.predict(Xte)
    acc = float((pred == yte).mean())
    print("\n" + "=" * 70)
    print(f"HELD-OUT ACCURACY: {acc*100:.1f}%  ({(pred == yte).sum()}/{len(yte)})  - synthetic test cells")
    print("=" * 70 + "\n")
    print(classification_report(yte, pred, labels=CLASSES, zero_division=0))

    print("Confusion matrix (rows = truth, cols = predicted)")
    cm = confusion_matrix(yte, pred, labels=CLASSES)
    w = max(map(len, CLASSES)) + 1
    print(" " * w + "".join(f"{c[:8]:>9}" for c in CLASSES))
    for name, row in zip(CLASSES, cm):
        print(f"{name:<{w}}" + "".join(f"{v:>9d}" for v in row))

    print("\nAccuracy by severity (low damage is the hard part):")
    sev_te = sev[is_test]
    for lo, hi in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.35), (0.35, 1.01)]:
        m = (sev_te >= lo) & (sev_te < hi)
        if m.any():
            print(f"  severity {lo:.2f}-{hi:.2f}: {(pred[m] == yte[m]).mean()*100:5.1f}%  ({m.sum()} curves)")

    print("\nTop 8 features the model relies on:")
    for n, v in sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1])[:8]:
        print(f"  {v*100:5.1f}%  {n}")

    path = os.path.join(OUT_DIR, MODEL_FILE)
    with open(path, "wb") as f:
        pickle.dump({"classifier": clf, "feature_names": FEATURE_NAMES,
                     "classes": CLASSES, "heldout_accuracy": acc}, f)
    print(f"\nSaved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())