"""
STEP 8 - predict_degradation.py   (with CALIBRATED confidence)
=============================================================
Uses the trained model (step 6) in three parts:

  Part 1  CALIBRATE the confidence.
          The raw "confidence" of a Random Forest is the share of its 300 trees
          that vote for the answer. That number is often too high or too low.
          We make NEW synthetic cells (true answer known, never seen in training),
          try 3 ways of turning the votes into a confidence (raw votes, sigmoid,
          isotonic) and keep the most honest one - so that "80 % sure" really
          means "right about 80 % of the time".

  Part 2  TEST on other new synthetic cells (true mode known).
          Accuracy, and a reliability table: for each confidence level,
          is the model really right that often?  Before vs after calibration.

  Part 3  The REAL NASA cycles from real_features.csv (step 7).
          Nobody labelled these, so there is no accuracy - the output is the
          diagnosis + its calibrated confidence + a trust level for every cycle.

Output: real_diagnosis.csv               one row per real cycle
        degradation_model_calibrated.pkl  the model + its confidence corrector

RUN:   python step8_predict_degradation.py     (after steps 6 and 7)
"""

import os
import sys
import csv
import pickle
import importlib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load(*names):
    """Import a step module whatever its file is called."""
    for n in names:
        try:
            return importlib.import_module(n)
        except ModuleNotFoundError:
            continue
    raise SystemExit(f"Cannot find {names[-1]}.py in this folder.")


_train = _load("step6_train_model", "Step6_train_model", "train_model")   # same labelling + features
CLASSES, FEATURE_NAMES = _train.CLASSES, _train.FEATURE_NAMES

# ------------------------------------------------------------------ config
OUT_DIR = r"C:\Users\99017994\battery_out"
MODEL_FILE = "degradation_model.pkl"
REAL_CSV = "real_features.csv"
CALIB_SEED = 777           # new cells used ONLY to calibrate the confidence
TEST_SEED = 12345          # other new cells used ONLY to test (never seen before)
N_CALIB_CELLS = 20         # per mode
N_TEST_CELLS = 12          # per mode
METHODS = ["raw", "sigmoid", "isotonic"]   # raw = tree votes as they are
TRUST = [(0.80, "high"), (0.60, "medium"), (0.0, "low")]   # calibrated confidence -> trust level


# ---------------------------------------------------------------- helpers
def load_model():
    path = os.path.join(OUT_DIR, MODEL_FILE)
    if not os.path.isfile(path):
        raise SystemExit(f"Model not found: {path}\nRun step 6 (train_model) first.")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if bundle["feature_names"] != FEATURE_NAMES:
        raise SystemExit("Model was trained with different features - re-run step 6.")
    return bundle["classifier"], bundle.get("heldout_accuracy", float("nan"))


class RawVotes:
    """'No correction': use the trees' own vote share as the confidence."""
    def __init__(self, clf):
        self.clf, self.classes_ = clf, clf.classes_

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


def make_calibrator(clf, method):
    """Correct only the confidence of the already-trained model (the model itself is unchanged)."""
    if method == "raw":
        return RawVotes(clf)
    try:                                              # scikit-learn 1.6 and newer
        from sklearn.frozen import FrozenEstimator
        return CalibratedClassifierCV(FrozenEstimator(clf), method=method)
    except ImportError:                               # older scikit-learn
        return CalibratedClassifierCV(clf, method=method, cv="prefit")


def confidence_of(cal, X, pred):
    """Calibrated probability of the answer the model actually gave."""
    return cal.predict_proba(X)[np.arange(len(pred)), np.searchsorted(cal.classes_, pred)]


def trust_level(conf):
    return next(name for limit, name in TRUST if conf >= limit)


def calibration_error(conf, correct, bins=10):
    """Average gap between 'how sure' and 'how often right' (0 = perfectly honest)."""
    edges = np.linspace(0, 1, bins + 1)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            err += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return err


def reliability_table(conf, correct, title):
    print(f"\n  {title}")
    print(f"  {'model says it is':<18}{'curves':>8}{'avg confidence':>16}{'really right':>14}")
    for lo, hi in [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]:
        m = (conf >= lo) & (conf < hi)
        if m.any():
            print(f"  {f'{lo*100:.0f}-{min(hi, 1)*100:.0f} % sure':<18}{m.sum():>8}"
                  f"{conf[m].mean()*100:>15.0f}%{correct[m].mean()*100:>13.0f}%")


def most_common(a):
    u, n = np.unique(a, return_counts=True)
    return u[n.argmax()]


# ------------------------------------------------- part 1: calibrate
def calibrate(clf):
    """Try 3 ways of turning tree votes into a confidence and keep the most honest one.
    Half of the calibration cells fit each method, the other half judge it."""
    print("=" * 72)
    print("PART 1 - calibrating the confidence on new synthetic cells")
    print("=" * 72)
    curves = _train.generate_curves(n_cells_per_mode=N_CALIB_CELLS, seed=CALIB_SEED)
    X, y, cells, _, _ = _train.build_table(curves)
    uniq = np.unique(cells)
    judge = np.isin(cells, uniq[::2])                 # every 2nd cell judges (split by cell)
    print(f"{len(y)} curves the model never saw  ({(~judge).sum()} to fit, {judge.sum()} to judge)\n")
    scores = {}
    for method in METHODS:
        cal = make_calibrator(clf, method)
        if method != "raw":
            cal.fit(X[~judge], y[~judge])
        pred = clf.predict(X[judge])
        conf = confidence_of(cal, X[judge], pred)
        scores[method] = calibration_error(conf, (pred == y[judge]).astype(float))
        print(f"  {method:<9} calibration error {scores[method]*100:4.1f} points")
    best = min(scores, key=scores.get)
    print(f"\nChosen: {best}  (most honest)")
    cal = make_calibrator(clf, best)
    if best != "raw":
        cal.fit(X, y)                                  # refit the winner on all calibration cells
    return cal, best


# ------------------------------------------------- part 2: honest test
def test_on_unseen_synthetic(clf, cal, method):
    print("\n" + "=" * 72)
    print("PART 2 - testing on other unseen synthetic cells (true mode known)")
    print("=" * 72)
    curves = _train.generate_curves(n_cells_per_mode=N_TEST_CELLS, seed=TEST_SEED)
    X, y, _, sev, _ = _train.build_table(curves)          # same labels as training
    pred = clf.predict(X)
    conf_raw = clf.predict_proba(X).max(axis=1)
    conf_cal = confidence_of(cal, X, pred)
    correct = (pred == y).astype(float)

    print(f"\ncurves tested: {len(y)}")
    print(f"ACCURACY     : {correct.mean()*100:.1f}%  ({int(correct.sum())}/{len(y)})")
    print("\nBy severity:")
    for lo, hi in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.35), (0.35, 1.01)]:
        m = (sev >= lo) & (sev < hi)
        if m.any():
            print(f"  severity {lo:.2f}-{hi:.2f}: {correct[m].mean()*100:5.1f}%"
                  f"   calibrated confidence {conf_cal[m].mean()*100:4.1f}%   ({m.sum()} curves)")

    print("\nIs the confidence honest?  ('avg confidence' should match 'really right')")
    reliability_table(conf_raw, correct, "BEFORE calibration (raw tree votes):")
    reliability_table(conf_cal, correct, f"AFTER calibration ({method}):")
    print(f"\nCalibration error  before: {calibration_error(conf_raw, correct)*100:.1f} points"
          f"   after: {calibration_error(conf_cal, correct)*100:.1f} points   (0 = perfectly honest)")
    print("\nTrust levels (calibrated confidence: high >= 80 %, medium >= 60 %, low below):")
    for _, name in TRUST:
        m = np.array([trust_level(c) == name for c in conf_cal])
        if m.any():
            print(f"  {name:<7} {m.sum():4d} curves   really right {correct[m].mean()*100:5.1f}%")


# ------------------------------------------------- part 3: the real cells
def run_on_real(clf, cal):
    print("\n" + "=" * 72)
    print("PART 3 - real NASA cycles (no answer key)")
    print("=" * 72)
    path = os.path.join(OUT_DIR, REAL_CSV)
    if not os.path.isfile(path):
        print(f"\n{REAL_CSV} not found. Run step 7 (build_real_features) first.")
        return

    with open(path, newline="", encoding="utf-8") as f:
        table = list(csv.DictReader(f))
    missing = [k for k in FEATURE_NAMES if k not in table[0]]
    if missing:
        raise SystemExit(f"{REAL_CSV} is from an older step 5 (missing {missing[:3]}...). "
                         "Re-run step 7.")

    X = np.nan_to_num(np.array([[float(r[k]) for k in FEATURE_NAMES] for r in table]),
                      nan=0.0, posinf=0.0, neginf=0.0)
    cells = np.array([r["cell"] for r in table])
    cycles = np.array([int(r["cycle"]) for r in table])
    caps = np.array([float(r["measured_capacity_Ah"]) for r in table])
    pred = clf.predict(X)
    conf = confidence_of(cal, X, pred)
    trust = np.array([trust_level(c) for c in conf])

    print(f"\nreal cycles diagnosed: {len(pred)}")
    print("\nAll cells together:")
    for c in CLASSES:
        m = pred == c
        if m.any():
            levels = ", ".join(f"{(trust[m] == n).sum()} {n}" for _, n in TRUST)
            print(f"  {c:<12} {m.sum():4d} cycles ({m.mean()*100:5.1f}%)"
                  f"   confidence {conf[m].mean()*100:3.0f}%   trust: {levels}")

    print("\nPer cell, first quarter of life vs last quarter (average confidence):")
    for cell in sorted(set(cells)):
        m = cells == cell
        order = np.argsort(cycles[m])
        p, cp, cf = pred[m][order], caps[m][order], conf[m][order]
        q = max(len(p) // 4, 1)
        print(f"  {cell}: {len(p):3d} cycles, {(1 - cp[-1]/cp[0])*100:4.1f}% fade   "
              f"early -> {most_common(p[:q]):<11}({cf[:q].mean()*100:3.0f}%)   "
              f"late -> {most_common(p[-q:]):<11}({cf[-q:].mean()*100:3.0f}%)")

    out = os.path.join(OUT_DIR, "real_diagnosis.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cell", "cycle", "measured_capacity_Ah", "diagnosis", "confidence_pct", "trust"])
        w.writerows(zip(cells, cycles, caps, pred, np.round(conf * 100, 1), trust))
    print(f"\nSaved: {out}")
    print("Note: the confidence is calibrated on synthetic cells. It covers the model's own")
    print("doubt, not the differences between the synthetic and the real battery.")


def main():
    clf, acc = load_model()
    print(f"Model loaded (held-out accuracy at training: {acc*100:.1f}%)\n")
    cal, method = calibrate(clf)
    test_on_unseen_synthetic(clf, cal, method)
    run_on_real(clf, cal)
    path = os.path.join(OUT_DIR, "degradation_model_calibrated.pkl")
    with open(path, "wb") as f:
        pickle.dump({"classifier": clf, "calibrated": cal, "feature_names": FEATURE_NAMES,
                     "classes": CLASSES, "calibration_method": method,
                     "trust_levels": TRUST}, f)
    print(f"Saved: {path}  (model + confidence corrector)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())