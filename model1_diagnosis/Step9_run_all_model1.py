"""
run_all.py - runs Model 1 from start to finish, in order.
Put it in the model1_diagnosis folder and press Run.
Finds each step by its number (Step4_..., step4_...) or by its old name.
Stops at the first step that fails.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [   # (step number, old file name, what it does)
    (1, "read_battery", "Plot the real NASA data (V-Q, ICA, capacity fade)"),
    (2, "calibrate_baseline", "Measure the fresh-cell ICA peaks -> baseline_fit.json"),
    (3, "degradation_equations", "Show how each mode bends the curve"),
    (4, "synthetic_degradation_generator", "Check the synthetic generator"),    (6, "train_model", "Train the classifier -> degradation_model.pkl"),
    (7, "build_real_features", "Features of every real cycle -> real_features.csv"),
    (8, "predict_degradation", "Test on unseen cells + diagnose the real cells"),
]
# step 5 (feature_extractor) has nothing to run on its own - steps 2-8 use it.


def find(num, old):
    hits = sorted(HERE.glob(f"[Ss]tep{num}_*.py")) or sorted(HERE.glob(f"{old}.py"))
    return hits[0] if hits else None


for n, (num, old, what) in enumerate(STEPS, 1):
    script = find(num, old)
    if script is None:
        sys.exit(f"Cannot find step {num} ({old}.py) in {HERE}")
    print("\n" + "=" * 70 + f"\nRUN {n}/{len(STEPS)}: {script.name}  -  {what}\n" + "=" * 70)
    if subprocess.run([sys.executable, str(script)], cwd=HERE).returncode != 0:
        sys.exit(f"\nSTOPPED: {script.name} failed. Fix it and run again.")

print("\nMODEL 1 FINISHED - all steps ran successfully.")