"""
MODEL 2 - STEP 4: Train the model and check how good it is
==========================================================
Input  : training_data.csv   (made by Step 3)
Output : contribution_model.pkl   (the trained model, used later on real data)
         step4_results.png        (picture: predicted vs true)

What happens:
  1. Load the table. Features (33 numbers per curve) = what the model SEES.
     Labels (4 percentages) = what the model must PREDICT.
  2. Hide 20 % of the cells from the model (the "test set"). It never
     trains on them, so they show how well it works on cells it has never seen.
  3. Train a Random Forest on the other 80 %.
     (A Random Forest = hundreds of small decision trees that each give an
      answer; the final answer is their average.)
  4. Predict the hidden 20 % and compare with the true answers.
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

HERE = Path(__file__).parent
MECHANISMS = ["LLI", "LAM_PE", "LAM_NE", "RES"]
LABELS = [f"pct_{m}" for m in MECHANISMS]


def to_percent(pred):
    """Make predictions valid percentages: no negatives, and each row sums to 100."""
    pred = np.clip(pred, 0, None)
    return 100 * pred / pred.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    # 1. load --------------------------------------------------------------
    df = pd.read_csv(HERE / "training_data.csv")
    feature_cols = [c for c in df.columns
                    if not c.startswith(("pct_", "amount_", "total_loss"))]
    X = df[feature_cols].values
    y = df[LABELS].values
    print(f"Loaded {len(df)} cells, {len(feature_cols)} features, {len(LABELS)} labels")

    # 2. split -------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0)
    print(f"Training on {len(X_train)} cells, testing on {len(X_test)} unseen cells")

    # 3. train -------------------------------------------------------------
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                  random_state=0, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Training done.\n")

    # 4. test --------------------------------------------------------------
    pred = to_percent(model.predict(X_test))

    print("How far off the model is on UNSEEN cells:")
    print(f"{'Mechanism':<12}{'Avg error':>12}{'R2 score':>10}")
    print("-" * 34)
    for i, m in enumerate(MECHANISMS):
        mae = mean_absolute_error(y_test[:, i], pred[:, i])
        r2 = r2_score(y_test[:, i], pred[:, i])
        print(f"{m:<12}{mae:>10.1f} %{r2:>10.2f}")
    print("(Avg error = how many percentage points off, on average."
          "  R2: 1.00 = perfect, 0 = useless)")

    # one example, so anyone can see what the model outputs
    k = 0
    print(f"\nExample - one unseen cell:")
    print(f"{'':<8}" + "".join(f"{m:>9}" for m in MECHANISMS))
    print(f"{'True':<8}" + "".join(f"{v:>8.1f}%" for v in y_test[k]))
    print(f"{'Model':<8}" + "".join(f"{v:>8.1f}%" for v in pred[k]))

    # picture: predicted vs true ------------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for i, m in enumerate(MECHANISMS):
        ax[i].scatter(y_test[:, i], pred[:, i], s=6, alpha=0.5)
        ax[i].plot([0, 100], [0, 100], "k--", lw=1)          # perfect line
        ax[i].set(title=m, xlabel="True share (%)", ylabel="Predicted share (%)",
                  xlim=(0, 100), ylim=(0, 100))
        ax[i].grid(alpha=0.3)
    plt.suptitle("Step 4: points on the dashed line = perfect prediction")
    plt.tight_layout()
    plt.savefig(HERE / "step4_results.png", dpi=110)

    # save the model --------------------------------------------------------
    joblib.dump({"model": model, "features": feature_cols, "mechanisms": MECHANISMS},
                HERE / "contribution_model.pkl")
    print("\nSaved model  :", HERE / "contribution_model.pkl")
    print("Saved picture:", HERE / "step4_results.png")