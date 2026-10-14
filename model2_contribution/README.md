# Model 2 – How much of the capacity loss comes from each degradation mechanism?

**Problem statement:** *The system shall classify / quantify the contribution of each degradation mechanism (LLI, LAM-PE, LAM-NE, resistance growth).*

**What this folder does:** for every discharge cycle of a real battery (NASA B0005) it
- **quantifies** what percentage of the lost capacity was caused by each of the four mechanisms (the four numbers always add up to 100 %), and
- **classifies** the cycle by its dominant mechanism (or *Mixed* / *Too early*).

| Short name | Full name | What it means |
|---|---|---|
| **LLI** | Loss of Lithium Inventory | Lithium gets trapped (e.g. in the SEI layer) and can no longer move between the electrodes |
| **LAM-PE** | Loss of Active Material – Positive Electrode | Part of the positive electrode stops working, so it can hold less lithium |
| **LAM-NE** | Loss of Active Material – Negative Electrode | Part of the negative (graphite) electrode stops working |
| **RES** | Resistance growth | The cell's internal resistance increases, so the voltage drops more under load and it hits the cut-off voltage sooner |

Model 1 (folder `model1_diagnosis`) answers *which* mechanism is happening. Model 2 answers *how much of the damage each one is responsible for*. The two models are completely separate.

---

## The idea in one paragraph

Real battery data does not come with the answer ("this cycle lost 30 % from LLI"), so we cannot train on real data directly. Instead we **build a simulated battery** from its two electrodes, **calibrate it so it behaves like the real B0005 cell**, then **damage it thousands of times** with random amounts of the four mechanisms. For each simulated cell we know the true answer, so we can train a machine-learning model to read the answer from the shape of the discharge curve (ICA / dQ-dV features). Finally we give the model the **real** B0005 curves.

```
 real B0005 data ──► Step 5a export ──► Step 5b calibrate ──┐
                                                             ▼
 Step 1 fresh cell ─► Step 2 damage ─► Step 3 make 4000 training cells ─► Step 4 train model
                                                                                   │
                                   real B0005 curves ──────────────────────► Step 6 predict
                                                                                   │
                                                          b0005_contributions.csv + chart
```

---

## How to run it (in this order)

Open each file in VS Code and press ▶. Every step takes less than a minute.

| # | File | What it does | Creates |
|---|---|---|---|
| 1 | `step1_fresh_cell.py` | Builds a healthy cell from its positive and negative electrode curves (calibrated to B0005) | `step1_fresh_cell.png` |
| 2 | `step2_damage_one_at_a_time.py` | Applies each mechanism alone so you can see how each one changes the curves | `step2_damage_one_at_a_time.png` |
| 5a | `step5a_export_b0005.py` | Reads the NASA `.mat` files (B0005, B0006, B0007, B0018) and saves all discharge cycles as CSVs. **Check `DATA_DIR` at the top of the file first.** | `b000X_discharge_curves.csv` |
| 5b | `step5b_calibrate.py` | Checks that the simulated cell matches the real one (fresh and worn) | `step5b_calibration.png` |
| 3 | `step3_make_training_data.py` | Makes 4000 simulated aged cells with all 4 mechanisms mixed, plus realistic noise, and works out the true % for each | `training_data.csv` |
| 4 | `step4_train_model.py` | Trains a Random Forest model and tests it on 800 cells it has never seen | `contribution_model.pkl`, `step4_results.png` |
| 6 | `step6_predict_real_cells.py` | Uses the trained model on every real cycle of B0005, B0006, B0007 and B0018 | `b000X_contributions.csv`, `step6_b000X_contributions.png`, `step6_all_cells.png` |

Steps 5a/5b come before 3 because the real data is used to calibrate the simulation. If you only want to re-run the result, run **3 → 4 → 6**.

**Needed Python packages:** `numpy pandas scipy matplotlib scikit-learn`
(install with `pip install numpy pandas scipy matplotlib scikit-learn`)

**Important:** all files must be in the same folder, and the file names must be exactly as above (e.g. `step3_make_training_data.py`, not `step_3_...`), because later steps import code from earlier ones.

---

## Results for NASA B0005

Capacity fell from **1.851 Ah to 1.322 Ah (28.6 % lost)** over 168 cycles.

| Cycle | Capacity lost | LLI | LAM-PE | LAM-NE | RES |
|---|---|---|---|---|---|
| 12 | 2.3 % | 41 % | 8 % | 48 % | 2 % |
| 57 | 7.8 % | 23 % | 27 % | 47 % | 2 % |
| 93 | 17.4 % | 5 % | 62 % | 30 % | 3 % |
| 129 | 25.9 % | 2 % | 67 % | 28 % | 3 % |
| 168 | 28.6 % | 3 % | 67 % | 27 % | 3 % |

**Classification – dominant mechanism over the life of B0005** (printed by Step 6, column `dominant_mechanism` in the CSV):

| Cycles | Class |
|---|---|
| 1 – 11 | Too early (less than 2 % capacity lost) |
| 12 – 35 | Mixed (LLI + LAM-NE) – top two within 10 points |
| 36 – 60 | **LAM-NE** |
| 61 – 74 | Mixed (LAM-PE + LAM-NE) – the hand-over |
| 75 – 168 | **LAM-PE** |

Rules: below 2 % (smoothed) capacity loss → *Too early*; if the two biggest shares are less than 10 points apart → *Mixed*; otherwise the mechanism with the biggest share.

**Confidence ranges.** Every percentage now comes with an **80 % confidence range** (columns `*_low`, `*_high`, `*_pm` in the CSV). The Random Forest is 300 decision trees; where they disagree the range is wider. The range is calibrated on unseen simulated cells so the true answer falls inside it 80 % of the time (checked: 79–82 %).

| Cycle | LLI | LAM-PE | LAM-NE | RES |
|---|---|---|---|---|
| 39 | 35 ± 23 % | 14 ± 18 % | 48 ± 17 % | 2 ± 2 % |
| 93 | 5 ± 7 % | 62 ± 25 % | 30 ± 19 % | 3 ± 2 % |
| 168 | 3 ± 4 % | 67 ± 15 % | 27 ± 14 % | 3 ± 1 % |

`label_confidence` = **high** when the winner's range does not overlap the runner-up's. For B0005 this is true for about half of the LAM-PE cycles (mostly after cycle 125) and for none of the LAM-NE cycles – so "LAM-PE dominates late life" is a confident result, while "LAM-NE dominates cycles 36–60" is only the most likely answer.
The ranges cover the model's own uncertainty, not differences between the simulation and the real cell.

**In plain words:** early in life the loss comes mainly from lithium being trapped (LLI) and negative-electrode wear (LAM-NE). From about cycle 70 onward, loss of positive-electrode material (LAM-PE) becomes the main cause. Resistance growth contributes only 2–3 %, because NASA discharges at a gentle 2 A.

**How good is it?**

| Check | Result |
|---|---|
| Simulated fresh cell vs real cycle 1 | 18 mV voltage error, same capacity (1.851 Ah) |
| Simulated worn cell vs real cycle 168 | 26 mV voltage error, same capacity (1.322 Ah) |
| Model error on 800 unseen simulated cells | LLI 4.9, LAM-PE 8.5, LAM-NE 8.2, RES 1.1 percentage points |
| Model vs a separate direct curve fit on real cycles 93–168 | agree within about 7 percentage points |

---

## Test on other cells (never seen by the model)

The simulation was calibrated on B0005 and the model trained on that. B0006, B0007 and B0018 are a blind test. All curves are cut at 2.7 V so every cell is compared the same way (NASA discharged them to different voltages).

| Cell | Cycles | Capacity lost | LLI | LAM-PE | LAM-NE | RES | Dominant (late life) |
|---|---|---|---|---|---|---|---|
| B0005 (calibration) | 168 | 28.5 % | 3 ± 4 % | 67 ± 15 % | 27 ± 14 % | 3 ± 1 % | LAM-PE from cycle 75 |
| B0006 (test) | 168 | 41.9 % | 0 ± 0 % | 59 ± 13 % | 38 ± 12 % | 4 ± 1 % | LAM-PE from cycle 56 |
| B0007 (test) | 168 | 23.8 % | 2 ± 2 % | 68 ± 19 % | 28 ± 16 % | 3 ± 1 % | LAM-PE from cycle 90 |
| B0018 (test) | 132 | 27.8 % | 3 ± 3 % | 68 ± 18 % | 27 ± 15 % | 3 ± 1 % | LAM-PE from cycle 61 |

**What this shows:**
- **All four cells follow the same path** when plotted against capacity lost (`step6_all_cells.png`): LLI + LAM-NE early, LAM-PE taking over at around 15–20 % capacity loss. The cells age at different speeds, but the *pattern* is the same – which is what you expect from four cells of the same type, and a good sign the model is not just memorising B0005.
- **Independent check:** at 3 points in the life of each cell (12 checks), a direct curve fit (no ML) was compared with the model. The **dominant mechanism agreed in 12 of 12** checks; the individual percentages differed by about **11 points on average**, which is inside the model's confidence ranges.
- **B0006 is different early on:** it shows more LLI at the start and fades fastest (42 %).



1. **"Share of capacity loss" is not the same as "how much of the mechanism happened".** By cycle 168 about 33 % of the lithium is gone, yet LLI's share of the capacity loss is only ~3 %. That is because the positive electrode had already lost even more room (~40 %), so the lost lithium would not have fitted anyway – it costs no *extra* capacity. The model reports the share of capacity loss (that is what "contribution" means here).
2. **Early cycles are less certain.** Before about 8 % capacity loss there is too little damage to split reliably.
3. **LAM-NE is the hardest to pin down.** It changes the curve the least, so its number has the largest uncertainty.
4. **Chemistry assumption.** The electrode curves are the published Doyle/Dualfoil curves for lithium cobalt oxide (positive) and graphite (negative). If the NASA cells use a different chemistry, Step 1 needs different curves.
5. **Calibrated on B0005 only, tested on three other cells.** B0006, B0007 and B0018 were never used to build or train anything (see *Test on other cells* below). B0006 goes beyond 30 % capacity loss, further than the training data covers, so its numbers after that point are less reliable.
6. **Final proof needs a teardown.** The ML model and the direct fit share the same simulation, so their agreement is supporting evidence, not proof. Confirming the split for certain would need post-mortem analysis of an aged cell.

---

## How the % is calculated (for the curious)

The mechanisms interact, so "loss with only LLI" + "loss with only LAM-PE" + … does not add up to the real total loss. We use the **fair-share rule (Shapley value)**: add the four mechanisms in every possible order, measure how much extra capacity each one removes, and average. The four shares then add up exactly to the total loss. If one mechanism is completely hidden by another, it gets 0 %.

---

*Folder: `model2_contribution` · Data: NASA Prognostics Center of Excellence battery dataset (B0005)*