> **Where to run these commands:** from the `battery_degradation_project/battery_intelligence_platform` folder of this repository.

# Battery Intelligence Platform

Analyses battery cycling data and reports State of Health (SOH), State of Charge (SOC), capacity fade, and Incremental Capacity (ICA) and Differential Voltage (DVA) curves.

## Setup

Requires Python 3.11 or newer.

```
pip install -r requirements.txt
```

## Run

Run these commands from this folder (`battery_intelligence_platform`). The code imports its modules relative to this folder, so running from elsewhere will fail.

```
python core/a_capability_engine.py
```

On Windows, set UTF-8 output first, especially if you redirect the output to a file. Without it, the script can fail with a `'charmap' codec` error when it prints check-mark characters:

```
set PYTHONIOENCODING=utf-8
python core/a_capability_engine.py
```

The script asks for a dataset path. Enter the **full** path to a battery CSV, for example:

```
C:\path\to\battery_degradation_project\b0005_discharge_curves.csv
```

A relative path such as `../b0005_discharge_curves.csv` did not work in testing: the script kept waiting for input.

Column names are matched automatically, and unit differences (mV, mA, minutes, K) are converted. Each analysis runs only if its columns are present, and the report lists which analyses ran and which were skipped:

- **SOH, degradation, ICA, DVA:** need capacity and voltage per cycle.
- **Coulomb-counting capacity:** needs time, current, and SOC. No capacity column needed.
- **Datasets missing these columns** are still processed, but the skipped analyses are reported as unavailable.

## Sample datasets

The sample files are in the `battery_degradation_project` folder, one level up from this one.

| File | What it gives |
|---|---|
| `b0005_discharge_curves.csv` | SOH, ICA, DVA, SOC (168 cycles) |
| `b0006_discharge_curves.csv` | SOH, ICA, DVA, SOC (168 cycles) |
| `Battery_Data_Cleaned.csv` | SOH per battery (cycle-level data, no ICA or DVA) |
| `50_Cycle_Lithium-Ion Battery Cycle Life.csv` | SOH per cycle (per-cycle summary, no voltage curve, so no ICA or DVA) |
| `soc_dataset_1Hz.csv` | Coulomb-counting capacity per recording (no SOH: no aging data) |
| `unit_conversion_test.csv` | SOH only, 2 cycles (too few for ICA or DVA) |
| `synthetic_full_truth.csv` | SOH, ICA, DVA, SOC with known true values, for checking the pipeline |

## Desktop GUI

To run the analysis with a window instead of the command line:

```
python gui_app.py
```

Click **Browse** to choose a dataset, then **Run analysis**. The window shows the summary, the charts (choose one from the dropdown), and the run log. It uses the same pipeline as the command line, so the results are identical. The GUI uses only tkinter, which ships with Python.

Each dataset has its own results folder, `outputs/<dataset name>/`. The summary and charts always belong to the dataset in the Dataset box. If that dataset hasn't been run, the window says "No results yet" rather than showing another dataset's charts. Some datasets have no ICA, DVA, or SOC results, and the window says so for those charts.

## Outputs

Results are written to `outputs/<dataset name>/`. For example, `b0005_discharge_curves.csv` writes to `outputs/b0005_discharge_curves/`. Inside each dataset folder:

| Folder | Contents |
|---|---|
| `reports/` | `battery_report.txt` (full report), `battery_summary.csv` (headline values) |
| `graphs/` | SOH and capacity vs. cycle charts |
| `ica_analysis/`, `dva_analysis/` | ICA and DVA curves, peaks per cycle, and charts |
| `soc_analysis/` | SOC estimates per row and per cycle, and chart |
| `capacity_estimate/` | Coulomb-counting capacity per recording (only for datasets with SOC data) |

Folders that a dataset doesn't need, such as ICA or DVA for a file with too few voltage points, are not created.

Headline results for the included test file (`b0005_discharge_curves.csv`, 168 cycles): health score 71/100, grade C, SOH 71.29%, capacity loss 28.71%.

## Capacity estimate for datasets without a capacity column

Some datasets have current, time, and SOC but no capacity column. For these, the pipeline estimates capacity by coulomb counting: the charge that moved through the battery divided by the SOC change it caused. Each recording is estimated separately.

- Only recordings with at least 50% SOC change are accepted. Shorter recordings are reported as rejected.
- Estimates are reported separately. They do not feed the SOH analysis, because a partial recording is not a full capacity measurement.
- On the included SOC dataset (`soc_dataset_1Hz.csv`), 64 of 69 recordings were accepted, with a mean of 2.98 Ah.

## Known limitations

- **ICA and DVA peak heights are underestimated.** On a synthetic test file with known values, the ICA peak was about 20% low and the DVA peak about 6% low. Most of the ICA gap comes from the smoothing step in `analytics/ica_dva/ica_analyzer.py`. Removing it made real-data peaks unreliable (noise spikes), so it was reverted. Peak positions are correct.
- **Validation is limited.** On the synthetic file, capacity, SOH, and SOC match the known values exactly. This confirms the calculations are implemented as defined. It does not confirm accuracy on measured cells, because no reference values exist for the included data.
- **Charts may not open automatically.** On Windows, the script tries to open the ICA and DVA charts and may report an error. The files are still saved to `outputs/`; open them manually.
