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

Expected columns are voltage, current, temperature, capacity, time, and cycle. Column names are matched automatically, and unit differences (mV, mA, minutes, K) are converted.

## Outputs

Results are written to `outputs/`:

| Folder | Contents |
|---|---|
| `reports/` | `battery_report.txt` (full report), `battery_summary.csv` (headline values) |
| `graphs/` | SOH and capacity vs. cycle charts |
| `ica_analysis/`, `dva_analysis/` | ICA and DVA curves, peaks per cycle, and charts |
| `soc_analysis/` | SOC estimates per row and per cycle, and chart |

Headline results for the included test file (`b0005_discharge_curves.csv`, 168 cycles): health score 71/100, grade C, SOH 71.29%, capacity loss 28.71%.

## Known limitations

- **ICA and DVA peak heights are underestimated.** On a synthetic test file with known values, the ICA peak was about 20% low and the DVA peak about 6% low. Most of the ICA gap comes from the smoothing step in `analytics/ica_dva/ica_analyzer.py`. Removing it made real-data peaks unreliable (noise spikes), so it was reverted. Peak positions are correct.
- **Validation is limited.** On the synthetic file, capacity, SOH, and SOC match the known values exactly. This confirms the calculations are implemented as defined. It does not confirm accuracy on measured cells, because no reference values exist for the included data.
- **Charts may not open automatically.** On Windows, the script tries to open the ICA and DVA charts and may report an error. The files are still saved to `outputs/`; open them manually.
