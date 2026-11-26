"""
Capacity estimate by coulomb counting against SOC change.

For a recording (or cycle) with current, time, and a reference SOC column,
the capacity is the charge that moved through the battery divided by the SOC
change it caused:

    capacity_ah = |integral(current dt) / 3600| / (|SOC change| / 100)

This needs no capacity column. It works only when:
  - the SOC change is large enough to be reliable (default at least 50 %)
  - current and SOC move in the same direction (charge in, SOC up, or
    discharge out, SOC down), which rules out sign mistakes

Input columns: time_s, current_a, soc_percent, plus an optional grouping
column (for example Source_File, or cycle) so each recording or cycle is
estimated separately.
"""

import numpy as np
import pandas as pd

MINIMUM_SOC_SPAN_PERCENT = 50.0


def estimate_capacity_from_soc(
    dataframe,
    group_column=None,
    time_column="time_s",
    current_column="current_a",
    soc_column="soc_percent",
    minimum_soc_span=MINIMUM_SOC_SPAN_PERCENT,
):
    """
    Return one row per group with the coulomb-counted capacity.

    Columns of the result: group, start_soc, end_soc, soc_span,
    charge_ah, capacity_ah, status, reason.
    """
    if group_column is None:
        groups = [("all", dataframe)]
    else:
        groups = dataframe.groupby(group_column, sort=False)

    results = []
    for name, group in groups:
        group = group.sort_values(time_column)
        time = group[time_column].to_numpy(dtype=float)
        current = group[current_column].to_numpy(dtype=float)
        soc = group[soc_column].to_numpy(dtype=float)

        record = {
            "group": name,
            "start_soc": float(soc[0]) if len(soc) else np.nan,
            "end_soc": float(soc[-1]) if len(soc) else np.nan,
        }

        if len(time) < 2:
            results.append({**record, "soc_span": np.nan, "charge_ah": np.nan,
                            "capacity_ah": np.nan, "status": "Rejected",
                            "reason": "Fewer than two samples."})
            continue

        # Time step to each sample; the first sample contributes nothing.
        step = np.diff(time, prepend=time[0])
        # Net charge in Ah. Discharge current is negative in this dataset.
        net_charge_ah = float(np.sum(current * step) / 3600.0)
        soc_change = float(soc[-1] - soc[0])
        soc_span = abs(soc_change)

        record.update({"soc_span": soc_span, "charge_ah": net_charge_ah})

        if soc_span < minimum_soc_span:
            results.append({**record, "capacity_ah": np.nan, "status": "Rejected",
                            "reason": f"SOC change {soc_span:.1f}% is below {minimum_soc_span:.0f}%."})
            continue

        if net_charge_ah == 0 or np.sign(net_charge_ah) != np.sign(soc_change):
            results.append({**record, "capacity_ah": np.nan, "status": "Rejected",
                            "reason": "Current and SOC change point in opposite directions."})
            continue

        capacity = abs(net_charge_ah) / (soc_span / 100.0)
        results.append({**record, "capacity_ah": capacity, "status": "Accepted",
                        "reason": "Coulomb-counted capacity."})

    return pd.DataFrame(results)
