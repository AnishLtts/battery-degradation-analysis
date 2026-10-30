from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# LOCATE PLATFORM FOLDERS
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
PLATFORM_FOLDER = CURRENT_FILE.parent.parent.parent
CORE_FOLDER = PLATFORM_FOLDER / "core"

sys.path.insert(0, str(CORE_FOLDER))

from file_reader import load_battery_dataset
from column_mapper import (
    apply_column_mapping,
    detect_column_mapping,
)


# ============================================================
# VOLTAGE DIAGNOSTIC CONFIGURATION
# ============================================================

# Broad screening limits for typical lithium-ion cell data.
# These are diagnostic thresholds, not universal safety limits.
LOW_VOLTAGE_WARNING_V = 2.50
HIGH_VOLTAGE_WARNING_V = 4.25

MINIMUM_POINTS_PER_CYCLE = 10
MAXIMUM_GRAPH_CYCLES = 6

# These settings are used only to calculate exploratory
# voltage-rise metrics.
#
# Voltage-rise values are not used for safety classification
# because filtering discharge rows can join measurements across
# rest periods, time gaps, and load transitions.
VOLTAGE_RISE_TOLERANCE_V = 0.05
MINIMUM_CONSECUTIVE_RISES = 3
VOLTAGE_SMOOTHING_WINDOW = 5


# ============================================================
# VALIDATE VOLTAGE DATASET
# ============================================================

def validate_voltage_dataset(dataframe):
    """
    Validate and clean a dataset for voltage diagnostics.

    Required columns:
        cycle
        voltage_v

    Optional columns:
        capacity_ah
        time_s
        current_a
        temperature_c
    """

    required_columns = [
        "cycle",
        "voltage_v",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Voltage diagnostics requires: "
            + ", ".join(missing_columns)
        )

    cleaned_data = dataframe.copy()

    numeric_columns = [
        "cycle",
        "voltage_v",
        "capacity_ah",
        "time_s",
        "current_a",
        "temperature_c",
    ]

    for column in numeric_columns:
        if column in cleaned_data.columns:
            cleaned_data[column] = pd.to_numeric(
                cleaned_data[column],
                errors="coerce",
            )

    cleaned_data = cleaned_data.dropna(
        subset=[
            "cycle",
            "voltage_v",
        ]
    )

    cleaned_data = cleaned_data[
        np.isfinite(cleaned_data["cycle"])
        & np.isfinite(cleaned_data["voltage_v"])
    ].copy()

    if cleaned_data.empty:
        raise ValueError(
            "No valid cycle and voltage records were found."
        )

    cleaned_data = cleaned_data.sort_values(
        by=["cycle"],
        kind="stable",
    ).reset_index(drop=True)

    return cleaned_data


# ============================================================
# SELECT DISCHARGE RECORDS
# ============================================================

def select_discharge_records(dataframe):
    """
    Identify discharge records using current direction.

    When current is unavailable or filtering leaves insufficient
    data, all valid voltage records are retained.
    """

    information = {
        "current_available": False,
        "discharge_sign": "Unknown",
        "original_records": int(len(dataframe)),
        "selected_records": int(len(dataframe)),
        "message": (
            "Current was unavailable. "
            "All valid voltage records were used."
        ),
    }

    if "current_a" not in dataframe.columns:
        return dataframe.copy(), information

    valid_current = dataframe["current_a"].dropna()

    if valid_current.empty:
        information["message"] = (
            "Current exists but contains no valid numeric values. "
            "All valid voltage records were used."
        )

        return dataframe.copy(), information

    information["current_available"] = True

    negative_count = int(
        (valid_current < -0.001).sum()
    )

    positive_count = int(
        (valid_current > 0.001).sum()
    )

    if negative_count == 0 and positive_count == 0:
        information["message"] = (
            "Current values are approximately zero. "
            "All valid voltage records were used."
        )

        return dataframe.copy(), information

    if negative_count >= positive_count:
        selected_data = dataframe[
            dataframe["current_a"] < -0.001
        ].copy()

        information["discharge_sign"] = "Negative"
        information["message"] = (
            "Negative current was treated as discharge."
        )

    else:
        selected_data = dataframe[
            dataframe["current_a"] > 0.001
        ].copy()

        information["discharge_sign"] = "Positive"
        information["message"] = (
            "Positive current was treated as discharge."
        )

    if (
        len(selected_data) < MINIMUM_POINTS_PER_CYCLE
        or selected_data["cycle"].nunique() == 0
    ):
        information["message"] = (
            "Discharge filtering left insufficient data. "
            "All valid voltage records were used."
        )

        return dataframe.copy(), information

    information["selected_records"] = int(
        len(selected_data)
    )

    return selected_data, information


# ============================================================
# SORT ONE CYCLE
# ============================================================

def sort_cycle_records(cycle_data):
    """
    Sort one cycle using time, capacity, or original row order.
    """

    sorted_cycle = cycle_data.copy()

    if (
        "time_s" in sorted_cycle.columns
        and sorted_cycle["time_s"].notna().sum() >= 2
    ):
        sorted_cycle = sorted_cycle.sort_values(
            "time_s",
            kind="stable",
        )

        sequence_column = "time_s"

    elif (
        "capacity_ah" in sorted_cycle.columns
        and sorted_cycle["capacity_ah"].notna().sum() >= 2
    ):
        sorted_cycle = sorted_cycle.sort_values(
            "capacity_ah",
            kind="stable",
        )

        sequence_column = "capacity_ah"

    else:
        sorted_cycle = sorted_cycle.sort_index()

        sequence_column = "original_row_order"

    return (
        sorted_cycle.reset_index(drop=True),
        sequence_column,
    )


# ============================================================
# SMOOTH VOLTAGE SIGNAL
# ============================================================

def smooth_voltage_signal(voltage_values):
    """
    Smooth voltage using a centered rolling median.

    This signal is used only for exploratory voltage-rise
    metrics. Raw voltage is used for threshold warnings.
    """

    voltage_values = np.asarray(
        voltage_values,
        dtype=float,
    )

    point_count = len(
        voltage_values
    )

    if point_count < 3:
        return voltage_values.copy()

    effective_window = min(
        VOLTAGE_SMOOTHING_WINDOW,
        point_count,
    )

    if effective_window % 2 == 0:
        effective_window -= 1

    effective_window = max(
        3,
        effective_window,
    )

    smoothed_values = (
        pd.Series(voltage_values)
        .rolling(
            window=effective_window,
            center=True,
            min_periods=1,
        )
        .median()
        .to_numpy(
            dtype=float
        )
    )

    return smoothed_values


# ============================================================
# COUNT CONSECUTIVE TRUE EVENTS
# ============================================================

def count_consecutive_events(
    boolean_flags,
    minimum_length,
):
    """
    Count separate runs containing at least minimum_length
    consecutive True values.
    """

    event_count = 0
    current_run_length = 0
    longest_run_length = 0

    for event_detected in boolean_flags:
        if event_detected:
            current_run_length += 1

            longest_run_length = max(
                longest_run_length,
                current_run_length,
            )

        else:
            if current_run_length >= minimum_length:
                event_count += 1

            current_run_length = 0

    if current_run_length >= minimum_length:
        event_count += 1

    return event_count, longest_run_length


# ============================================================
# ANALYZE ONE CYCLE
# ============================================================

def analyze_voltage_cycle(
    cycle_number,
    cycle_data,
):
    """
    Calculate voltage statistics and defensible warnings for
    one cycle.

    Threshold screening uses raw voltage values.

    Voltage-rise metrics are calculated for exploration only.
    They are not added to warning_count because discharge-row
    filtering can join records across rest periods, current
    transitions, and time gaps.
    """

    sorted_cycle, sequence_column = (
        sort_cycle_records(
            cycle_data
        )
    )

    sorted_cycle = sorted_cycle[
        sorted_cycle["voltage_v"].notna()
    ].reset_index(drop=True)

    voltage_values = (
        sorted_cycle["voltage_v"]
        .to_numpy(
            dtype=float
        )
    )

    point_count = int(
        len(voltage_values)
    )

    if point_count == 0:
        return {
            "cycle": cycle_number,
            "status": "Rejected",
            "reason": (
                "No valid voltage measurements were found."
            ),
            "sequence_column": sequence_column,
            "point_count": 0,
            "warning_count": 1,
        }

    start_voltage = float(
        voltage_values[0]
    )

    end_voltage = float(
        voltage_values[-1]
    )

    minimum_voltage = float(
        np.min(voltage_values)
    )

    maximum_voltage = float(
        np.max(voltage_values)
    )

    mean_voltage = float(
        np.mean(voltage_values)
    )

    median_voltage = float(
        np.median(voltage_values)
    )

    standard_deviation = float(
        np.std(
            voltage_values,
            ddof=0,
        )
    )

    voltage_range = float(
        maximum_voltage - minimum_voltage
    )

    voltage_drop = float(
        start_voltage - end_voltage
    )

    low_voltage_record_count = int(
        (
            voltage_values
            < LOW_VOLTAGE_WARNING_V
        ).sum()
    )

    high_voltage_record_count = int(
        (
            voltage_values
            > HIGH_VOLTAGE_WARNING_V
        ).sum()
    )

    # --------------------------------------------------------
    # EXPLORATORY VOLTAGE-RISE METRICS
    # --------------------------------------------------------

    smoothed_voltage_values = smooth_voltage_signal(
        voltage_values
    )

    voltage_differences = np.diff(
        smoothed_voltage_values
    )

    voltage_rise_flags = (
        voltage_differences
        > VOLTAGE_RISE_TOLERANCE_V
    )

    voltage_drop_flags = (
        voltage_differences
        < -VOLTAGE_RISE_TOLERANCE_V
    )

    significant_voltage_rise_count = int(
        voltage_rise_flags.sum()
    )

    significant_voltage_drop_count = int(
        voltage_drop_flags.sum()
    )

    (
        consecutive_voltage_rise_events,
        longest_voltage_rise_run,
    ) = count_consecutive_events(
        voltage_rise_flags,
        MINIMUM_CONSECUTIVE_RISES,
    )

    (
        consecutive_voltage_drop_events,
        longest_voltage_drop_run,
    ) = count_consecutive_events(
        voltage_drop_flags,
        MINIMUM_CONSECUTIVE_RISES,
    )

    maximum_positive_voltage_step = 0.0
    maximum_negative_voltage_step = 0.0

    if len(voltage_differences) > 0:
        maximum_positive_voltage_step = float(
            max(
                0.0,
                np.max(voltage_differences),
            )
        )

        maximum_negative_voltage_step = float(
            min(
                0.0,
                np.min(voltage_differences),
            )
        )

    # --------------------------------------------------------
    # DEFENSIBLE CYCLE WARNINGS
    # --------------------------------------------------------

    warnings = []

    if point_count < MINIMUM_POINTS_PER_CYCLE:
        warnings.append(
            "Insufficient points for a reliable cycle curve."
        )

    if minimum_voltage < LOW_VOLTAGE_WARNING_V:
        warnings.append(
            "Voltage dropped below the low-voltage "
            f"screening threshold of "
            f"{LOW_VOLTAGE_WARNING_V:.2f} V."
        )

    if maximum_voltage > HIGH_VOLTAGE_WARNING_V:
        warnings.append(
            "Voltage exceeded the high-voltage "
            f"screening threshold of "
            f"{HIGH_VOLTAGE_WARNING_V:.2f} V."
        )

    if voltage_drop < 0:
        warnings.append(
            "End voltage is higher than start voltage. "
            "Verify operation direction and record ordering."
        )

    # Important:
    # Do not add consecutive_voltage_rise_events to warnings.
    # These values remain exploratory until current continuity
    # and adjacent time gaps are analyzed.

    warning_text = (
        " | ".join(warnings)
        if warnings
        else "No cycle-level voltage warning."
    )

    status = (
        "Accepted"
        if point_count >= MINIMUM_POINTS_PER_CYCLE
        else "Limited"
    )

    result = {
        "cycle": cycle_number,
        "status": status,
        "reason": warning_text,
        "sequence_column": sequence_column,
        "point_count": point_count,
        "start_voltage_v": start_voltage,
        "end_voltage_v": end_voltage,
        "minimum_voltage_v": minimum_voltage,
        "maximum_voltage_v": maximum_voltage,
        "mean_voltage_v": mean_voltage,
        "median_voltage_v": median_voltage,
        "voltage_standard_deviation_v": (
            standard_deviation
        ),
        "voltage_range_v": voltage_range,
        "voltage_drop_v": voltage_drop,
        "low_voltage_record_count": (
            low_voltage_record_count
        ),
        "high_voltage_record_count": (
            high_voltage_record_count
        ),
        "significant_voltage_rise_count_exploratory": (
            significant_voltage_rise_count
        ),
        "significant_voltage_drop_count_exploratory": (
            significant_voltage_drop_count
        ),
        "consecutive_voltage_rise_events_exploratory": (
            consecutive_voltage_rise_events
        ),
        "consecutive_voltage_drop_events_exploratory": (
            consecutive_voltage_drop_events
        ),
        "longest_voltage_rise_run_exploratory": (
            longest_voltage_rise_run
        ),
        "longest_voltage_drop_run_exploratory": (
            longest_voltage_drop_run
        ),
        "maximum_positive_voltage_step_v_exploratory": (
            maximum_positive_voltage_step
        ),
        "maximum_negative_voltage_step_v_exploratory": (
            maximum_negative_voltage_step
        ),
        "voltage_rise_metrics_used_for_safety": False,
        "warning_count": int(
            len(warnings)
        ),
    }

    if (
        "capacity_ah" in sorted_cycle.columns
        and sorted_cycle["capacity_ah"].notna().any()
    ):
        result["minimum_capacity_ah"] = float(
            sorted_cycle["capacity_ah"].min()
        )

        result["maximum_capacity_ah"] = float(
            sorted_cycle["capacity_ah"].max()
        )

    return result


# ============================================================
# SELECT REPRESENTATIVE CYCLES
# ============================================================

def select_graph_cycles(
    available_cycles,
    maximum_cycles=MAXIMUM_GRAPH_CYCLES,
):
    """
    Select evenly distributed cycles for a readable graph.
    """

    sorted_cycles = sorted(
        available_cycles
    )

    if len(sorted_cycles) <= maximum_cycles:
        return sorted_cycles

    selected_indices = np.linspace(
        0,
        len(sorted_cycles) - 1,
        maximum_cycles,
        dtype=int,
    )

    selected_cycles = [
        sorted_cycles[index]
        for index in selected_indices
    ]

    return list(
        dict.fromkeys(selected_cycles)
    )


# ============================================================
# RUN VOLTAGE DIAGNOSTICS
# ============================================================

def analyze_voltage_diagnostics(dataframe):
    """
    Run voltage diagnostics across all available cycles.
    """

    validated_data = validate_voltage_dataset(
        dataframe
    )

    discharge_data, discharge_information = (
        select_discharge_records(
            validated_data
        )
    )

    cycle_records = []

    for cycle_number, cycle_data in (
        discharge_data.groupby(
            "cycle",
            sort=True,
        )
    ):
        cycle_record = analyze_voltage_cycle(
            cycle_number,
            cycle_data,
        )

        cycle_records.append(
            cycle_record
        )

    cycle_summary = pd.DataFrame(
        cycle_records
    )

    if cycle_summary.empty:
        raise ValueError(
            "No cycle-level voltage results were created."
        )

    accepted_cycles = (
        cycle_summary.loc[
            cycle_summary["status"].isin(
                [
                    "Accepted",
                    "Limited",
                ]
            ),
            "cycle",
        ]
        .tolist()
    )

    graph_cycles = select_graph_cycles(
        accepted_cycles
    )

    voltage_values = (
        discharge_data["voltage_v"]
        .dropna()
    )

    cycles_with_warnings = int(
        (
            cycle_summary["warning_count"]
            > 0
        ).sum()
    )

    cycles_below_low_threshold = int(
        (
            cycle_summary[
                "low_voltage_record_count"
            ]
            > 0
        ).sum()
    )

    cycles_above_high_threshold = int(
        (
            cycle_summary[
                "high_voltage_record_count"
            ]
            > 0
        ).sum()
    )

    analysis_summary = {
        "original_records": int(
            len(dataframe)
        ),
        "validated_records": int(
            len(validated_data)
        ),
        "selected_discharge_records": int(
            len(discharge_data)
        ),
        "cycles_found": int(
            discharge_data["cycle"].nunique()
        ),
        "cycles_analyzed": int(
            len(cycle_summary)
        ),
        "cycles_with_warnings": (
            cycles_with_warnings
        ),
        "cycles_below_low_threshold": (
            cycles_below_low_threshold
        ),
        "cycles_above_high_threshold": (
            cycles_above_high_threshold
        ),
        "overall_minimum_voltage_v": float(
            voltage_values.min()
        ),
        "overall_maximum_voltage_v": float(
            voltage_values.max()
        ),
        "overall_mean_voltage_v": float(
            voltage_values.mean()
        ),
        "low_voltage_threshold_v": (
            LOW_VOLTAGE_WARNING_V
        ),
        "high_voltage_threshold_v": (
            HIGH_VOLTAGE_WARNING_V
        ),
        "voltage_rise_metrics_used_for_safety": False,
        "voltage_rise_exclusion_reason": (
            "Load transitions, rest periods, and time gaps can "
            "create voltage-recovery events after discharge-row "
            "filtering."
        ),
        "graph_cycles": graph_cycles,
    }

    return (
        validated_data,
        discharge_data,
        cycle_summary,
        analysis_summary,
        discharge_information,
    )


# ============================================================
# CREATE VOLTAGE GRAPH
# ============================================================

def create_voltage_graph(
    discharge_data,
    graph_cycles,
):
    """
    Create representative voltage curves.

    Capacity is used on the X-axis when available. Otherwise,
    elapsed time or measurement index is used.
    """

    has_capacity = (
        "capacity_ah" in discharge_data.columns
        and discharge_data["capacity_ah"].notna().any()
    )

    has_time = (
        "time_s" in discharge_data.columns
        and discharge_data["time_s"].notna().any()
    )

    figure, axis = plt.subplots(
        figsize=(13, 8),
    )

    color_map = plt.get_cmap(
        "viridis"
    )

    number_of_cycles = len(
        graph_cycles
    )

    x_label = "Measurement Index"

    for index, cycle_number in enumerate(
        graph_cycles
    ):
        cycle_data = discharge_data[
            discharge_data["cycle"]
            == cycle_number
        ].copy()

        if cycle_data.empty:
            continue

        if has_capacity:
            cycle_data = cycle_data.dropna(
                subset=[
                    "capacity_ah",
                    "voltage_v",
                ]
            )

            cycle_data = cycle_data.sort_values(
                "capacity_ah",
                kind="stable",
            )

            x_values = cycle_data[
                "capacity_ah"
            ]

            x_label = (
                "Discharged Capacity (Ah)"
            )

        elif has_time:
            cycle_data = cycle_data.dropna(
                subset=[
                    "time_s",
                    "voltage_v",
                ]
            )

            cycle_data = cycle_data.sort_values(
                "time_s",
                kind="stable",
            )

            x_values = cycle_data[
                "time_s"
            ]

            x_label = "Elapsed Time (s)"

        else:
            cycle_data = cycle_data.dropna(
                subset=[
                    "voltage_v",
                ]
            )

            x_values = np.arange(
                len(cycle_data)
            )

        if cycle_data.empty:
            continue

        if number_of_cycles == 1:
            color_position = 0.5
        else:
            color_position = (
                index
                / (number_of_cycles - 1)
            )

        axis.plot(
            x_values,
            cycle_data["voltage_v"],
            linewidth=2,
            color=color_map(color_position),
            label=f"Cycle {cycle_number:g}",
        )

    axis.axhline(
        LOW_VOLTAGE_WARNING_V,
        color="#D62728",
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Low-voltage screen "
            f"({LOW_VOLTAGE_WARNING_V:.2f} V)"
        ),
    )

    axis.axhline(
        HIGH_VOLTAGE_WARNING_V,
        color="#E69F00",
        linestyle="--",
        linewidth=1.5,
        label=(
            f"High-voltage screen "
            f"({HIGH_VOLTAGE_WARNING_V:.2f} V)"
        ),
    )

    axis.set_title(
        "Battery Voltage Diagnostics",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    axis.set_xlabel(
        x_label,
        fontsize=12,
        fontweight="bold",
    )

    axis.set_ylabel(
        "Voltage (V)",
        fontsize=12,
        fontweight="bold",
    )

    axis.grid(
        True,
        alpha=0.30,
        linestyle="--",
    )

    axis.legend(
        title="Representative cycles",
        fontsize=9,
        title_fontsize=10,
        ncol=2,
    )

    figure.tight_layout()

    return figure


# ============================================================
# BUILD VOLTAGE WARNINGS
# ============================================================

def build_voltage_warnings(
    cycle_summary,
    analysis_summary,
):
    """
    Create defensible dataset-level voltage warnings.

    Exploratory voltage-rise values are deliberately excluded
    from safety and warning classification.
    """

    warnings = []

    if (
        analysis_summary[
            "cycles_below_low_threshold"
        ]
        > 0
    ):
        warnings.append(
            f"{analysis_summary['cycles_below_low_threshold']} "
            "cycles contain voltage below "
            f"{LOW_VOLTAGE_WARNING_V:.2f} V."
        )

    if (
        analysis_summary[
            "cycles_above_high_threshold"
        ]
        > 0
    ):
        warnings.append(
            f"{analysis_summary['cycles_above_high_threshold']} "
            "cycles contain voltage above "
            f"{HIGH_VOLTAGE_WARNING_V:.2f} V."
        )

    cycles_with_negative_drop = cycle_summary[
        cycle_summary["voltage_drop_v"] < 0
    ]

    if not cycles_with_negative_drop.empty:
        warnings.append(
            f"{len(cycles_with_negative_drop)} cycles have "
            "end voltage greater than start voltage. "
            "Verify record ordering or operation direction."
        )

    limited_cycles = cycle_summary[
        cycle_summary["status"] == "Limited"
    ]

    if not limited_cycles.empty:
        warnings.append(
            f"{len(limited_cycles)} cycles contain fewer than "
            f"{MINIMUM_POINTS_PER_CYCLE} voltage measurements."
        )

    warnings.append(
        "Voltage-rise anomaly screening was not used for safety "
        "classification because load transitions, rest periods, "
        "and time gaps can create voltage-recovery events."
    )

    return warnings


# ============================================================
# SAVE VOLTAGE RESULTS
# ============================================================

def save_voltage_results(
    discharge_data,
    cycle_summary,
    analysis_summary,
    warnings,
    output_directory,
):
    """
    Save voltage records, summaries, warnings, and graph.
    """

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    detailed_path = (
        output_directory
        / "voltage_detailed_records.csv"
    )

    cycle_summary_path = (
        output_directory
        / "voltage_cycle_summary.csv"
    )

    analysis_summary_path = (
        output_directory
        / "voltage_analysis_summary.csv"
    )

    warning_path = (
        output_directory
        / "voltage_warnings.csv"
    )

    graph_path = (
        output_directory
        / "voltage_diagnostics.png"
    )

    discharge_data.to_csv(
        detailed_path,
        index=False,
    )

    cycle_summary.to_csv(
        cycle_summary_path,
        index=False,
    )

    summary_for_csv = {
        key: (
            ", ".join(
                str(value_item)
                for value_item in value
            )
            if isinstance(value, list)
            else value
        )
        for key, value in analysis_summary.items()
    }

    pd.DataFrame(
        [summary_for_csv]
    ).to_csv(
        analysis_summary_path,
        index=False,
    )

    pd.DataFrame(
        {
            "warning": warnings,
        }
    ).to_csv(
        warning_path,
        index=False,
    )

    figure = create_voltage_graph(
        discharge_data,
        analysis_summary["graph_cycles"],
    )

    figure.savefig(
        graph_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        figure
    )

    try:
        if os.name == "nt":
            os.startfile(
                str(graph_path)
            )

    except Exception as open_error:
        print(
            "\nThe voltage graph was saved successfully, "
            "but it could not be opened automatically."
        )

        print(
            f"Opening error: {open_error}"
        )

    return {
        "detailed_records": detailed_path,
        "cycle_summary": cycle_summary_path,
        "analysis_summary": analysis_summary_path,
        "warnings": warning_path,
        "graph": graph_path,
    }


# ============================================================
# PRINT VOLTAGE REPORT
# ============================================================

def print_voltage_report(
    analysis_summary,
    discharge_information,
    warnings,
    saved_files,
    unknown_columns,
):
    """
    Print a readable voltage diagnostic report.
    """

    print("\n" + "=" * 75)
    print("BATTERY VOLTAGE DIAGNOSTICS")
    print("=" * 75)

    print(
        f"\nOriginal records: "
        f"{analysis_summary['original_records']:,}"
    )

    print(
        f"Validated records: "
        f"{analysis_summary['validated_records']:,}"
    )

    print(
        f"Selected discharge records: "
        f"{analysis_summary['selected_discharge_records']:,}"
    )

    print(
        f"Cycles found: "
        f"{analysis_summary['cycles_found']}"
    )

    print(
        f"Cycles analyzed: "
        f"{analysis_summary['cycles_analyzed']}"
    )

    print(
        "Cycles with threshold or ordering warnings: "
        f"{analysis_summary['cycles_with_warnings']}"
    )

    print("\nOVERALL VOLTAGE RANGE")

    print(
        f"  Minimum voltage: "
        f"{analysis_summary['overall_minimum_voltage_v']:.4f} V"
    )

    print(
        f"  Maximum voltage: "
        f"{analysis_summary['overall_maximum_voltage_v']:.4f} V"
    )

    print(
        f"  Mean voltage: "
        f"{analysis_summary['overall_mean_voltage_v']:.4f} V"
    )

    print("\nTHRESHOLD SCREENING")

    print(
        f"  Low threshold: "
        f"{LOW_VOLTAGE_WARNING_V:.2f} V"
    )

    print(
        f"  Cycles below low threshold: "
        f"{analysis_summary['cycles_below_low_threshold']}"
    )

    print(
        f"  High threshold: "
        f"{HIGH_VOLTAGE_WARNING_V:.2f} V"
    )

    print(
        f"  Cycles above high threshold: "
        f"{analysis_summary['cycles_above_high_threshold']}"
    )

    print("\nVOLTAGE-RISE METRIC STATUS")

    print(
        "  Used for safety classification: No"
    )

    print(
        "  Reason: "
        f"{analysis_summary['voltage_rise_exclusion_reason']}"
    )

    print("\nDISCHARGE DETECTION")

    print(
        f"  {discharge_information['message']}"
    )

    print("\nWARNINGS")

    for warning in warnings:
        print(f"  - {warning}")

    if unknown_columns:
        print("\nUNRECOGNIZED ORIGINAL COLUMNS")

        for column in unknown_columns:
            print(f"  - {column}")

    print("\nSAVED RESULTS")

    for result_name, result_path in (
        saved_files.items()
    ):
        print(
            f"  {result_name}: {result_path}"
        )

    print("\nIMPORTANT")

    print(
        "  The 2.50 V and 4.25 V limits are broad screening "
        "thresholds for cell-level lithium-ion data."
    )

    print(
        "  Actual voltage limits depend on chemistry, cell "
        "manufacturer specifications, pack configuration, "
        "measurement accuracy, and BMS settings."
    )

    print(
        "  Exploratory voltage-rise metrics remain in the "
        "cycle summary but must not be interpreted as battery "
        "faults or safety events."
    )

    print(
        "  A threshold warning is a diagnostic flag and not "
        "proof of an unsafe battery condition."
    )

    print("=" * 75)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":
    dataset_path = input(
        "Paste the full path of a raw battery dataset: "
    ).strip().strip('"')

    try:
        battery_data = load_battery_dataset(
            dataset_path
        )

        mapping, unknown_columns = (
            detect_column_mapping(
                battery_data
            )
        )

        standardized_data = apply_column_mapping(
            battery_data,
            mapping,
        )

        (
            validated_data,
            discharge_data,
            voltage_cycle_summary,
            voltage_summary,
            discharge_info,
        ) = analyze_voltage_diagnostics(
            standardized_data
        )

        voltage_warnings = build_voltage_warnings(
            voltage_cycle_summary,
            voltage_summary,
        )

        output_folder = (
            PLATFORM_FOLDER
            / "outputs"
            / Path(dataset_path).stem
            / "voltage_diagnostics"
        )

        saved_results = save_voltage_results(
            discharge_data,
            voltage_cycle_summary,
            voltage_summary,
            voltage_warnings,
            output_folder,
        )

        print_voltage_report(
            voltage_summary,
            discharge_info,
            voltage_warnings,
            saved_results,
            unknown_columns,
        )

    except Exception as error:
        print(
            f"\nVoltage diagnostics failed: {error}"
        )