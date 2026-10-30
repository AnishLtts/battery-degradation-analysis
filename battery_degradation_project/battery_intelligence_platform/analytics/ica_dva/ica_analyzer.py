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
# ICA CONFIGURATION
# ============================================================

MINIMUM_POINTS_PER_CYCLE = 20
MINIMUM_VOLTAGE_RANGE = 0.10
MINIMUM_CAPACITY_RANGE = 0.01

# Number of voltage points used for the common interpolation grid.
INTERPOLATION_POINTS = 250

# Rolling smoothing window. Must be an odd number.
SMOOTHING_WINDOW = 11

# Maximum number of cycles displayed in one graph.
MAXIMUM_GRAPH_CYCLES = 6


# ============================================================
# VALIDATE ICA DATASET
# ============================================================

def validate_ica_dataset(dataframe):
    """
    Validate and clean a battery dataset for ICA.

    ICA requires:
        cycle
        voltage_v
        capacity_ah

    Current is optional, but it is used to identify discharge
    measurements when available.
    """

    required_columns = [
        "cycle",
        "voltage_v",
        "capacity_ah",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "ICA requires the following columns: "
            + ", ".join(missing_columns)
        )

    cleaned_data = dataframe.copy()

    numeric_columns = [
        "cycle",
        "voltage_v",
        "capacity_ah",
        "current_a",
        "time_s",
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
            "capacity_ah",
        ]
    )

    cleaned_data = cleaned_data[
        np.isfinite(cleaned_data["cycle"])
        & np.isfinite(cleaned_data["voltage_v"])
        & np.isfinite(cleaned_data["capacity_ah"])
    ].copy()

    cleaned_data = cleaned_data[
        cleaned_data["capacity_ah"] >= 0
    ]

    if cleaned_data.empty:
        raise ValueError(
            "No valid cycle, voltage, and capacity records "
            "were found."
        )

    cleaned_data = cleaned_data.sort_values(
        by=["cycle"],
        kind="stable",
    ).reset_index(drop=True)

    return cleaned_data


# ============================================================
# IDENTIFY DISCHARGE DATA
# ============================================================

def select_discharge_records(dataframe):
    """
    Identify discharge measurements using current.

    NASA B0005 commonly uses negative current during discharge.
    If current is unavailable or filtering leaves too little
    data, all valid records are retained.
    """

    information = {
        "current_available": False,
        "discharge_sign": "Unknown",
        "original_records": int(len(dataframe)),
        "selected_records": int(len(dataframe)),
        "message": (
            "Current was unavailable. All valid records were used."
        ),
    }

    if "current_a" not in dataframe.columns:
        return dataframe.copy(), information

    valid_current = dataframe["current_a"].dropna()

    if valid_current.empty:
        information["message"] = (
            "Current was present but contained no valid values. "
            "All valid records were used."
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
            "Current values were approximately zero. "
            "All valid records were used."
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

    selected_cycle_count = selected_data[
        "cycle"
    ].nunique()

    if (
        len(selected_data) < MINIMUM_POINTS_PER_CYCLE
        or selected_cycle_count == 0
    ):
        information["message"] = (
            "Discharge filtering left insufficient data. "
            "All valid voltage-capacity records were used."
        )

        return dataframe.copy(), information

    information["selected_records"] = int(
        len(selected_data)
    )

    return selected_data, information


# ============================================================
# SMOOTH A NUMERIC SIGNAL
# ============================================================

def smooth_signal(values, window=SMOOTHING_WINDOW):
    """
    Smooth a numeric signal using a centered rolling median
    followed by a centered rolling mean.

    This avoids requiring an additional SciPy installation.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    number_of_values = len(values)

    if number_of_values < 5:
        return values.copy()

    effective_window = min(
        window,
        number_of_values,
    )

    if effective_window % 2 == 0:
        effective_window -= 1

    effective_window = max(
        3,
        effective_window,
    )

    signal_series = pd.Series(values)

    median_smoothed = (
        signal_series
        .rolling(
            window=effective_window,
            center=True,
            min_periods=1,
        )
        .median()
    )

    mean_smoothed = (
        median_smoothed
        .rolling(
            window=effective_window,
            center=True,
            min_periods=1,
        )
        .mean()
    )

    return mean_smoothed.to_numpy(
        dtype=float
    )


# ============================================================
# PREPARE ONE CYCLE
# ============================================================

def prepare_cycle_for_ica(cycle_data):
    """
    Prepare one cycle for ICA calculation.

    The function:
    1. Removes invalid measurements.
    2. Averages duplicate voltage measurements.
    3. Sorts voltage in ascending order.
    4. Checks point count and signal ranges.
    """

    prepared_cycle = cycle_data[
        [
            "voltage_v",
            "capacity_ah",
        ]
    ].copy()

    prepared_cycle = prepared_cycle.dropna(
        subset=[
            "voltage_v",
            "capacity_ah",
        ]
    )

    prepared_cycle = prepared_cycle[
        np.isfinite(prepared_cycle["voltage_v"])
        & np.isfinite(prepared_cycle["capacity_ah"])
    ]

    if prepared_cycle.empty:
        return None, "No valid voltage-capacity records."

    # Round voltage slightly before grouping to reduce tiny
    # floating-point differences between near-duplicate values.
    prepared_cycle["voltage_group"] = (
        prepared_cycle["voltage_v"].round(5)
    )

    prepared_cycle = (
        prepared_cycle.groupby(
            "voltage_group",
            as_index=False,
        )
        .agg(
            voltage_v=("voltage_v", "mean"),
            capacity_ah=("capacity_ah", "median"),
        )
    )

    prepared_cycle = prepared_cycle.sort_values(
        "voltage_v"
    ).reset_index(drop=True)

    point_count = len(prepared_cycle)

    voltage_range = (
        prepared_cycle["voltage_v"].max()
        - prepared_cycle["voltage_v"].min()
    )

    capacity_range = (
        prepared_cycle["capacity_ah"].max()
        - prepared_cycle["capacity_ah"].min()
    )

    if point_count < MINIMUM_POINTS_PER_CYCLE:
        return (
            None,
            f"Only {point_count} unique voltage points were found.",
        )

    if voltage_range < MINIMUM_VOLTAGE_RANGE:
        return (
            None,
            "The voltage range is too small for reliable ICA.",
        )

    if capacity_range < MINIMUM_CAPACITY_RANGE:
        return (
            None,
            "The capacity range is too small for reliable ICA.",
        )

    return prepared_cycle, None


# ============================================================
# CALCULATE ICA FOR ONE CYCLE
# ============================================================

def calculate_cycle_ica(
    cycle_data,
    cycle_number,
):
    """
    Calculate incremental capacity dQ/dV for one cycle.

    The original voltage-capacity measurements are interpolated
    onto an evenly spaced voltage grid before differentiation.
    """

    prepared_cycle, rejection_reason = (
        prepare_cycle_for_ica(
            cycle_data
        )
    )

    if prepared_cycle is None:
        return None, {
            "cycle": cycle_number,
            "status": "Rejected",
            "reason": rejection_reason,
        }

    voltage_values = prepared_cycle[
        "voltage_v"
    ].to_numpy(
        dtype=float
    )

    capacity_values = prepared_cycle[
        "capacity_ah"
    ].to_numpy(
        dtype=float
    )

    minimum_voltage = float(
        voltage_values.min()
    )

    maximum_voltage = float(
        voltage_values.max()
    )

    voltage_grid = np.linspace(
        minimum_voltage,
        maximum_voltage,
        INTERPOLATION_POINTS,
    )

    interpolated_capacity = np.interp(
        voltage_grid,
        voltage_values,
        capacity_values,
    )

    smoothed_capacity = smooth_signal(
        interpolated_capacity
    )

    # Calculate dQ/dV.
    incremental_capacity = np.gradient(
        smoothed_capacity,
        voltage_grid,
    )

    # Discharge datasets can produce negative dQ/dV depending
    # on orientation. ICA magnitude is used for a clear positive
    # peak representation.
    incremental_capacity = np.abs(
        incremental_capacity
    )

    incremental_capacity = smooth_signal(
        incremental_capacity,
        window=9,
    )

    # Remove extreme derivative spikes using robust percentile
    # clipping. This affects visualization stability but does
    # not fabricate additional measurements.
    lower_limit = float(
        np.nanpercentile(
            incremental_capacity,
            1,
        )
    )

    upper_limit = float(
        np.nanpercentile(
            incremental_capacity,
            99,
        )
    )

    if upper_limit <= lower_limit:
        return None, {
            "cycle": cycle_number,
            "status": "Rejected",
            "reason": (
                "The calculated ICA signal contained "
                "insufficient variation."
            ),
        }

    clipped_incremental_capacity = np.clip(
        incremental_capacity,
        lower_limit,
        upper_limit,
    )

    peak_index = int(
        np.nanargmax(
            clipped_incremental_capacity
        )
    )

    peak_voltage = float(
        voltage_grid[peak_index]
    )

    peak_ica = float(
        clipped_incremental_capacity[peak_index]
    )

    result = pd.DataFrame(
        {
            "cycle": cycle_number,
            "voltage_v": voltage_grid,
            "interpolated_capacity_ah": (
                interpolated_capacity
            ),
            "smoothed_capacity_ah": (
                smoothed_capacity
            ),
            "ica_ah_per_v": (
                clipped_incremental_capacity
            ),
        }
    )

    summary = {
        "cycle": cycle_number,
        "status": "Accepted",
        "reason": "ICA calculated successfully.",
        "original_cycle_records": int(
            len(cycle_data)
        ),
        "unique_voltage_points": int(
            len(prepared_cycle)
        ),
        "minimum_voltage_v": minimum_voltage,
        "maximum_voltage_v": maximum_voltage,
        "voltage_range_v": float(
            maximum_voltage - minimum_voltage
        ),
        "minimum_capacity_ah": float(
            capacity_values.min()
        ),
        "maximum_capacity_ah": float(
            capacity_values.max()
        ),
        "capacity_range_ah": float(
            capacity_values.max()
            - capacity_values.min()
        ),
        "peak_ica_ah_per_v": peak_ica,
        "peak_voltage_v": peak_voltage,
    }

    return result, summary


# ============================================================
# SELECT REPRESENTATIVE GRAPH CYCLES
# ============================================================

def select_graph_cycles(
    accepted_cycles,
    maximum_cycles=MAXIMUM_GRAPH_CYCLES,
):
    """
    Select evenly distributed cycles for a readable ICA graph.

    Example for 168 cycles:
        approximately cycles 1, 34, 68, 101, 135, and 168.
    """

    available_cycles = sorted(
        accepted_cycles
    )

    if len(available_cycles) <= maximum_cycles:
        return available_cycles

    selected_indices = np.linspace(
        0,
        len(available_cycles) - 1,
        maximum_cycles,
        dtype=int,
    )

    selected_cycles = [
        available_cycles[index]
        for index in selected_indices
    ]

    return list(
        dict.fromkeys(selected_cycles)
    )


# ============================================================
# RUN ICA FOR ALL CYCLES
# ============================================================

def analyze_ica(dataframe):
    """
    Calculate ICA for every valid discharge cycle.
    """

    validated_data = validate_ica_dataset(
        dataframe
    )

    discharge_data, discharge_information = (
        select_discharge_records(
            validated_data
        )
    )

    detailed_results = []
    summary_records = []

    for cycle_number, cycle_data in (
        discharge_data.groupby(
            "cycle",
            sort=True,
        )
    ):
        cycle_result, cycle_summary = (
            calculate_cycle_ica(
                cycle_data,
                cycle_number,
            )
        )

        summary_records.append(
            cycle_summary
        )

        if cycle_result is not None:
            detailed_results.append(
                cycle_result
            )

    summary_table = pd.DataFrame(
        summary_records
    )

    if not detailed_results:
        raise ValueError(
            "ICA could not be calculated for any cycle. "
            "Check the voltage and capacity measurements."
        )

    combined_results = pd.concat(
        detailed_results,
        ignore_index=True,
    )

    accepted_cycles = (
        summary_table.loc[
            summary_table["status"] == "Accepted",
            "cycle",
        ]
        .tolist()
    )

    graph_cycles = select_graph_cycles(
        accepted_cycles
    )

    analysis_summary = {
        "original_records": int(
            len(dataframe)
        ),
        "validated_records": int(
            len(validated_data)
        ),
        "discharge_records": int(
            len(discharge_data)
        ),
        "cycles_found": int(
            discharge_data["cycle"].nunique()
        ),
        "cycles_analyzed": int(
            len(accepted_cycles)
        ),
        "cycles_rejected": int(
            (
                summary_table["status"]
                == "Rejected"
            ).sum()
        ),
        "graph_cycles": graph_cycles,
    }

    return (
        combined_results,
        summary_table,
        analysis_summary,
        discharge_information,
    )


# ============================================================
# CREATE ICA GRAPH
# ============================================================

def create_ica_graph(
    detailed_results,
    graph_cycles,
):
    """
    Create an ICA-versus-voltage graph for representative cycles.
    """

    figure, axis = plt.subplots(
        figsize=(13, 8),
    )

    color_map = plt.get_cmap(
        "viridis"
    )

    number_of_cycles = len(
        graph_cycles
    )

    for index, cycle_number in enumerate(
        graph_cycles
    ):
        cycle_result = detailed_results[
            detailed_results["cycle"]
            == cycle_number
        ]

        if cycle_result.empty:
            continue

        if number_of_cycles == 1:
            color_position = 0.5
        else:
            color_position = (
                index
                / (number_of_cycles - 1)
            )

        axis.plot(
            cycle_result["voltage_v"],
            cycle_result["ica_ah_per_v"],
            linewidth=2,
            color=color_map(color_position),
            label=f"Cycle {cycle_number:g}",
        )

    axis.set_title(
        "Incremental Capacity Analysis (ICA)",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    axis.set_xlabel(
        "Voltage (V)",
        fontsize=12,
        fontweight="bold",
    )

    axis.set_ylabel(
        "Incremental Capacity |dQ/dV| (Ah/V)",
        fontsize=12,
        fontweight="bold",
    )

    axis.grid(
        True,
        alpha=0.30,
        linestyle="--",
    )

    axis.legend(
        title="Selected discharge cycles",
        fontsize=9,
        title_fontsize=10,
        ncol=2,
    )

    figure.tight_layout()

    return figure


# ============================================================
# SAVE ICA RESULTS
# ============================================================

def save_ica_results(
    detailed_results,
    cycle_summary,
    analysis_summary,
    output_directory,
):
    """
    Save detailed ICA data, cycle summary, analysis summary,
    and the ICA graph.
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
        / "ica_detailed_results.csv"
    )

    cycle_summary_path = (
        output_directory
        / "ica_cycle_summary.csv"
    )

    analysis_summary_path = (
        output_directory
        / "ica_analysis_summary.csv"
    )

    graph_path = (
        output_directory
        / "ica_analysis.png"
    )

    detailed_results.to_csv(
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
                str(cycle)
                for cycle in value
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

    figure = create_ica_graph(
        detailed_results,
        analysis_summary["graph_cycles"],
    )

    figure.savefig(
        graph_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)

    # Automatically open the graph on Windows.
    try:
        if os.name == "nt":
            os.startfile(str(graph_path))
        else:
            print(
                "\nAutomatic graph opening is only enabled "
                "for Windows."
            )
    except Exception as open_error:
        print(
            "\nThe ICA graph was saved successfully, "
            "but it could not be opened automatically."
        )
        print(
            f"Opening error: {open_error}"
        )

    return {
        "detailed_results": detailed_path,
        "cycle_summary": cycle_summary_path,
        "analysis_summary": analysis_summary_path,
        "graph": graph_path,
    }


# ============================================================
# PRINT ICA REPORT
# ============================================================

def print_ica_report(
    analysis_summary,
    discharge_information,
    cycle_summary,
    saved_files,
    unknown_columns,
):
    """
    Print a readable ICA analysis report.
    """

    print("\n" + "=" * 75)
    print("INCREMENTAL CAPACITY ANALYSIS")
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
        f"Discharge records used: "
        f"{analysis_summary['discharge_records']:,}"
    )

    print(
        f"Cycles found: "
        f"{analysis_summary['cycles_found']}"
    )

    print(
        f"Cycles successfully analyzed: "
        f"{analysis_summary['cycles_analyzed']}"
    )

    print(
        f"Cycles rejected: "
        f"{analysis_summary['cycles_rejected']}"
    )

    graph_cycle_text = ", ".join(
        str(cycle)
        for cycle in analysis_summary[
            "graph_cycles"
        ]
    )

    print(
        f"Cycles displayed in graph: "
        f"{graph_cycle_text}"
    )

    print("\nDISCHARGE DETECTION")

    print(
        f"  {discharge_information['message']}"
    )

    accepted_summary = cycle_summary[
        cycle_summary["status"] == "Accepted"
    ]

    if not accepted_summary.empty:
        peak_ica_cycle_index = (
            accepted_summary[
                "peak_ica_ah_per_v"
            ].idxmax()
        )

        strongest_peak = accepted_summary.loc[
            peak_ica_cycle_index
        ]

        print("\nICA PEAK INFORMATION")

        print(
            f"  Strongest calculated peak cycle: "
            f"{strongest_peak['cycle']:g}"
        )

        print(
            f"  Peak voltage: "
            f"{strongest_peak['peak_voltage_v']:.4f} V"
        )

        print(
            f"  Peak ICA magnitude: "
            f"{strongest_peak['peak_ica_ah_per_v']:.4f} Ah/V"
        )

    print("\nSAVED RESULTS")

    for result_name, result_path in (
        saved_files.items()
    ):
        print(
            f"  {result_name}: {result_path}"
        )

    if unknown_columns:
        print("\nUNRECOGNIZED ORIGINAL COLUMNS")

        for column in unknown_columns:
            print(f"  - {column}")

    print("\nIMPORTANT")

    print(
        "  ICA peaks are sensitive to measurement noise, "
        "sampling interval, interpolation, and smoothing."
    )

    print(
        "  This module is intended for analytical diagnostics. "
        "Peak interpretation should be validated against cell "
        "chemistry and controlled laboratory measurements."
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
            ica_results,
            ica_cycle_summary,
            ica_summary,
            discharge_info,
        ) = analyze_ica(
            standardized_data
        )

        output_folder = (
            PLATFORM_FOLDER
            / "outputs"
            / Path(dataset_path).stem
            / "ica_analysis"
        )

        saved_results = save_ica_results(
            ica_results,
            ica_cycle_summary,
            ica_summary,
            output_folder,
        )

        print_ica_report(
            ica_summary,
            discharge_info,
            ica_cycle_summary,
            saved_results,
            unknown_columns,
        )

    except Exception as error:
        print(
            f"\nICA analysis failed: {error}"
        )