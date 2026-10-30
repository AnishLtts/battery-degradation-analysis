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
# DVA CONFIGURATION
# ============================================================

MINIMUM_POINTS_PER_CYCLE = 20
MINIMUM_CAPACITY_RANGE = 0.01
MINIMUM_VOLTAGE_RANGE = 0.10

INTERPOLATION_POINTS = 250
SMOOTHING_WINDOW = 11
MAXIMUM_GRAPH_CYCLES = 6


# ============================================================
# VALIDATE DVA DATASET
# ============================================================

def validate_dva_dataset(dataframe):
    """
    Validate and clean a battery dataset for DVA.

    DVA requires:
        cycle
        voltage_v
        capacity_ah

    Current is optional and is used to identify discharge
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
            "DVA requires the following columns: "
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
    Identify discharge records using the current column.

    Negative current is treated as discharge when negative
    measurements are dominant. Otherwise, positive current
    is treated as discharge.
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

    if len(selected_data) < MINIMUM_POINTS_PER_CYCLE:
        information["message"] = (
            "Discharge filtering left insufficient data. "
            "All valid records were used."
        )

        return dataframe.copy(), information

    information["selected_records"] = int(
        len(selected_data)
    )

    return selected_data, information


# ============================================================
# SMOOTH SIGNAL
# ============================================================

def smooth_signal(
    values,
    window=SMOOTHING_WINDOW,
):
    """
    Smooth a numeric signal using a centered rolling median
    followed by a centered rolling mean.
    """

    numeric_values = np.asarray(
        values,
        dtype=float,
    )

    number_of_values = len(
        numeric_values
    )

    if number_of_values < 5:
        return numeric_values.copy()

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

    signal_series = pd.Series(
        numeric_values
    )

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

def prepare_cycle_for_dva(cycle_data):
    """
    Prepare one battery cycle for DVA.

    Capacity is used as the independent variable, so duplicate
    capacity values are grouped before interpolation.
    """

    prepared_cycle = cycle_data[
        [
            "capacity_ah",
            "voltage_v",
        ]
    ].copy()

    prepared_cycle = prepared_cycle.dropna(
        subset=[
            "capacity_ah",
            "voltage_v",
        ]
    )

    prepared_cycle = prepared_cycle[
        np.isfinite(prepared_cycle["capacity_ah"])
        & np.isfinite(prepared_cycle["voltage_v"])
    ].copy()

    if prepared_cycle.empty:
        return None, "No valid capacity-voltage records."

    # Group almost-identical capacity values to prevent
    # division by zero during derivative calculation.
    prepared_cycle["capacity_group"] = (
        prepared_cycle["capacity_ah"].round(6)
    )

    prepared_cycle = (
        prepared_cycle.groupby(
            "capacity_group",
            as_index=False,
        )
        .agg(
            capacity_ah=("capacity_ah", "mean"),
            voltage_v=("voltage_v", "median"),
        )
    )

    prepared_cycle = prepared_cycle.sort_values(
        "capacity_ah"
    ).reset_index(drop=True)

    point_count = int(
        len(prepared_cycle)
    )

    capacity_range = float(
        prepared_cycle["capacity_ah"].max()
        - prepared_cycle["capacity_ah"].min()
    )

    voltage_range = float(
        prepared_cycle["voltage_v"].max()
        - prepared_cycle["voltage_v"].min()
    )

    if point_count < MINIMUM_POINTS_PER_CYCLE:
        return (
            None,
            f"Only {point_count} unique capacity points were found.",
        )

    if capacity_range < MINIMUM_CAPACITY_RANGE:
        return (
            None,
            "The capacity range is too small for reliable DVA.",
        )

    if voltage_range < MINIMUM_VOLTAGE_RANGE:
        return (
            None,
            "The voltage range is too small for reliable DVA.",
        )

    return prepared_cycle, None


# ============================================================
# CALCULATE DVA FOR ONE CYCLE
# ============================================================

def calculate_cycle_dva(
    cycle_data,
    cycle_number,
):
    """
    Calculate differential voltage dV/dQ for one cycle.
    """

    prepared_cycle, rejection_reason = (
        prepare_cycle_for_dva(
            cycle_data
        )
    )

    if prepared_cycle is None:
        return None, {
            "cycle": cycle_number,
            "status": "Rejected",
            "reason": rejection_reason,
        }

    capacity_values = prepared_cycle[
        "capacity_ah"
    ].to_numpy(
        dtype=float
    )

    voltage_values = prepared_cycle[
        "voltage_v"
    ].to_numpy(
        dtype=float
    )

    minimum_capacity = float(
        capacity_values.min()
    )

    maximum_capacity = float(
        capacity_values.max()
    )

    capacity_grid = np.linspace(
        minimum_capacity,
        maximum_capacity,
        INTERPOLATION_POINTS,
    )

    interpolated_voltage = np.interp(
        capacity_grid,
        capacity_values,
        voltage_values,
    )

    smoothed_voltage = smooth_signal(
        interpolated_voltage
    )

    # Calculate dV/dQ.
    differential_voltage = np.gradient(
        smoothed_voltage,
        capacity_grid,
    )

    # Discharge voltage normally decreases as discharged
    # capacity increases. Absolute magnitude gives a positive
    # and readable representation of the DVA peaks.
    differential_voltage = np.abs(
        differential_voltage
    )

    differential_voltage = smooth_signal(
        differential_voltage,
        window=9,
    )

    finite_values = differential_voltage[
        np.isfinite(differential_voltage)
    ]

    if len(finite_values) < 5:
        return None, {
            "cycle": cycle_number,
            "status": "Rejected",
            "reason": (
                "The calculated DVA signal did not contain "
                "enough finite values."
            ),
        }

    lower_limit = float(
        np.nanpercentile(
            finite_values,
            1,
        )
    )

    upper_limit = float(
        np.nanpercentile(
            finite_values,
            99,
        )
    )

    if upper_limit <= lower_limit:
        return None, {
            "cycle": cycle_number,
            "status": "Rejected",
            "reason": (
                "The calculated DVA signal contained "
                "insufficient variation."
            ),
        }

    clipped_differential_voltage = np.clip(
        differential_voltage,
        lower_limit,
        upper_limit,
    )

    peak_index = int(
        np.nanargmax(
            clipped_differential_voltage
        )
    )

    peak_capacity = float(
        capacity_grid[peak_index]
    )

    peak_dva = float(
        clipped_differential_voltage[peak_index]
    )

    peak_voltage = float(
        smoothed_voltage[peak_index]
    )

    detailed_result = pd.DataFrame(
        {
            "cycle": cycle_number,
            "capacity_ah": capacity_grid,
            "interpolated_voltage_v": (
                interpolated_voltage
            ),
            "smoothed_voltage_v": (
                smoothed_voltage
            ),
            "dva_v_per_ah": (
                clipped_differential_voltage
            ),
        }
    )

    cycle_summary = {
        "cycle": cycle_number,
        "status": "Accepted",
        "reason": "DVA calculated successfully.",
        "original_cycle_records": int(
            len(cycle_data)
        ),
        "unique_capacity_points": int(
            len(prepared_cycle)
        ),
        "minimum_capacity_ah": minimum_capacity,
        "maximum_capacity_ah": maximum_capacity,
        "capacity_range_ah": float(
            maximum_capacity - minimum_capacity
        ),
        "minimum_voltage_v": float(
            voltage_values.min()
        ),
        "maximum_voltage_v": float(
            voltage_values.max()
        ),
        "voltage_range_v": float(
            voltage_values.max()
            - voltage_values.min()
        ),
        "peak_dva_v_per_ah": peak_dva,
        "peak_capacity_ah": peak_capacity,
        "peak_voltage_v": peak_voltage,
    }

    return detailed_result, cycle_summary


# ============================================================
# SELECT REPRESENTATIVE GRAPH CYCLES
# ============================================================

def select_graph_cycles(
    accepted_cycles,
    maximum_cycles=MAXIMUM_GRAPH_CYCLES,
):
    """
    Select evenly distributed cycles for the graph.
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
# RUN DVA FOR ALL CYCLES
# ============================================================

def analyze_dva(dataframe):
    """
    Calculate DVA for every valid discharge cycle.
    """

    validated_data = validate_dva_dataset(
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
            calculate_cycle_dva(
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

    cycle_summary_table = pd.DataFrame(
        summary_records
    )

    if not detailed_results:
        raise ValueError(
            "DVA could not be calculated for any cycle. "
            "Check the voltage and capacity measurements."
        )

    combined_results = pd.concat(
        detailed_results,
        ignore_index=True,
    )

    accepted_cycles = (
        cycle_summary_table.loc[
            cycle_summary_table["status"] == "Accepted",
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
                cycle_summary_table["status"]
                == "Rejected"
            ).sum()
        ),
        "graph_cycles": graph_cycles,
    }

    return (
        combined_results,
        cycle_summary_table,
        analysis_summary,
        discharge_information,
    )


# ============================================================
# CREATE DVA GRAPH
# ============================================================

def create_dva_graph(
    detailed_results,
    graph_cycles,
):
    """
    Create a DVA-versus-capacity graph.
    """

    figure, axis = plt.subplots(
        figsize=(13, 8),
    )

    color_map = plt.get_cmap(
        "plasma"
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
            cycle_result["capacity_ah"],
            cycle_result["dva_v_per_ah"],
            linewidth=2,
            color=color_map(color_position),
            label=f"Cycle {cycle_number:g}",
        )

    axis.set_title(
        "Differential Voltage Analysis (DVA)",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    axis.set_xlabel(
        "Discharged Capacity (Ah)",
        fontsize=12,
        fontweight="bold",
    )

    axis.set_ylabel(
        "Differential Voltage |dV/dQ| (V/Ah)",
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
# SAVE DVA RESULTS
# ============================================================

def save_dva_results(
    detailed_results,
    cycle_summary,
    analysis_summary,
    output_directory,
):
    """
    Save DVA results, summaries, and graph.
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
        / "dva_detailed_results.csv"
    )

    cycle_summary_path = (
        output_directory
        / "dva_cycle_summary.csv"
    )

    analysis_summary_path = (
        output_directory
        / "dva_analysis_summary.csv"
    )

    graph_path = (
        output_directory
        / "dva_analysis.png"
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

    figure = create_dva_graph(
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

    try:
        if os.name == "nt":
            os.startfile(
                str(graph_path)
            )
        else:
            print(
                "\nAutomatic graph opening is only "
                "enabled for Windows."
            )

    except Exception as open_error:
        print(
            "\nThe DVA graph was saved successfully, "
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
# PRINT DVA REPORT
# ============================================================

def print_dva_report(
    analysis_summary,
    discharge_information,
    cycle_summary,
    saved_files,
    unknown_columns,
):
    """
    Print a readable DVA analysis report.
    """

    print("\n" + "=" * 75)
    print("DIFFERENTIAL VOLTAGE ANALYSIS")
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
        peak_index = accepted_summary[
            "peak_dva_v_per_ah"
        ].idxmax()

        strongest_peak = accepted_summary.loc[
            peak_index
        ]

        print("\nDVA PEAK INFORMATION")

        print(
            f"  Strongest calculated peak cycle: "
            f"{strongest_peak['cycle']:g}"
        )

        print(
            f"  Peak capacity: "
            f"{strongest_peak['peak_capacity_ah']:.4f} Ah"
        )

        print(
            f"  Voltage at peak: "
            f"{strongest_peak['peak_voltage_v']:.4f} V"
        )

        print(
            f"  Peak DVA magnitude: "
            f"{strongest_peak['peak_dva_v_per_ah']:.4f} V/Ah"
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
        "  DVA peaks are sensitive to measurement noise, "
        "capacity resolution, interpolation, and smoothing."
    )

    print(
        "  Peak interpretation should be validated against "
        "battery chemistry and controlled laboratory data."
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
            dva_results,
            dva_cycle_summary,
            dva_summary,
            discharge_info,
        ) = analyze_dva(
            standardized_data
        )

        output_folder = (
            PLATFORM_FOLDER
            / "outputs"
            / Path(dataset_path).stem
            / "dva_analysis"
        )

        saved_results = save_dva_results(
            dva_results,
            dva_cycle_summary,
            dva_summary,
            output_folder,
        )

        print_dva_report(
            dva_summary,
            discharge_info,
            dva_cycle_summary,
            saved_results,
            unknown_columns,
        )

    except Exception as error:
        print(
            f"\nDVA analysis failed: {error}"
        )