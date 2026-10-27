from pathlib import Path
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
# VALIDATE SOC DATASET
# ============================================================

def validate_soc_dataset(dataframe):
    """
    Confirm that the standardized dataset contains the
    signals required for capacity-based SOC estimation.
    """

    required_columns = [
        "time_s",
        "current_a",
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
            "SOC estimation cannot continue. Missing columns: "
            + ", ".join(missing_columns)
        )

    validated_data = dataframe.copy()

    numeric_columns = [
        "time_s",
        "current_a",
        "voltage_v",
        "capacity_ah",
    ]

    if "temperature_c" in validated_data.columns:
        numeric_columns.append(
            "temperature_c"
        )

    if "cycle" in validated_data.columns:
        numeric_columns.append(
            "cycle"
        )

    for column in numeric_columns:
        validated_data[column] = pd.to_numeric(
            validated_data[column],
            errors="coerce",
        )

    validated_data = validated_data.dropna(
        subset=[
            "time_s",
            "current_a",
            "voltage_v",
            "capacity_ah",
        ]
    )

    validated_data = validated_data[
        validated_data["time_s"] >= 0
    ]

    validated_data = validated_data[
        validated_data["capacity_ah"] >= 0
    ]

    if validated_data.empty:
        raise ValueError(
            "No valid SOC records remain after cleaning."
        )

    if "cycle" not in validated_data.columns:
        validated_data["cycle"] = 1

    validated_data = validated_data.sort_values(
        by=["cycle", "time_s"],
        kind="stable",
    ).reset_index(drop=True)

    return validated_data


# ============================================================
# CALCULATE COULOMB-COUNTED DISCHARGE
# ============================================================

def calculate_coulomb_counting(dataframe):
    """
    Integrate current over time separately for every cycle.

    Negative current represents discharge in this dataset.

    Discharged charge increment:

        delta_Ah = max(-current, 0) * delta_time / 3600
    """

    result_groups = []

    for cycle_number, cycle_data in dataframe.groupby(
        "cycle",
        sort=True,
    ):
        cycle_data = cycle_data.copy()

        cycle_data = cycle_data.sort_values(
            "time_s",
            kind="stable",
        ).reset_index(drop=True)

        cycle_data["delta_time_s"] = (
            cycle_data["time_s"]
            .diff()
            .fillna(0)
        )

        # Prevent invalid integration if timestamps repeat
        # or move backwards.
        cycle_data.loc[
            cycle_data["delta_time_s"] < 0,
            "delta_time_s",
        ] = 0

        previous_current = (
            cycle_data["current_a"]
            .shift(1)
            .fillna(cycle_data["current_a"])
        )

        average_current = (
            cycle_data["current_a"]
            + previous_current
        ) / 2

        # Negative current is discharge.
        discharge_current = (
            -average_current
        ).clip(lower=0)

        cycle_data["delta_discharge_ah"] = (
            discharge_current
            * cycle_data["delta_time_s"]
            / 3600
        )

        cycle_data[
            "coulomb_counted_discharge_ah"
        ] = (
            cycle_data["delta_discharge_ah"]
            .cumsum()
        )

        result_groups.append(
            cycle_data
        )

    return pd.concat(
        result_groups,
        ignore_index=True,
    )


# ============================================================
# ESTIMATE SOC
# ============================================================

def estimate_soc(dataframe):
    """
    Estimate SOC separately for every discharge cycle.

    The maximum measured cumulative discharge capacity
    in each cycle is used as that cycle's usable capacity.

    SOC (%) =
        100 * (1 - discharged_capacity / usable_capacity)
    """

    soc_groups = []
    cycle_summary_rows = []

    for cycle_number, cycle_data in dataframe.groupby(
        "cycle",
        sort=True,
    ):
        cycle_data = cycle_data.copy()

        cycle_data = cycle_data.sort_values(
            "time_s",
            kind="stable",
        ).reset_index(drop=True)

        measured_usable_capacity = (
            cycle_data["capacity_ah"].max()
        )

        integrated_usable_capacity = (
            cycle_data[
                "coulomb_counted_discharge_ah"
            ].max()
        )

        if measured_usable_capacity <= 0:
            raise ValueError(
                f"Cycle {cycle_number} has no valid "
                "measured discharge capacity."
            )

        # SOC derived from the dataset's cumulative
        # discharged-capacity signal.
        cycle_data["soc_percent"] = (
            1
            - cycle_data["capacity_ah"]
            / measured_usable_capacity
        ) * 100

        cycle_data["soc_percent"] = (
            cycle_data["soc_percent"]
            .clip(lower=0, upper=100)
        )

        # Independent SOC estimate from current integration.
        if integrated_usable_capacity > 0:
            cycle_data[
                "coulomb_counting_soc_percent"
            ] = (
                1
                - cycle_data[
                    "coulomb_counted_discharge_ah"
                ]
                / integrated_usable_capacity
            ) * 100

            cycle_data[
                "coulomb_counting_soc_percent"
            ] = (
                cycle_data[
                    "coulomb_counting_soc_percent"
                ]
                .clip(lower=0, upper=100)
            )

            cycle_data[
                "soc_difference_percent"
            ] = (
                cycle_data["soc_percent"]
                - cycle_data[
                    "coulomb_counting_soc_percent"
                ]
            ).abs()

            mean_soc_difference = (
                cycle_data[
                    "soc_difference_percent"
                ].mean()
            )

        else:
            cycle_data[
                "coulomb_counting_soc_percent"
            ] = np.nan

            cycle_data[
                "soc_difference_percent"
            ] = np.nan

            mean_soc_difference = np.nan

        cycle_data["usable_capacity_ah"] = (
            measured_usable_capacity
        )

        cycle_data["soc_category"] = pd.cut(
            cycle_data["soc_percent"],
            bins=[
                -0.01,
                20,
                50,
                80,
                100.01,
            ],
            labels=[
                "Critical",
                "Low",
                "Medium",
                "High",
            ],
            include_lowest=True,
        )

        cycle_summary_rows.append(
            {
                "cycle": cycle_number,
                "records": len(cycle_data),
                "duration_s": (
                    cycle_data["time_s"].max()
                    - cycle_data["time_s"].min()
                ),
                "starting_voltage_v": (
                    cycle_data["voltage_v"].iloc[0]
                ),
                "ending_voltage_v": (
                    cycle_data["voltage_v"].iloc[-1]
                ),
                "measured_usable_capacity_ah": (
                    measured_usable_capacity
                ),
                "integrated_capacity_ah": (
                    integrated_usable_capacity
                ),
                "mean_soc_difference_percent": (
                    mean_soc_difference
                ),
                "minimum_soc_percent": (
                    cycle_data["soc_percent"].min()
                ),
                "maximum_temperature_c": (
                    cycle_data["temperature_c"].max()
                    if "temperature_c"
                    in cycle_data.columns
                    else np.nan
                ),
            }
        )

        soc_groups.append(
            cycle_data
        )

    soc_results = pd.concat(
        soc_groups,
        ignore_index=True,
    )

    cycle_summary = pd.DataFrame(
        cycle_summary_rows
    )

    numeric_columns = cycle_summary.select_dtypes(
        include="number"
    ).columns

    cycle_summary[numeric_columns] = (
        cycle_summary[numeric_columns].round(4)
    )

    return soc_results, cycle_summary


# ============================================================
# CREATE SOC GRAPH
# ============================================================

def create_soc_graph(soc_results):
    """
    Create a readable SOC graph using representative cycles.

    Showing every cycle on one graph would be overcrowded,
    so the first, middle, and last cycles are selected.
    """

    available_cycles = sorted(
        soc_results["cycle"]
        .dropna()
        .unique()
        .tolist()
    )

    if not available_cycles:
        raise ValueError(
            "No valid cycles are available for plotting."
        )

    selected_cycles = [
        available_cycles[0],
        available_cycles[
            len(available_cycles) // 2
        ],
        available_cycles[-1],
    ]

    # Remove duplicates when the dataset contains
    # fewer than three cycles.
    selected_cycles = list(
        dict.fromkeys(selected_cycles)
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(13, 10),
        sharex=False,
    )

    soc_axis = axes[0]
    voltage_axis = axes[1]

    colors = [
        "#0072B2",
        "#E69F00",
        "#009E73",
    ]

    for color, cycle_number in zip(
        colors,
        selected_cycles,
    ):
        cycle_data = soc_results[
            soc_results["cycle"]
            == cycle_number
        ].copy()

        cycle_data = cycle_data.sort_values(
            "time_s"
        )

        soc_axis.plot(
            cycle_data["time_s"],
            cycle_data["soc_percent"],
            color=color,
            linewidth=2,
            label=f"Cycle {cycle_number}",
        )

        voltage_axis.plot(
            cycle_data["soc_percent"],
            cycle_data["voltage_v"],
            color=color,
            linewidth=2,
            label=f"Cycle {cycle_number}",
        )

    soc_axis.axhline(
        20,
        color="#D62728",
        linestyle="--",
        linewidth=1.5,
        label="20% low-SOC threshold",
    )

    soc_axis.set_title(
        "Estimated State of Charge During Discharge",
        fontsize=15,
        fontweight="bold",
    )

    soc_axis.set_xlabel(
        "Time (seconds)"
    )

    soc_axis.set_ylabel(
        "SOC (%)"
    )

    soc_axis.set_ylim(
        0,
        105,
    )

    soc_axis.grid(
        True,
        alpha=0.3,
        linestyle="--",
    )

    soc_axis.legend(
        fontsize=9
    )

    voltage_axis.axvline(
        20,
        color="#D62728",
        linestyle="--",
        linewidth=1.5,
        label="20% SOC",
    )

    voltage_axis.set_title(
        "Voltage versus Estimated SOC",
        fontsize=15,
        fontweight="bold",
    )

    voltage_axis.set_xlabel(
        "SOC (%)"
    )

    voltage_axis.set_ylabel(
        "Voltage (V)"
    )

    voltage_axis.invert_xaxis()

    voltage_axis.grid(
        True,
        alpha=0.3,
        linestyle="--",
    )

    voltage_axis.legend(
        fontsize=9
    )

    figure.tight_layout()

    return figure


# ============================================================
# SAVE SOC RESULTS
# ============================================================

def save_soc_results(
    soc_results,
    cycle_summary,
    output_directory,
):
    """
    Save detailed SOC results, cycle summary, and graph.
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
        / "soc_detailed_results.csv"
    )

    summary_path = (
        output_directory
        / "soc_cycle_summary.csv"
    )

    graph_path = (
        output_directory
        / "soc_analysis.png"
    )

    soc_results.to_csv(
        detailed_path,
        index=False,
    )

    cycle_summary.to_csv(
        summary_path,
        index=False,
    )

    figure = create_soc_graph(
        soc_results
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

    return {
        "detailed_results": detailed_path,
        "cycle_summary": summary_path,
        "graph": graph_path,
    }


# ============================================================
# PRINT SOC REPORT
# ============================================================

def print_soc_report(
    soc_results,
    cycle_summary,
    saved_files,
):
    """
    Print a readable SOC estimation report.
    """

    print("\n" + "=" * 75)
    print("COULOMB-COUNTING BATTERY SOC ESTIMATION")
    print("=" * 75)

    print(
        "\nSOC method:"
        "\n  Capacity-based SOC with independent "
        "current-integration validation"
    )

    print(
        "\nCurrent convention:"
        "\n  Negative current represents battery discharge"
    )

    print(
        f"\nCycles analyzed: "
        f"{cycle_summary['cycle'].nunique()}"
    )

    print(
        f"Records analyzed: "
        f"{len(soc_results)}"
    )

    print("\nCYCLE SUMMARY")

    print(
        cycle_summary.to_string(
            index=False
        )
    )

    print("\nSAVED RESULTS")

    for result_name, result_path in (
        saved_files.items()
    ):
        print(
            f"  {result_name}: {result_path}"
        )

    print("\nImportant:")
    print(
        "  SOC is estimated within each discharge cycle."
    )

    print(
        "  Each cycle is assumed to begin at approximately "
        "100% SOC."
    )

    print(
        "  The maximum cumulative discharged capacity in "
        "each cycle is treated as that cycle's usable capacity."
    )

    print("=" * 75)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":
    dataset_path = input(
        "Paste the full path of an SOC-compatible dataset: "
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

        validated_data = validate_soc_dataset(
            standardized_data
        )

        integrated_data = calculate_coulomb_counting(
            validated_data
        )

        soc_results, cycle_summary = estimate_soc(
            integrated_data
        )

        output_folder = (
            PLATFORM_FOLDER
            / "outputs"
            / Path(dataset_path).stem
            / "soc_analysis"
        )

        saved_results = save_soc_results(
            soc_results,
            cycle_summary,
            output_folder,
        )

        print_soc_report(
            soc_results,
            cycle_summary,
            saved_results,
        )

        if unknown_columns:
            print("\nUnrecognized original columns:")

            for column in unknown_columns:
                print(f"  - {column}")

    except Exception as error:
        print(
            f"\nSOC estimation failed: {error}"
        )