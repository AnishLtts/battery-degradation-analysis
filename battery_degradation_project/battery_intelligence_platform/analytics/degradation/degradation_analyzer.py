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
# VALIDATE DATASET
# ============================================================

def validate_degradation_dataset(dataframe):
    """
    Validate and clean a dataset for cycle-based
    battery degradation analysis.
    """

    required_columns = [
        "cycle",
        "capacity_ah",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Degradation analysis requires: "
            + ", ".join(missing_columns)
        )

    cleaned_data = dataframe.copy()

    cleaned_data["cycle"] = pd.to_numeric(
        cleaned_data["cycle"],
        errors="coerce",
    )

    cleaned_data["capacity_ah"] = pd.to_numeric(
        cleaned_data["capacity_ah"],
        errors="coerce",
    )

    cleaned_data = cleaned_data.dropna(
        subset=["cycle", "capacity_ah"]
    )

    cleaned_data = cleaned_data[
        cleaned_data["capacity_ah"] >= 0
    ]

    if cleaned_data.empty:
        raise ValueError(
            "No valid cycle and capacity records were found."
        )

    cleaned_data = cleaned_data.sort_values(
        by=["cycle"],
        kind="stable",
    ).reset_index(drop=True)

    return cleaned_data


# ============================================================
# CREATE ONE CAPACITY VALUE PER CYCLE
# ============================================================

def create_cycle_capacity_table(dataframe):
    """
    Extract one usable-capacity value for each cycle.

    For cumulative discharge-capacity datasets, the maximum
    capacity in each cycle represents the usable capacity.
    """

    cycle_capacity = (
        dataframe.groupby(
            "cycle",
            as_index=False,
        )["capacity_ah"]
        .max()
    )

    cycle_capacity = cycle_capacity.rename(
        columns={
            "capacity_ah": "usable_capacity_ah"
        }
    )

    cycle_capacity = cycle_capacity[
        cycle_capacity["usable_capacity_ah"] > 0
    ].copy()

    cycle_capacity = cycle_capacity.sort_values(
        "cycle"
    ).reset_index(drop=True)

    if len(cycle_capacity) < 2:
        raise ValueError(
            "At least two valid cycles are required "
            "for degradation analysis."
        )

    return cycle_capacity


# ============================================================
# CALCULATE DEGRADATION METRICS
# ============================================================

def calculate_degradation_metrics(cycle_capacity):
    """
    Calculate SOH, capacity retention, capacity loss,
    degradation rate, and end-of-life crossing.
    """

    results = cycle_capacity.copy()

    reference_count = min(
        5,
        len(results),
    )

    reference_capacity = (
        results["usable_capacity_ah"]
        .head(reference_count)
        .median()
    )

    if reference_capacity <= 0:
        raise ValueError(
            "Reference capacity must be greater than zero."
        )

    results["reference_capacity_ah"] = (
        reference_capacity
    )

    results["capacity_retention_percent"] = (
        results["usable_capacity_ah"]
        / reference_capacity
        * 100
    )

    results["soh_percent"] = (
        results["capacity_retention_percent"]
    )

    results["capacity_loss_percent"] = (
        100
        - results["capacity_retention_percent"]
    )

    results["capacity_change_ah"] = (
        results["usable_capacity_ah"].diff()
    )

    results["capacity_change_percent"] = (
        results["usable_capacity_ah"]
        .pct_change()
        * 100
    )

    results["rolling_capacity_ah"] = (
        results["usable_capacity_ah"]
        .rolling(
            window=5,
            min_periods=1,
            center=True,
        )
        .median()
    )

    results["health_category"] = pd.cut(
        results["soh_percent"],
        bins=[
            -np.inf,
            70,
            80,
            90,
            np.inf,
        ],
        labels=[
            "Severely Degraded",
            "Degraded",
            "Moderately Aged",
            "Healthy",
        ],
        right=False,
    )

    cycle_values = results["cycle"].to_numpy(
        dtype=float
    )

    capacity_values = results[
        "usable_capacity_ah"
    ].to_numpy(
        dtype=float
    )

    slope, intercept = np.polyfit(
        cycle_values,
        capacity_values,
        1,
    )

    results["linear_capacity_prediction_ah"] = (
        slope * results["cycle"] + intercept
    )

    degradation_rate_ah_per_cycle = slope

    degradation_rate_percent_per_cycle = (
        slope
        / reference_capacity
        * 100
    )

    eol_capacity_ah = (
        0.80 * reference_capacity
    )

    below_eol = results[
        results["soh_percent"] <= 80
    ]

    if below_eol.empty:
        observed_eol_cycle = None
    else:
        observed_eol_cycle = int(
            below_eol.iloc[0]["cycle"]
        )

    if slope < 0:
        predicted_eol_cycle = (
            eol_capacity_ah - intercept
        ) / slope

        predicted_eol_cycle = max(
            0,
            float(predicted_eol_cycle),
        )
    else:
        predicted_eol_cycle = None

    summary = {
        "cycles_analyzed": int(
            results["cycle"].nunique()
        ),
        "reference_capacity_ah": float(
            reference_capacity
        ),
        "initial_capacity_ah": float(
            results["usable_capacity_ah"].iloc[0]
        ),
        "latest_capacity_ah": float(
            results["usable_capacity_ah"].iloc[-1]
        ),
        "latest_soh_percent": float(
            results["soh_percent"].iloc[-1]
        ),
        "total_capacity_loss_percent": float(
            results["capacity_loss_percent"].iloc[-1]
        ),
        "degradation_rate_ah_per_cycle": float(
            degradation_rate_ah_per_cycle
        ),
        "degradation_rate_percent_per_cycle": float(
            degradation_rate_percent_per_cycle
        ),
        "observed_80_percent_eol_cycle": (
            observed_eol_cycle
        ),
        "predicted_80_percent_eol_cycle": (
            predicted_eol_cycle
        ),
    }

    return results, summary


# ============================================================
# CREATE DEGRADATION GRAPH
# ============================================================

def create_degradation_graph(
    degradation_results,
    summary,
):
    """
    Create capacity and SOH degradation visualizations.

    Both graphs display Cycle Number on the X-axis.
    """

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(13, 10),
        sharex=True,
    )

    capacity_axis = axes[0]
    soh_axis = axes[1]

    # --------------------------------------------------------
    # FIRST GRAPH: BATTERY CAPACITY DEGRADATION
    # --------------------------------------------------------

    capacity_axis.scatter(
        degradation_results["cycle"],
        degradation_results["usable_capacity_ah"],
        color="#0072B2",
        s=22,
        alpha=0.65,
        label="Measured usable capacity",
    )

    capacity_axis.plot(
        degradation_results["cycle"],
        degradation_results["rolling_capacity_ah"],
        color="#E69F00",
        linewidth=2.2,
        label="5-cycle rolling median",
    )

    capacity_axis.plot(
        degradation_results["cycle"],
        degradation_results[
            "linear_capacity_prediction_ah"
        ],
        color="#D55E00",
        linestyle="--",
        linewidth=1.8,
        label="Linear degradation trend",
    )

    eol_capacity = (
        summary["reference_capacity_ah"]
        * 0.80
    )

    capacity_axis.axhline(
        eol_capacity,
        color="#CC0000",
        linestyle=":",
        linewidth=2,
        label="80% end-of-life capacity",
    )

    capacity_axis.set_title(
        "Battery Capacity Degradation",
        fontsize=15,
        fontweight="bold",
    )

    # X-axis title for the first graph
    capacity_axis.set_xlabel(
        "Cycle Number",
        fontsize=11,
        fontweight="bold",
        labelpad=8,
    )

    capacity_axis.set_ylabel(
        "Usable Capacity (Ah)",
        fontsize=11,
        fontweight="bold",
    )

    # Display cycle-number tick values under the first graph.
    # sharex=True normally hides these values.
    capacity_axis.tick_params(
        axis="x",
        which="both",
        labelbottom=True,
    )

    capacity_axis.grid(
        True,
        alpha=0.3,
        linestyle="--",
    )

    capacity_axis.legend(
        fontsize=9,
    )

    # --------------------------------------------------------
    # SECOND GRAPH: STATE OF HEALTH
    # --------------------------------------------------------

    soh_axis.plot(
        degradation_results["cycle"],
        degradation_results["soh_percent"],
        color="#009E73",
        linewidth=2,
        label="Cycle SOH",
    )

    soh_axis.axhline(
        90,
        color="#2E8B57",
        linestyle="--",
        linewidth=1.2,
        label="90% healthy boundary",
    )

    soh_axis.axhline(
        80,
        color="#E69F00",
        linestyle="--",
        linewidth=1.5,
        label="80% EOL threshold",
    )

    soh_axis.axhline(
        70,
        color="#D62728",
        linestyle="--",
        linewidth=1.2,
        label="70% severe boundary",
    )

    observed_eol = summary[
        "observed_80_percent_eol_cycle"
    ]

    if observed_eol is not None:
        soh_axis.axvline(
            observed_eol,
            color="#CC0000",
            linestyle=":",
            linewidth=2,
            label=(
                f"Observed 80% EOL: cycle "
                f"{observed_eol}"
            ),
        )

    soh_axis.set_title(
        "Cycle-Level State of Health",
        fontsize=15,
        fontweight="bold",
    )

    soh_axis.set_xlabel(
        "Cycle Number",
        fontsize=11,
        fontweight="bold",
        labelpad=8,
    )

    soh_axis.set_ylabel(
        "SOH (%)",
        fontsize=11,
        fontweight="bold",
    )

    soh_axis.grid(
        True,
        alpha=0.3,
        linestyle="--",
    )

    soh_axis.legend(
        fontsize=9,
        ncol=2,
    )

    # Increase space so the first X-axis title remains visible.
    figure.subplots_adjust(
        hspace=0.32,
    )

    figure.tight_layout()

    return figure


# ============================================================
# SAVE RESULTS
# ============================================================
def save_degradation_results(
    degradation_results,
    summary,
    output_directory,
):
    """
    Save the cycle results, summary, and degradation graph.
    Automatically open the saved graph after creating it.
    """

    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    detailed_path = (
        output_directory
        / "degradation_cycle_results.csv"
    )

    summary_path = (
        output_directory
        / "degradation_summary.csv"
    )

    graph_path = (
        output_directory
        / "degradation_analysis.png"
    )

    degradation_results.to_csv(
        detailed_path,
        index=False,
    )

    summary_table = pd.DataFrame([summary])

    summary_table.to_csv(
        summary_path,
        index=False,
    )

    figure = create_degradation_graph(
        degradation_results,
        summary,
    )

    figure.savefig(
        graph_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)

    # Automatically open the saved graph on Windows.
    try:
        os.startfile(str(graph_path))
    except Exception as open_error:
        print(
            "\nThe graph was saved successfully, "
            "but it could not be opened automatically."
        )
        print(f"Open the graph manually from: {graph_path}")
        print(f"Opening error: {open_error}")

    return {
        "cycle_results": detailed_path,
        "summary": summary_path,
        "graph": graph_path,
    }

# ============================================================
# PRINT REPORT
# ============================================================

def format_cycle_value(value):
    """
    Format EOL values for terminal output.
    """

    if value is None:
        return "Not reached"

    return f"{value:.1f}"


def print_degradation_report(
    summary,
    saved_files,
):
    """
    Print the degradation-analysis results.
    """

    print("\n" + "=" * 75)
    print("BATTERY DEGRADATION ANALYSIS")
    print("=" * 75)

    print(
        f"\nCycles analyzed: "
        f"{summary['cycles_analyzed']}"
    )

    print(
        f"Reference capacity: "
        f"{summary['reference_capacity_ah']:.4f} Ah"
    )

    print(
        f"Initial capacity: "
        f"{summary['initial_capacity_ah']:.4f} Ah"
    )

    print(
        f"Latest capacity: "
        f"{summary['latest_capacity_ah']:.4f} Ah"
    )

    print(
        f"Latest SOH: "
        f"{summary['latest_soh_percent']:.2f}%"
    )

    print(
        f"Total capacity loss: "
        f"{summary['total_capacity_loss_percent']:.2f}%"
    )

    print(
        "\nEstimated degradation rate:"
    )

    print(
        f"  {summary['degradation_rate_ah_per_cycle']:.6f} "
        "Ah per cycle"
    )

    print(
        f"  {summary['degradation_rate_percent_per_cycle']:.4f}% "
        "per cycle"
    )

    print(
        "\n80% end-of-life threshold:"
    )

    print(
        "  Observed crossing cycle: "
        + format_cycle_value(
            summary[
                "observed_80_percent_eol_cycle"
            ]
        )
    )

    print(
        "  Linear predicted crossing cycle: "
        + format_cycle_value(
            summary[
                "predicted_80_percent_eol_cycle"
            ]
        )
    )

    print("\nSAVED RESULTS")

    for result_name, result_path in saved_files.items():
        print(
            f"  {result_name}: {result_path}"
        )

    print("\nImportant:")

    print(
        "  The predicted EOL cycle uses a simple linear "
        "trend and should be treated as an analytical "
        "estimate, not a production-grade RUL prediction."
    )

    print("=" * 75)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":
    dataset_path = input(
        "Paste the full path of a cycle-capacity dataset: "
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

        validated_data = validate_degradation_dataset(
            standardized_data
        )

        cycle_capacity = create_cycle_capacity_table(
            validated_data
        )

        degradation_results, degradation_summary = (
            calculate_degradation_metrics(
                cycle_capacity
            )
        )

        output_folder = (
            PLATFORM_FOLDER
            / "outputs"
            / Path(dataset_path).stem
            / "degradation_analysis"
        )

        saved_results = save_degradation_results(
            degradation_results,
            degradation_summary,
            output_folder,
        )

        print_degradation_report(
            degradation_summary,
            saved_results,
        )

        if unknown_columns:
            print("\nUnrecognized original columns:")

            for column in unknown_columns:
                print(f"  - {column}")

    except Exception as error:
        print(
            f"\nDegradation analysis failed: {error}"
        )