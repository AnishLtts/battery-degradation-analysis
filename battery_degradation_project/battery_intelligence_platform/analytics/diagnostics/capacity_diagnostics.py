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
# SETTINGS
# ============================================================

# Diagnostic defaults
END_OF_LIFE_RETENTION_PERCENT = 80.0
ABNORMAL_DROP_PERCENT = 2.0


# ============================================================
# VALIDATION
# ============================================================

def validate_capacity_dataset(dataframe):
    """
    Validate that a column-mapped dataframe has what capacity
    diagnostics needs.
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
            "Capacity diagnostics requires these columns: "
            + ", ".join(missing_columns)
        )

    return dataframe.copy()


def prepare_data(dataframe):
    prepared = dataframe.copy()

    numeric_columns = [
        "cycle",
        "capacity_ah",
        "time_s",
        "voltage_v",
        "current_a",
    ]

    for column in numeric_columns:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(
                prepared[column],
                errors="coerce",
            )

    prepared = prepared.dropna(
        subset=["cycle", "capacity_ah"]
    ).copy()

    prepared = prepared[
        prepared["capacity_ah"] >= 0
    ].copy()

    if prepared.empty:
        raise ValueError(
            "No valid cycle and capacity values were found."
        )

    prepared = prepared.sort_values(
        ["cycle"],
        kind="stable",
    ).reset_index(drop=True)

    return prepared


# ============================================================
# CAPACITY ANALYSIS
# ============================================================

def safe_slope(x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    valid_mask = (
        np.isfinite(x_values)
        & np.isfinite(y_values)
    )

    x_values = x_values[valid_mask]
    y_values = y_values[valid_mask]

    if len(x_values) < 2:
        return np.nan

    if np.ptp(x_values) == 0:
        return np.nan

    return float(
        np.polyfit(x_values, y_values, 1)[0]
    )


def extract_cycle_capacity(dataframe):
    """
    Extract one representative capacity value for each cycle.

    The maximum capacity value is used because row-level battery
    datasets commonly store cumulative capacity within each cycle.
    """
    cycle_summary = (
        dataframe.groupby("cycle", as_index=False)
        .agg(
            capacity_ah=("capacity_ah", "max"),
            minimum_capacity_ah=("capacity_ah", "min"),
            average_capacity_ah=("capacity_ah", "mean"),
            record_count=("capacity_ah", "size"),
        )
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    if cycle_summary.empty:
        raise ValueError(
            "Cycle-level capacity values could not be calculated."
        )

    return cycle_summary


def calculate_capacity_metrics(cycle_summary):
    results = cycle_summary.copy()

    initial_capacity = results[
        "capacity_ah"
    ].iloc[0]

    if initial_capacity <= 0:
        positive_capacities = results.loc[
            results["capacity_ah"] > 0,
            "capacity_ah",
        ]

        if positive_capacities.empty:
            raise ValueError(
                "A positive initial capacity could not be found."
            )

        initial_capacity = positive_capacities.iloc[0]

    results["capacity_retention_percent"] = (
        results["capacity_ah"]
        / initial_capacity
        * 100
    )

    results["capacity_loss_ah"] = (
        initial_capacity
        - results["capacity_ah"]
    )

    results["capacity_loss_percent"] = (
        100
        - results["capacity_retention_percent"]
    )

    results["cycle_capacity_change_ah"] = (
        results["capacity_ah"].diff()
    )

    results["cycle_capacity_change_percent"] = (
        results["capacity_ah"].pct_change()
        * 100
    )

    results["capacity_drop_percent"] = (
        -results["cycle_capacity_change_percent"]
    )

    results["abnormal_drop_flag"] = (
        results["capacity_drop_percent"]
        > ABNORMAL_DROP_PERCENT
    )

    results["below_80_percent_flag"] = (
        results["capacity_retention_percent"]
        <= END_OF_LIFE_RETENTION_PERCENT
    )

    rolling_window = min(
        10,
        max(2, len(results)),
    )

    results["rolling_capacity_ah"] = (
        results["capacity_ah"]
        .rolling(
            window=rolling_window,
            min_periods=1,
        )
        .mean()
    )

    return results, initial_capacity


def build_analysis_summary(
    cycle_results,
    initial_capacity,
):
    final_capacity = cycle_results[
        "capacity_ah"
    ].iloc[-1]

    final_retention = cycle_results[
        "capacity_retention_percent"
    ].iloc[-1]

    total_capacity_loss = (
        initial_capacity - final_capacity
    )

    total_capacity_loss_percent = (
        100 - final_retention
    )

    capacity_slope = safe_slope(
        cycle_results["cycle"],
        cycle_results["capacity_ah"],
    )

    retention_slope = safe_slope(
        cycle_results["cycle"],
        cycle_results[
            "capacity_retention_percent"
        ],
    )

    abnormal_drop_count = int(
        cycle_results[
            "abnormal_drop_flag"
        ].sum()
    )

    below_eol = cycle_results[
        cycle_results["below_80_percent_flag"]
    ]

    if below_eol.empty:
        first_eol_cycle = np.nan
    else:
        first_eol_cycle = below_eol[
            "cycle"
        ].iloc[0]

    summary = pd.DataFrame(
        [
            {
                "metric": "Total cycles analyzed",
                "value": len(cycle_results),
                "unit": "cycles",
            },
            {
                "metric": "Initial capacity",
                "value": initial_capacity,
                "unit": "Ah",
            },
            {
                "metric": "Latest capacity",
                "value": final_capacity,
                "unit": "Ah",
            },
            {
                "metric": "Total capacity loss",
                "value": total_capacity_loss,
                "unit": "Ah",
            },
            {
                "metric": "Total capacity loss percentage",
                "value": total_capacity_loss_percent,
                "unit": "%",
            },
            {
                "metric": "Latest capacity retention",
                "value": final_retention,
                "unit": "%",
            },
            {
                "metric": "Capacity degradation rate",
                "value": capacity_slope,
                "unit": "Ah per cycle",
            },
            {
                "metric": "Retention degradation rate",
                "value": retention_slope,
                "unit": "% per cycle",
            },
            {
                "metric": "Abnormal capacity drops",
                "value": abnormal_drop_count,
                "unit": "cycles",
            },
            {
                "metric": "First cycle at or below 80 percent",
                "value": first_eol_cycle,
                "unit": "cycle",
            },
        ]
    )

    return summary


def build_capacity_warnings(cycle_results):
    warnings = []

    below_eol = cycle_results[
        cycle_results["below_80_percent_flag"]
    ]

    abnormal_drops = cycle_results[
        cycle_results["abnormal_drop_flag"]
    ]

    final_retention = cycle_results[
        "capacity_retention_percent"
    ].iloc[-1]

    if not below_eol.empty:
        first_cycle = below_eol["cycle"].iloc[0]

        warnings.append(
            {
                "severity": "High",
                "warning_type": "Low capacity retention",
                "affected_cycles": len(below_eol),
                "first_affected_cycle": first_cycle,
                "message": (
                    "Capacity retention reached or dropped "
                    "below 80 percent."
                ),
            }
        )

    if not abnormal_drops.empty:
        first_cycle = abnormal_drops[
            "cycle"
        ].iloc[0]

        warnings.append(
            {
                "severity": "Medium",
                "warning_type": "Abnormal capacity drop",
                "affected_cycles": len(abnormal_drops),
                "first_affected_cycle": first_cycle,
                "message": (
                    "One or more cycles lost more than "
                    f"{ABNORMAL_DROP_PERCENT:.1f} percent "
                    "capacity compared with the previous cycle."
                ),
            }
        )

    if final_retention > 80:
        warnings.append(
            {
                "severity": "Information",
                "warning_type": "Capacity above default EOL level",
                "affected_cycles": 0,
                "first_affected_cycle": np.nan,
                "message": (
                    "Latest capacity retention remains above "
                    "the default 80 percent end-of-life reference."
                ),
            }
        )

    if not warnings:
        warnings.append(
            {
                "severity": "Information",
                "warning_type": "No major capacity warning",
                "affected_cycles": 0,
                "first_affected_cycle": np.nan,
                "message": (
                    "No major capacity warning was detected "
                    "using the default thresholds."
                ),
            }
        )

    return pd.DataFrame(warnings)


# ============================================================
# GRAPH
# ============================================================

def create_capacity_graph(
    cycle_results,
    output_path,
):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(16, 10),
    )

    figure.suptitle(
        "Battery Capacity Diagnostics",
        fontsize=17,
        fontweight="bold",
    )

    # Graph 1: Capacity across cycles
    axes[0, 0].plot(
        cycle_results["cycle"],
        cycle_results["capacity_ah"],
        color="tab:blue",
        marker="o",
        markersize=3,
        linewidth=1.3,
        label="Measured capacity",
    )

    axes[0, 0].plot(
        cycle_results["cycle"],
        cycle_results["rolling_capacity_ah"],
        color="red",
        linewidth=2,
        label="Smoothed trend",
    )

    axes[0, 0].set_title(
        "Capacity Across Cycles"
    )
    axes[0, 0].set_xlabel("Cycle")
    axes[0, 0].set_ylabel("Capacity (Ah)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Graph 2: Capacity retention
    axes[0, 1].plot(
        cycle_results["cycle"],
        cycle_results[
            "capacity_retention_percent"
        ],
        color="tab:green",
        linewidth=1.5,
    )

    axes[0, 1].axhline(
        END_OF_LIFE_RETENTION_PERCENT,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="80% reference",
    )

    axes[0, 1].set_title(
        "Capacity Retention"
    )
    axes[0, 1].set_xlabel("Cycle")
    axes[0, 1].set_ylabel(
        "Capacity Retention (%)"
    )
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Graph 3: Capacity loss
    axes[1, 0].bar(
        cycle_results["cycle"],
        cycle_results["capacity_loss_percent"],
        color="tab:orange",
        alpha=0.85,
    )

    axes[1, 0].set_title(
        "Accumulated Capacity Loss"
    )
    axes[1, 0].set_xlabel("Cycle")
    axes[1, 0].set_ylabel("Capacity Loss (%)")
    axes[1, 0].grid(
        True,
        axis="y",
        alpha=0.3,
    )

    # Graph 4: Cycle-to-cycle changes
    normal_cycles = cycle_results[
        ~cycle_results["abnormal_drop_flag"]
    ]

    abnormal_cycles = cycle_results[
        cycle_results["abnormal_drop_flag"]
    ]

    axes[1, 1].scatter(
        normal_cycles["cycle"],
        normal_cycles["capacity_drop_percent"],
        color="tab:blue",
        s=18,
        label="Normal change",
    )

    if not abnormal_cycles.empty:
        axes[1, 1].scatter(
            abnormal_cycles["cycle"],
            abnormal_cycles["capacity_drop_percent"],
            color="red",
            s=35,
            label="Abnormal drop",
        )

    axes[1, 1].axhline(
        ABNORMAL_DROP_PERCENT,
        color="red",
        linestyle="--",
        linewidth=1.3,
        label=(
            f"Abnormal drop reference: "
            f"{ABNORMAL_DROP_PERCENT:.0f}%"
        ),
    )

    axes[1, 1].set_title(
        "Cycle-to-Cycle Capacity Drop"
    )
    axes[1, 1].set_xlabel("Cycle")
    axes[1, 1].set_ylabel(
        "Capacity Drop (%)"
    )
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    figure.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# OUTPUT
# ============================================================

def save_capacity_results(
    cycle_results,
    analysis_summary,
    warnings,
    output_directory,
):
    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    cycle_results_path = (
        output_directory
        / "capacity_cycle_results.csv"
    )

    analysis_summary_path = (
        output_directory
        / "capacity_analysis_summary.csv"
    )

    warnings_path = (
        output_directory
        / "capacity_warnings.csv"
    )

    graph_path = (
        output_directory
        / "capacity_diagnostics.png"
    )

    cycle_results.to_csv(
        cycle_results_path,
        index=False,
    )

    analysis_summary.to_csv(
        analysis_summary_path,
        index=False,
    )

    warnings.to_csv(
        warnings_path,
        index=False,
    )

    create_capacity_graph(
        cycle_results,
        graph_path,
    )

    return {
        "cycle_results": cycle_results_path,
        "analysis_summary": analysis_summary_path,
        "warnings": warnings_path,
        "graph": graph_path,
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def analyze_capacity_diagnostics(dataframe):
    """
    Run the capacity diagnostics analysis on an already
    column-mapped battery dataframe.
    """
    validated = validate_capacity_dataset(dataframe)

    prepared = prepare_data(validated)

    cycle_summary = extract_cycle_capacity(
        prepared
    )

    cycle_results, initial_capacity = (
        calculate_capacity_metrics(
            cycle_summary
        )
    )

    analysis_summary = build_analysis_summary(
        cycle_results,
        initial_capacity,
    )

    return prepared, cycle_results, analysis_summary


def main():
    print("\nBattery Capacity Diagnostics")
    print("=" * 40)

    dataset_path = input(
        "Paste the full path of a raw or normalized "
        "battery dataset: "
    ).strip().strip('"').strip("'")

    if not dataset_path:
        print(
            "\nCapacity diagnostics failed: "
            "No dataset path was supplied."
        )
        sys.exit(1)

    try:
        battery_data = load_battery_dataset(dataset_path)

        mapping, unknown_columns = detect_column_mapping(
            battery_data
        )

        standardized_data = apply_column_mapping(
            battery_data,
            mapping,
        )

        (
            prepared,
            cycle_results,
            analysis_summary,
        ) = analyze_capacity_diagnostics(
            standardized_data
        )

        warnings = build_capacity_warnings(
            cycle_results
        )

        output_directory = (
            PLATFORM_FOLDER
            / "outputs"
            / Path(dataset_path).stem
            / "capacity_diagnostics"
        )

        output_files = save_capacity_results(
            cycle_results,
            analysis_summary,
            warnings,
            output_directory,
        )

        print(
            "\nCapacity diagnostics completed successfully."
        )

        print(
            f"\nOutput folder:\n"
            f"{output_directory}"
        )

        print("\nGenerated files:")

        for output_name, output_path in output_files.items():
            print(
                f"  {output_name}: {output_path}"
            )

        print("\nImportant summary:")

        for _, row in analysis_summary.iterrows():
            value = row["value"]

            if isinstance(
                value,
                (float, np.floating),
            ):
                value = (
                    f"{value:.4f}"
                    if np.isfinite(value)
                    else "Not reached"
                )

            print(
                f"  {row['metric']}: "
                f"{value} {row['unit']}"
            )

        print("\nWarnings:")

        for _, row in warnings.iterrows():
            print(
                f"  [{row['severity']}] "
                f"{row['warning_type']}: "
                f"{row['message']}"
            )

        print(
            "\nNote: The 80 percent end-of-life and "
            "2 percent abnormal-drop values are default "
            "diagnostic references. Confirm final limits "
            "using the cell manufacturer's datasheet."
        )

    except Exception as error:
        print(
            f"\nCapacity diagnostics failed: {error}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()