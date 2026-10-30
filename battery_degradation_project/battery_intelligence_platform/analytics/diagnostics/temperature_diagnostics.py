from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# COLUMN NAME DEFINITIONS
# ============================================================

COLUMN_ALIASES = {
    "cycle": [
        "cycle",
        "cycle_index",
        "cycle_number",
        "cycle_no",
        "cycleno",
    ],
    "time_s": [
        "time_s",
        "time",
        "test_time",
        "test_time_s",
        "elapsed_time",
        "elapsed_time_s",
        "seconds",
    ],
    "temperature_c": [
        "temperature_c",
        "temperature",
        "temp_c",
        "temp",
        "cell_temperature_c",
        "battery_temperature_c",
    ],
    "current_a": [
        "current_a",
        "current",
        "current_amp",
        "current_amps",
        "amps",
    ],
    "voltage_v": [
        "voltage_v",
        "voltage",
        "cell_voltage_v",
        "battery_voltage_v",
    ],
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_column_name(column_name):
    """
    Convert a column name into a consistent lowercase format.
    Example:
        Temperature (C) -> temperature_c
    """
    normalized = str(column_name).strip().lower()

    replacements = {
        "°": "",
        "(": "_",
        ")": "",
        "[": "_",
        "]": "",
        "/": "_",
        "\\": "_",
        "-": "_",
        " ": "_",
        ".": "_",
    }

    for old_value, new_value in replacements.items():
        normalized = normalized.replace(old_value, new_value)

    while "__" in normalized:
        normalized = normalized.replace("__", "_")

    return normalized.strip("_")


def standardize_columns(dataframe):
    """
    Normalize dataset column names and map known aliases to the
    standard names used by this diagnostics module.
    """
    dataframe = dataframe.copy()

    dataframe.columns = [
        normalize_column_name(column)
        for column in dataframe.columns
    ]

    rename_map = {}

    for standard_name, aliases in COLUMN_ALIASES.items():
        normalized_aliases = {
            normalize_column_name(alias)
            for alias in aliases
        }

        for column in dataframe.columns:
            if column in normalized_aliases:
                rename_map[column] = standard_name
                break

    return dataframe.rename(columns=rename_map)


def load_dataset(dataset_path):
    """
    Load a CSV dataset after validating its path.
    """
    dataset_path = Path(dataset_path).expanduser()

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset was not found: {dataset_path}"
        )

    if not dataset_path.is_file():
        raise ValueError(
            f"The supplied path is not a file: {dataset_path}"
        )

    if dataset_path.suffix.lower() != ".csv":
        raise ValueError(
            "Temperature diagnostics currently supports CSV files only."
        )

    dataframe = pd.read_csv(dataset_path, low_memory=False)

    if dataframe.empty:
        raise ValueError("The dataset is empty.")

    dataframe = standardize_columns(dataframe)

    if "temperature_c" not in dataframe.columns:
        raise ValueError(
            "Temperature diagnostics requires a temperature column. "
            "Expected names include temperature_c, temp_c, or temperature."
        )

    return dataframe, dataset_path


def prepare_temperature_data(dataframe):
    """
    Clean the columns required for temperature analysis.
    """
    prepared = dataframe.copy()

    numeric_columns = [
        "cycle",
        "time_s",
        "temperature_c",
        "current_a",
        "voltage_v",
    ]

    for column in numeric_columns:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(
                prepared[column],
                errors="coerce",
            )

    prepared = prepared.dropna(
        subset=["temperature_c"]
    ).copy()

    if prepared.empty:
        raise ValueError(
            "No valid numeric temperature values were found."
        )

    if "cycle" not in prepared.columns:
        prepared["cycle"] = 1

    prepared["cycle"] = prepared["cycle"].fillna(0)

    if "time_s" not in prepared.columns:
        prepared["time_s"] = np.arange(
            len(prepared),
            dtype=float,
        )
    else:
        fallback_time = pd.Series(
            np.arange(len(prepared), dtype=float),
            index=prepared.index,
        )

        prepared["time_s"] = prepared["time_s"].fillna(
            fallback_time
        )

    prepared = prepared.sort_values(
        ["cycle", "time_s"],
        kind="stable",
    ).reset_index(drop=True)

    return prepared


def classify_temperature(temperature_c):
    """
    Assign a practical operating category to each temperature value.

    These thresholds are diagnostic defaults, not universal safety limits.
    Actual limits should come from the battery cell manufacturer's datasheet.
    """
    if pd.isna(temperature_c):
        return "Unknown"
    if temperature_c < 0:
        return "Very cold"
    if temperature_c < 10:
        return "Cold"
    if temperature_c <= 40:
        return "Normal"
    if temperature_c <= 50:
        return "Elevated"
    if temperature_c <= 60:
        return "High"
    return "Critical"


def classify_temperature_rise(rise_c):
    """
    Classify the rise between cycle minimum and maximum temperature.
    """
    if pd.isna(rise_c):
        return "Unknown"
    if rise_c <= 5:
        return "Stable"
    if rise_c <= 10:
        return "Moderate rise"
    if rise_c <= 20:
        return "Large rise"
    return "Extreme rise"


def safe_slope(x_values, y_values):
    """
    Calculate a linear slope while safely handling invalid data.
    """
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


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def build_detailed_records(dataframe):
    """
    Create row-level temperature diagnostic information.
    """
    detailed = dataframe.copy()

    detailed["temperature_category"] = detailed[
        "temperature_c"
    ].apply(classify_temperature)

    detailed["temperature_change_c"] = (
        detailed.groupby("cycle")["temperature_c"].diff()
    )

    detailed["temperature_change_rate_c_per_s"] = (
        detailed["temperature_change_c"]
        / detailed.groupby("cycle")["time_s"].diff()
    )

    detailed[
        "temperature_change_rate_c_per_s"
    ] = detailed[
        "temperature_change_rate_c_per_s"
    ].replace([np.inf, -np.inf], np.nan)

    detailed["high_temperature_flag"] = (
        detailed["temperature_c"] > 50
    )

    detailed["critical_temperature_flag"] = (
        detailed["temperature_c"] > 60
    )

    detailed["cold_temperature_flag"] = (
        detailed["temperature_c"] < 0
    )

    detailed["rapid_heating_flag"] = (
        detailed["temperature_change_rate_c_per_s"] > 0.1
    )

    return detailed


def build_cycle_summary(detailed):
    """
    Generate one temperature summary row for every cycle.
    """
    summary_rows = []

    for cycle_value, cycle_data in detailed.groupby(
        "cycle",
        sort=True,
    ):
        cycle_data = cycle_data.sort_values("time_s")

        minimum_temperature = cycle_data[
            "temperature_c"
        ].min()

        maximum_temperature = cycle_data[
            "temperature_c"
        ].max()

        average_temperature = cycle_data[
            "temperature_c"
        ].mean()

        median_temperature = cycle_data[
            "temperature_c"
        ].median()

        temperature_standard_deviation = cycle_data[
            "temperature_c"
        ].std()

        temperature_rise = (
            maximum_temperature - minimum_temperature
        )

        duration_s = (
            cycle_data["time_s"].max()
            - cycle_data["time_s"].min()
        )

        slope_c_per_s = safe_slope(
            cycle_data["time_s"],
            cycle_data["temperature_c"],
        )

        if np.isfinite(slope_c_per_s):
            slope_c_per_min = slope_c_per_s * 60
        else:
            slope_c_per_min = np.nan

        summary_rows.append(
            {
                "cycle": cycle_value,
                "record_count": len(cycle_data),
                "duration_s": duration_s,
                "minimum_temperature_c": minimum_temperature,
                "maximum_temperature_c": maximum_temperature,
                "average_temperature_c": average_temperature,
                "median_temperature_c": median_temperature,
                "temperature_std_c": temperature_standard_deviation,
                "temperature_rise_c": temperature_rise,
                "temperature_rise_category": (
                    classify_temperature_rise(
                        temperature_rise
                    )
                ),
                "temperature_slope_c_per_min": slope_c_per_min,
                "high_temperature_records": int(
                    cycle_data[
                        "high_temperature_flag"
                    ].sum()
                ),
                "critical_temperature_records": int(
                    cycle_data[
                        "critical_temperature_flag"
                    ].sum()
                ),
                "cold_temperature_records": int(
                    cycle_data[
                        "cold_temperature_flag"
                    ].sum()
                ),
                "rapid_heating_records": int(
                    cycle_data[
                        "rapid_heating_flag"
                    ].sum()
                ),
                "maximum_temperature_category": (
                    classify_temperature(
                        maximum_temperature
                    )
                ),
            }
        )

    return pd.DataFrame(summary_rows)


def build_analysis_summary(detailed, cycle_summary):
    """
    Generate a compact overall summary for the complete dataset.
    """
    temperature_series = detailed["temperature_c"]

    hottest_row_index = temperature_series.idxmax()
    coldest_row_index = temperature_series.idxmin()

    hottest_row = detailed.loc[hottest_row_index]
    coldest_row = detailed.loc[coldest_row_index]

    cycle_maximums = cycle_summary.set_index(
        "cycle"
    )["maximum_temperature_c"]

    cycle_trend_slope = safe_slope(
        cycle_maximums.index,
        cycle_maximums.values,
    )

    summary_data = [
        {
            "metric": "Total records analyzed",
            "value": len(detailed),
            "unit": "records",
        },
        {
            "metric": "Total cycles analyzed",
            "value": detailed["cycle"].nunique(),
            "unit": "cycles",
        },
        {
            "metric": "Minimum dataset temperature",
            "value": temperature_series.min(),
            "unit": "C",
        },
        {
            "metric": "Maximum dataset temperature",
            "value": temperature_series.max(),
            "unit": "C",
        },
        {
            "metric": "Average dataset temperature",
            "value": temperature_series.mean(),
            "unit": "C",
        },
        {
            "metric": "Median dataset temperature",
            "value": temperature_series.median(),
            "unit": "C",
        },
        {
            "metric": "Temperature standard deviation",
            "value": temperature_series.std(),
            "unit": "C",
        },
        {
            "metric": "Overall temperature range",
            "value": (
                temperature_series.max()
                - temperature_series.min()
            ),
            "unit": "C",
        },
        {
            "metric": "Hottest cycle",
            "value": hottest_row["cycle"],
            "unit": "cycle",
        },
        {
            "metric": "Coldest cycle",
            "value": coldest_row["cycle"],
            "unit": "cycle",
        },
        {
            "metric": "Records above 50 C",
            "value": int(
                detailed["high_temperature_flag"].sum()
            ),
            "unit": "records",
        },
        {
            "metric": "Records above 60 C",
            "value": int(
                detailed[
                    "critical_temperature_flag"
                ].sum()
            ),
            "unit": "records",
        },
        {
            "metric": "Records below 0 C",
            "value": int(
                detailed[
                    "cold_temperature_flag"
                ].sum()
            ),
            "unit": "records",
        },
        {
            "metric": "Rapid heating records",
            "value": int(
                detailed["rapid_heating_flag"].sum()
            ),
            "unit": "records",
        },
        {
            "metric": "Maximum cycle temperature trend",
            "value": cycle_trend_slope,
            "unit": "C per cycle",
        },
    ]

    return pd.DataFrame(summary_data)


def build_warnings(detailed, cycle_summary):
    """
    Create human-readable thermal warnings.
    """
    warnings = []

    critical_records = detailed[
        detailed["temperature_c"] > 60
    ]

    high_records = detailed[
        (detailed["temperature_c"] > 50)
        & (detailed["temperature_c"] <= 60)
    ]

    cold_records = detailed[
        detailed["temperature_c"] < 0
    ]

    rapid_heating_records = detailed[
        detailed["rapid_heating_flag"]
    ]

    large_rise_cycles = cycle_summary[
        cycle_summary["temperature_rise_c"] > 20
    ]

    if not critical_records.empty:
        warnings.append(
            {
                "severity": "Critical",
                "warning_type": "Critical temperature",
                "affected_records": len(critical_records),
                "affected_cycles": (
                    critical_records["cycle"].nunique()
                ),
                "message": (
                    "Temperature exceeded 60 C. Verify the "
                    "cell datasheet limits, sensor calibration, "
                    "cooling system, and operating current."
                ),
            }
        )

    if not high_records.empty:
        warnings.append(
            {
                "severity": "High",
                "warning_type": "High temperature",
                "affected_records": len(high_records),
                "affected_cycles": (
                    high_records["cycle"].nunique()
                ),
                "message": (
                    "Temperature was between 50 C and 60 C. "
                    "Inspect thermal management and load conditions."
                ),
            }
        )

    if not cold_records.empty:
        warnings.append(
            {
                "severity": "High",
                "warning_type": "Sub-zero temperature",
                "affected_records": len(cold_records),
                "affected_cycles": (
                    cold_records["cycle"].nunique()
                ),
                "message": (
                    "Temperature dropped below 0 C. Charging a "
                    "lithium-ion cell at sub-zero temperature can "
                    "increase lithium-plating risk."
                ),
            }
        )

    if not rapid_heating_records.empty:
        warnings.append(
            {
                "severity": "Medium",
                "warning_type": "Rapid temperature rise",
                "affected_records": len(
                    rapid_heating_records
                ),
                "affected_cycles": (
                    rapid_heating_records[
                        "cycle"
                    ].nunique()
                ),
                "message": (
                    "Temperature increased faster than "
                    "0.1 C per second in one or more records. "
                    "Check for noisy timestamps, sensor spikes, "
                    "high current, or abnormal heating."
                ),
            }
        )

    if not large_rise_cycles.empty:
        warnings.append(
            {
                "severity": "Medium",
                "warning_type": "Large within-cycle rise",
                "affected_records": len(
                    large_rise_cycles
                ),
                "affected_cycles": len(
                    large_rise_cycles
                ),
                "message": (
                    "One or more cycles showed a temperature "
                    "rise greater than 20 C."
                ),
            }
        )

    if not warnings:
        warnings.append(
            {
                "severity": "Information",
                "warning_type": "No major thermal warning",
                "affected_records": 0,
                "affected_cycles": 0,
                "message": (
                    "No temperature values crossed the default "
                    "diagnostic warning thresholds."
                ),
            }
        )

    return pd.DataFrame(warnings)


# ============================================================
# VISUALIZATION
# ============================================================

def select_representative_cycles(cycle_values, maximum_cycles=8):
    """
    Select cycles distributed across the dataset for a readable graph.
    """
    unique_cycles = np.array(
        sorted(pd.Series(cycle_values).dropna().unique())
    )

    if len(unique_cycles) <= maximum_cycles:
        return unique_cycles.tolist()

    selected_indices = np.linspace(
        0,
        len(unique_cycles) - 1,
        maximum_cycles,
        dtype=int,
    )

    return unique_cycles[selected_indices].tolist()


def create_temperature_graph(
    detailed,
    cycle_summary,
    output_path,
):
    """
    Create a four-panel temperature diagnostics graph.
    """
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(16, 10),
    )

    figure.suptitle(
        "Battery Temperature Diagnostics",
        fontsize=17,
        fontweight="bold",
    )

    selected_cycles = select_representative_cycles(
        detailed["cycle"],
        maximum_cycles=8,
    )

    for cycle_value in selected_cycles:
        cycle_data = detailed[
            detailed["cycle"] == cycle_value
        ].sort_values("time_s")

        relative_time_min = (
            cycle_data["time_s"]
            - cycle_data["time_s"].min()
        ) / 60

        axes[0, 0].plot(
            relative_time_min,
            cycle_data["temperature_c"],
            linewidth=1.5,
            label=f"Cycle {cycle_value:g}",
        )

    axes[0, 0].axhline(
        50,
        color="orange",
        linestyle="--",
        linewidth=1.2,
        label="High threshold: 50 C",
    )

    axes[0, 0].axhline(
        60,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label="Critical threshold: 60 C",
    )

    axes[0, 0].set_title(
        "Temperature Profiles for Representative Cycles"
    )
    axes[0, 0].set_xlabel(
        "Relative Time (minutes)"
    )
    axes[0, 0].set_ylabel(
        "Temperature (C)"
    )
    axes[0, 0].grid(
        True,
        alpha=0.3,
    )
    axes[0, 0].legend(
        fontsize=8,
        loc="best",
    )

    axes[0, 1].plot(
        cycle_summary["cycle"],
        cycle_summary["minimum_temperature_c"],
        marker="o",
        markersize=3,
        linewidth=1.3,
        label="Minimum",
    )

    axes[0, 1].plot(
        cycle_summary["cycle"],
        cycle_summary["average_temperature_c"],
        marker="o",
        markersize=3,
        linewidth=1.3,
        label="Average",
    )

    axes[0, 1].plot(
        cycle_summary["cycle"],
        cycle_summary["maximum_temperature_c"],
        marker="o",
        markersize=3,
        linewidth=1.3,
        label="Maximum",
    )

    axes[0, 1].axhline(
        50,
        color="orange",
        linestyle="--",
        linewidth=1.1,
    )

    axes[0, 1].axhline(
        60,
        color="red",
        linestyle="--",
        linewidth=1.1,
    )

    axes[0, 1].set_title(
        "Temperature Trend Across Cycles"
    )
    axes[0, 1].set_xlabel("Cycle")
    axes[0, 1].set_ylabel("Temperature (C)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].bar(
        cycle_summary["cycle"].astype(str),
        cycle_summary["temperature_rise_c"],
        color="tab:orange",
        alpha=0.85,
    )

    axes[1, 0].axhline(
        10,
        color="orange",
        linestyle="--",
        linewidth=1.1,
        label="Moderate rise: 10 C",
    )

    axes[1, 0].axhline(
        20,
        color="red",
        linestyle="--",
        linewidth=1.1,
        label="Large rise: 20 C",
    )

    axes[1, 0].set_title(
        "Within-Cycle Temperature Rise"
    )
    axes[1, 0].set_xlabel("Cycle")
    axes[1, 0].set_ylabel(
        "Temperature Rise (C)"
    )
    axes[1, 0].tick_params(
        axis="x",
        rotation=90,
        labelsize=7,
    )
    axes[1, 0].grid(
        True,
        axis="y",
        alpha=0.3,
    )
    axes[1, 0].legend(fontsize=8)

    finite_temperatures = detailed[
        "temperature_c"
    ].dropna()

    bin_count = min(
        50,
        max(10, int(np.sqrt(len(finite_temperatures)))),
    )

    axes[1, 1].hist(
        finite_temperatures,
        bins=bin_count,
        color="tab:blue",
        edgecolor="black",
        alpha=0.8,
    )

    axes[1, 1].axvline(
        finite_temperatures.mean(),
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=(
            "Mean: "
            f"{finite_temperatures.mean():.2f} C"
        ),
    )

    axes[1, 1].axvline(
        50,
        color="orange",
        linestyle="--",
        linewidth=1.2,
        label="High threshold: 50 C",
    )

    axes[1, 1].axvline(
        60,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label="Critical threshold: 60 C",
    )

    axes[1, 1].set_title(
        "Temperature Distribution"
    )
    axes[1, 1].set_xlabel(
        "Temperature (C)"
    )
    axes[1, 1].set_ylabel(
        "Record Count"
    )
    axes[1, 1].grid(
        True,
        axis="y",
        alpha=0.3,
    )
    axes[1, 1].legend(fontsize=8)

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
# OUTPUT HANDLING
# ============================================================

def get_output_directory(dataset_path):
    """
    Store the output in the dataset folder when the input is the
    normalized dataset. This keeps the result under the existing
    b0005_discharge_curves output directory.
    """
    dataset_path = Path(dataset_path).resolve()

    if (
        dataset_path.name.lower()
        == "normalized_battery_data.csv"
        and dataset_path.parent.name.lower()
        == "unit_normalization"
    ):
        dataset_output_root = dataset_path.parent.parent
    else:
        project_root = (
            Path(__file__).resolve().parents[2]
        )

        dataset_output_root = (
            project_root
            / "outputs"
            / dataset_path.stem
        )

    output_directory = (
        dataset_output_root
        / "temperature_diagnostics"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory


def save_results(
    detailed,
    cycle_summary,
    analysis_summary,
    warnings,
    output_directory,
):
    """
    Save all temperature diagnostic reports and the graph.
    """
    detailed_path = (
        output_directory
        / "temperature_detailed_records.csv"
    )

    cycle_summary_path = (
        output_directory
        / "temperature_cycle_summary.csv"
    )

    analysis_summary_path = (
        output_directory
        / "temperature_analysis_summary.csv"
    )

    warnings_path = (
        output_directory
        / "temperature_warnings.csv"
    )

    graph_path = (
        output_directory
        / "temperature_diagnostics.png"
    )

    detailed.to_csv(
        detailed_path,
        index=False,
    )

    cycle_summary.to_csv(
        cycle_summary_path,
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

    create_temperature_graph(
        detailed,
        cycle_summary,
        graph_path,
    )

    return {
        "detailed_records": detailed_path,
        "cycle_summary": cycle_summary_path,
        "analysis_summary": analysis_summary_path,
        "warnings": warnings_path,
        "graph": graph_path,
    }


# ============================================================
# MAIN DIAGNOSTICS PIPELINE
# ============================================================

def run_temperature_diagnostics(dataset_path):
    """
    Run the complete temperature diagnostics pipeline.
    """
    dataframe, resolved_dataset_path = load_dataset(
        dataset_path
    )

    prepared = prepare_temperature_data(dataframe)

    detailed = build_detailed_records(prepared)

    cycle_summary = build_cycle_summary(detailed)

    analysis_summary = build_analysis_summary(
        detailed,
        cycle_summary,
    )

    warnings = build_warnings(
        detailed,
        cycle_summary,
    )

    output_directory = get_output_directory(
        resolved_dataset_path
    )

    output_files = save_results(
        detailed,
        cycle_summary,
        analysis_summary,
        warnings,
        output_directory,
    )

    return {
        "output_directory": output_directory,
        "output_files": output_files,
        "analysis_summary": analysis_summary,
        "warnings": warnings,
    }


def main():
    print("\nBattery Temperature Diagnostics")
    print("=" * 40)

    dataset_path = input(
        "Paste the full path of a raw or normalized "
        "battery dataset: "
    ).strip().strip('"').strip("'")

    if not dataset_path:
        print(
            "\nTemperature diagnostics failed: "
            "No dataset path was supplied."
        )
        sys.exit(1)

    try:
        results = run_temperature_diagnostics(
            dataset_path
        )

        print(
            "\nTemperature diagnostics completed successfully."
        )

        print(
            f"\nOutput folder:\n"
            f"{results['output_directory']}"
        )

        print("\nGenerated files:")

        for file_name, file_path in results[
            "output_files"
        ].items():
            print(f"  {file_name}: {file_path}")

        print("\nImportant summary:")

        for _, row in results[
            "analysis_summary"
        ].iterrows():
            value = row["value"]

            if isinstance(
                value,
                (float, np.floating),
            ):
                value = (
                    f"{value:.4f}"
                    if np.isfinite(value)
                    else "Not available"
                )

            print(
                f"  {row['metric']}: "
                f"{value} {row['unit']}"
            )

        print("\nWarnings:")

        for _, row in results["warnings"].iterrows():
            print(
                f"  [{row['severity']}] "
                f"{row['warning_type']}: "
                f"{row['message']}"
            )

        print(
            "\nNote: Diagnostic thresholds are defaults. "
            "Use the battery manufacturer's cell datasheet "
            "for final operating and safety limits."
        )

    except Exception as error:
        print(
            f"\nTemperature diagnostics failed: {error}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()