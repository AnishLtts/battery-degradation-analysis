from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# ============================================================
# LOCATE PLATFORM FOLDERS
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
PLATFORM_FOLDER = CURRENT_FILE.parent.parent
CORE_FOLDER = PLATFORM_FOLDER / "core"

# Allow imports from the core folder
sys.path.insert(0, str(CORE_FOLDER))

from file_reader import load_battery_dataset
from column_mapper import (
    apply_column_mapping,
    detect_column_mapping,
)


# ============================================================
# DATA CLEANING
# ============================================================

def remove_invalid_capacity_rows(dataframe):
    """
    Remove rows that cannot be used for capacity-based SOH.
    """

    if "capacity_ah" not in dataframe.columns:
        raise ValueError(
            "SOH calculation requires a capacity_ah column."
        )

    cleaned_data = dataframe.copy()

    cleaned_data["capacity_ah"] = pd.to_numeric(
        cleaned_data["capacity_ah"],
        errors="coerce",
    )

    # Remove records with missing capacity values
    cleaned_data = cleaned_data.dropna(
        subset=["capacity_ah"]
    )

    # Capacity must be greater than zero
    cleaned_data = cleaned_data[
        cleaned_data["capacity_ah"] > 0
    ]

    # If test type is available, prefer discharge records
    if "test_type" in cleaned_data.columns:
        test_type = (
            cleaned_data["test_type"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        discharge_mask = test_type.str.contains(
            "discharge",
            na=False,
        )

        if discharge_mask.any():
            cleaned_data = cleaned_data[
                discharge_mask
            ].copy()

    if cleaned_data.empty:
        raise ValueError(
            "No valid positive capacity records were found."
        )

    return cleaned_data


# ============================================================
# DATA SORTING
# ============================================================

def sort_battery_records(dataframe):
    """
    Sort records into a reasonable chronological order.
    """

    sorted_data = dataframe.copy()

    sort_columns = []

    if "battery_id" in sorted_data.columns:
        sort_columns.append("battery_id")

    if "cycle" in sorted_data.columns:
        sort_columns.append("cycle")

    elif "test_id" in sorted_data.columns:
        sort_columns.append("test_id")

    elif "uid" in sorted_data.columns:
        sort_columns.append("uid")

    if sort_columns:
        sorted_data = sorted_data.sort_values(
            by=sort_columns,
            kind="stable",
        )

    return sorted_data.reset_index(drop=True)


def add_battery_cycle_number(dataframe):
    """
    Add a sequential record number for each battery.
    """

    numbered_data = dataframe.copy()

    if "battery_id" in numbered_data.columns:
        numbered_data["battery_record_number"] = (
            numbered_data.groupby(
                "battery_id"
            ).cumcount() + 1
        )

    else:
        numbered_data["battery_record_number"] = (
            np.arange(len(numbered_data)) + 1
        )

    return numbered_data


# ============================================================
# REFERENCE CAPACITY
# ============================================================

def calculate_reference_capacity(
    dataframe,
    rated_capacity_ah=None,
    initial_records=3,
):
    """
    Determine the reference capacity.

    If rated_capacity_ah is supplied, the same user-supplied
    reference is used for all batteries.

    Otherwise, the median of the highest plausible capacity
    records is calculated separately for every battery.
    """

    data_with_reference = dataframe.copy()

    if rated_capacity_ah is not None:
        rated_capacity_ah = float(
            rated_capacity_ah
        )

        if rated_capacity_ah <= 0:
            raise ValueError(
                "Rated capacity must be greater than zero."
            )

        data_with_reference["reference_capacity_ah"] = (
            rated_capacity_ah
        )

        data_with_reference["reference_source"] = (
            "User-supplied rated capacity"
        )

        return data_with_reference

    initial_records = max(
        1,
        int(initial_records),
    )

    if "battery_id" in data_with_reference.columns:
        reference_rows = []

        for battery_id, battery_data in (
            data_with_reference.groupby(
                "battery_id",
                sort=False,
            )
        ):
            battery_data = battery_data.copy()

            battery_median = battery_data[
                "capacity_ah"
            ].median()

            # Remove suspicious incomplete-capacity records
            plausible_data = battery_data[
                battery_data["capacity_ah"]
                >= 0.70 * battery_median
            ].copy()

            if plausible_data.empty:
                plausible_data = battery_data.copy()

            # Use the median of the highest plausible capacity values
            candidate_count = min(
                max(initial_records, 5),
                len(plausible_data),
            )

            reference_capacity = (
                plausible_data["capacity_ah"]
                .nlargest(candidate_count)
                .median()
            )

            reference_rows.append(
                {
                    "battery_id": battery_id,
                    "reference_capacity_ah": reference_capacity,
                }
            )

        reference_table = pd.DataFrame(
            reference_rows
        )

        data_with_reference = data_with_reference.merge(
            reference_table,
            on="battery_id",
            how="left",
        )

    else:
        reference_capacity = (
            data_with_reference["capacity_ah"]
            .head(initial_records)
            .median()
        )

        data_with_reference["reference_capacity_ah"] = (
            reference_capacity
        )

    data_with_reference["reference_source"] = (
        "Median of the highest plausible capacity records "
        "for each battery"
    )

    return data_with_reference


# ============================================================
# HEALTH CLASSIFICATION
# ============================================================

def assign_health_category(soh_percent):
    """
    Assign an understandable battery health category.

    These categories are dashboard categories.
    They are not universal battery safety limits.
    """

    if pd.isna(soh_percent):
        return "Unknown"

    if soh_percent >= 90:
        return "Healthy"

    if soh_percent >= 80:
        return "Moderately Aged"

    if soh_percent >= 70:
        return "Degraded"

    return "Severely Degraded"


# ============================================================
# SOH CALCULATION
# ============================================================

def calculate_capacity_soh(
    dataframe,
    rated_capacity_ah=None,
    initial_records=3,
):
    """
    Calculate capacity-based battery SOH.

    SOH (%) =
        measured capacity / reference capacity * 100
    """

    valid_data = remove_invalid_capacity_rows(
        dataframe
    )

    valid_data = sort_battery_records(
        valid_data
    )

    valid_data = add_battery_cycle_number(
        valid_data
    )

    soh_data = calculate_reference_capacity(
        valid_data,
        rated_capacity_ah=rated_capacity_ah,
        initial_records=initial_records,
    )

    soh_data["soh_percent"] = (
        soh_data["capacity_ah"]
        / soh_data["reference_capacity_ah"]
        * 100
    )

    soh_data["capacity_loss_percent"] = (
        100 - soh_data["soh_percent"]
    )

    soh_data["health_category"] = (
        soh_data["soh_percent"]
        .apply(assign_health_category)
    )

    return soh_data


# ============================================================
# BATTERY SUMMARY
# ============================================================

def create_battery_summary(soh_data):
    """
    Create one summary row for each battery.
    """

    if "battery_id" not in soh_data.columns:
        latest_record = soh_data.iloc[-1]

        summary = pd.DataFrame(
            {
                "battery_id": ["dataset"],
                "reference_capacity_ah": [
                    soh_data[
                        "reference_capacity_ah"
                    ].iloc[0]
                ],
                "latest_capacity_ah": [
                    latest_record["capacity_ah"]
                ],
                "latest_soh_percent": [
                    latest_record["soh_percent"]
                ],
                "minimum_soh_percent": [
                    soh_data["soh_percent"].min()
                ],
                "capacity_loss_percent": [
                    latest_record[
                        "capacity_loss_percent"
                    ]
                ],
                "health_category": [
                    latest_record[
                        "health_category"
                    ]
                ],
                "records_analyzed": [
                    len(soh_data)
                ],
            }
        )

    else:
        summary_rows = []

        for battery_id, battery_data in (
            soh_data.groupby(
                "battery_id",
                sort=False,
            )
        ):
            battery_data = battery_data.sort_values(
                "battery_record_number"
            )

            latest_record = battery_data.iloc[-1]

            summary_rows.append(
                {
                    "battery_id": battery_id,
                    "reference_capacity_ah": (
                        battery_data[
                            "reference_capacity_ah"
                        ].iloc[0]
                    ),
                    "latest_capacity_ah": (
                        latest_record[
                            "capacity_ah"
                        ]
                    ),
                    "latest_soh_percent": (
                        latest_record[
                            "soh_percent"
                        ]
                    ),
                    "minimum_soh_percent": (
                        battery_data[
                            "soh_percent"
                        ].min()
                    ),
                    "capacity_loss_percent": (
                        latest_record[
                            "capacity_loss_percent"
                        ]
                    ),
                    "health_category": (
                        latest_record[
                            "health_category"
                        ]
                    ),
                    "records_analyzed": len(
                        battery_data
                    ),
                }
            )

        summary = pd.DataFrame(
            summary_rows
        )

    numeric_columns = summary.select_dtypes(
        include="number"
    ).columns

    summary[numeric_columns] = (
        summary[numeric_columns].round(3)
    )

    return summary


# ============================================================
# CLEAN BATTERY SOH SUMMARY GRAPH
# ============================================================

def create_soh_graph(summary):
    """
    Create a clean bar chart showing the latest SOH
    percentage for every battery.

    Bar colors represent the battery health category.
    """

    if summary.empty:
        raise ValueError(
            "Battery summary is empty. "
            "The SOH graph cannot be created."
        )

    graph_data = summary.copy()

    graph_data["latest_soh_percent"] = pd.to_numeric(
        graph_data["latest_soh_percent"],
        errors="coerce",
    )

    graph_data = graph_data.dropna(
        subset=["latest_soh_percent"]
    )

    if graph_data.empty:
        raise ValueError(
            "No valid latest SOH values are available "
            "for the graph."
        )

    # Sort batteries by battery ID where possible
    graph_data["_battery_sort"] = pd.to_numeric(
        graph_data["battery_id"],
        errors="coerce",
    )

    graph_data = graph_data.sort_values(
        by=["_battery_sort", "battery_id"],
        na_position="last",
    ).drop(
        columns=["_battery_sort"]
    )

    graph_data = graph_data.reset_index(
        drop=True
    )

    category_colors = {
        "Healthy": "#2E8B57",
        "Moderately Aged": "#F4A261",
        "Degraded": "#E76F51",
        "Severely Degraded": "#9B2226",
        "Unknown": "#808080",
    }

    bar_colors = (
        graph_data["health_category"]
        .map(category_colors)
        .fillna("#808080")
    )

    number_of_batteries = len(graph_data)

    figure_width = max(
        14,
        number_of_batteries * 0.48,
    )

    figure, axis = plt.subplots(
        figsize=(figure_width, 8)
    )

    x_positions = np.arange(
        number_of_batteries
    )

    bars = axis.bar(
        x_positions,
        graph_data["latest_soh_percent"],
        color=bar_colors,
        edgecolor="#333333",
        linewidth=0.6,
        width=0.72,
    )

    # Add health threshold lines
    axis.axhline(
        y=90,
        color="#2E8B57",
        linestyle="--",
        linewidth=1.3,
        alpha=0.8,
    )

    axis.axhline(
        y=80,
        color="#F4A261",
        linestyle="--",
        linewidth=1.3,
        alpha=0.8,
    )

    axis.axhline(
        y=70,
        color="#E76F51",
        linestyle="--",
        linewidth=1.3,
        alpha=0.8,
    )

    # Add SOH percentage above every bar
    for bar, soh_value in zip(
        bars,
        graph_data["latest_soh_percent"],
    ):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{soh_value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            rotation=90,
        )

    axis.set_title(
        "Latest Capacity-Based State of Health by Battery",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )

    axis.set_xlabel(
        "Battery ID",
        fontsize=12,
        fontweight="bold",
    )

    axis.set_ylabel(
        "Latest SOH (%)",
        fontsize=12,
        fontweight="bold",
    )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        graph_data["battery_id"].astype(str),
        rotation=45,
        ha="right",
        fontsize=9,
    )

    maximum_soh = graph_data[
        "latest_soh_percent"
    ].max()

    axis.set_ylim(
        0,
        max(110, maximum_soh + 15),
    )

    axis.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.3,
    )

    axis.set_axisbelow(True)

    legend_items = [
        Patch(
            facecolor="#2E8B57",
            edgecolor="#333333",
            label="Healthy: 90% or above",
        ),
        Patch(
            facecolor="#F4A261",
            edgecolor="#333333",
            label="Moderately Aged: 80% to 89.9%",
        ),
        Patch(
            facecolor="#E76F51",
            edgecolor="#333333",
            label="Degraded: 70% to 79.9%",
        ),
        Patch(
            facecolor="#9B2226",
            edgecolor="#333333",
            label="Severely Degraded: below 70%",
        ),
    ]

    axis.legend(
        handles=legend_items,
        loc="upper right",
        fontsize=9,
        frameon=True,
    )

    figure.tight_layout()

    return figure


# ============================================================
# SAVE RESULTS
# ============================================================

def save_soh_results(
    soh_data,
    summary,
    output_directory,
):
    """
    Save detailed SOH results, battery summary,
    and the latest-SOH graph.
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
        / "soh_detailed_results.csv"
    )

    summary_path = (
        output_directory
        / "soh_battery_summary.csv"
    )

    graph_path = (
        output_directory
        / "soh_latest_summary.png"
    )

    soh_data.to_csv(
        detailed_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    figure = create_soh_graph(
        summary
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
        "summary": summary_path,
        "graph": graph_path,
    }


# ============================================================
# PRINT REPORT
# ============================================================

def print_soh_report(
    soh_data,
    summary,
    saved_files,
):
    """
    Print the calculated SOH results.
    """

    print("\n" + "=" * 75)
    print("CAPACITY-BASED BATTERY SOH ANALYSIS")
    print("=" * 75)

    print(
        "\nSOH formula: "
        "Measured Capacity / Reference Capacity x 100"
    )

    print(
        "\nReference source: "
        f"{soh_data['reference_source'].iloc[0]}"
    )

    print("\nBATTERY SUMMARY")

    print(
        summary.to_string(
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

    print("\nGRAPH COLOR MEANING")
    print("  Green: Healthy, 90% or above")
    print("  Orange: Moderately Aged, 80% to 89.9%")
    print("  Red: Degraded, 70% to 79.9%")
    print("  Dark Red: Severely Degraded, below 70%")

    print("\nImportant:")
    print(
        "  Capacity-based SOH describes capacity retention. "
        "It does not alone represent every aspect of "
        "battery health or safety."
    )

    print("=" * 75)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":
    dataset_path = input(
        "Paste the full path of a battery dataset: "
    ).strip().strip('"')

    rated_capacity_text = input(
        "Enter rated capacity in Ah, or press Enter "
        "to estimate the reference capacity for each battery: "
    ).strip()

    try:
        rated_capacity = (
            float(rated_capacity_text)
            if rated_capacity_text
            else None
        )

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

        soh_results = calculate_capacity_soh(
            standardized_data,
            rated_capacity_ah=rated_capacity,
            initial_records=3,
        )

        battery_summary = create_battery_summary(
            soh_results
        )

        output_folder = (
            PLATFORM_FOLDER
            / "outputs"
            / Path(dataset_path).stem
            / "soh_analysis"
        )

        saved_results = save_soh_results(
            soh_results,
            battery_summary,
            output_folder,
        )

        print_soh_report(
            soh_results,
            battery_summary,
            saved_results,
        )

        if unknown_columns:
            print("\nUnrecognized original columns:")

            for column in unknown_columns:
                print(f"  - {column}")

            print(
                "\nNote: Unrecognized columns are retained "
                "in the dataset but are not required for "
                "the current SOH calculation."
            )

    except Exception as error:
        print(
            f"\nSOH analysis failed: {error}"
        )