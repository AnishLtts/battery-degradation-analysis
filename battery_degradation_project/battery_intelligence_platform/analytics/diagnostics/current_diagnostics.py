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
# CURRENT DIAGNOSTIC CONFIGURATION
# ============================================================

MINIMUM_POINTS_PER_CYCLE = 10
MAXIMUM_GRAPH_CYCLES = 6

# Current values with magnitude at or below this threshold are
# treated as near-zero or rest records.
REST_CURRENT_THRESHOLD_A = 0.01

# Current changes larger than this fixed value are recorded as
# exploratory abrupt-change events.
CURRENT_CHANGE_THRESHOLD_A = 0.25

# Current stability is measured using coefficient of variation:
#
# standard deviation / absolute mean current * 100
#
# These thresholds are analytical screening values.
STABLE_CURRENT_CV_PERCENT = 2.0
MODERATE_CURRENT_CV_PERCENT = 5.0

# A cycle requires this minimum mean current magnitude before
# coefficient-of-variation classification is considered valid.
MINIMUM_ACTIVE_CURRENT_A = 0.05

# Current spikes are identified using median absolute deviation.
ROBUST_SPIKE_Z_THRESHOLD = 6.0

# Current spike metrics remain exploratory and are not treated
# as safety events without battery specifications and C-rate.
CURRENT_SPIKES_USED_FOR_SAFETY = False


# ============================================================
# VALIDATE CURRENT DATASET
# ============================================================

def validate_current_dataset(dataframe):
    """
    Validate and clean a dataset for current diagnostics.

    Required columns:
        cycle
        current_a

    Optional columns:
        time_s
        voltage_v
        capacity_ah
        temperature_c
    """

    required_columns = [
        "cycle",
        "current_a",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Current diagnostics requires: "
            + ", ".join(missing_columns)
        )

    cleaned_data = dataframe.copy()

    numeric_columns = [
        "cycle",
        "current_a",
        "time_s",
        "voltage_v",
        "capacity_ah",
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
            "current_a",
        ]
    )

    cleaned_data = cleaned_data[
        np.isfinite(cleaned_data["cycle"])
        & np.isfinite(cleaned_data["current_a"])
    ].copy()

    if cleaned_data.empty:
        raise ValueError(
            "No valid cycle and current records were found."
        )

    cleaned_data = cleaned_data.sort_values(
        by=["cycle"],
        kind="stable",
    ).reset_index(drop=True)

    return cleaned_data


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
# DETECT CURRENT SIGN CONVENTION
# ============================================================

def detect_current_sign_convention(dataframe):
    """
    Detect the dominant active-current direction.

    This identifies the dominant sign but does not universally
    prove whether the sign represents charge or discharge.
    """

    current_values = (
        dataframe["current_a"]
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    negative_count = int(
        (
            current_values
            < -REST_CURRENT_THRESHOLD_A
        ).sum()
    )

    positive_count = int(
        (
            current_values
            > REST_CURRENT_THRESHOLD_A
        ).sum()
    )

    rest_count = int(
        (
            np.abs(current_values)
            <= REST_CURRENT_THRESHOLD_A
        ).sum()
    )

    active_count = (
        negative_count
        + positive_count
    )

    if active_count == 0:
        dominant_sign = "None"
        dominant_operation = "No active current detected"
        dominant_fraction = 0.0

    elif negative_count >= positive_count:
        dominant_sign = "Negative"
        dominant_operation = (
            "Negative current is the dominant active direction"
        )

        dominant_fraction = (
            negative_count
            / active_count
            * 100
        )

    else:
        dominant_sign = "Positive"
        dominant_operation = (
            "Positive current is the dominant active direction"
        )

        dominant_fraction = (
            positive_count
            / active_count
            * 100
        )

    return {
        "negative_record_count": negative_count,
        "positive_record_count": positive_count,
        "rest_record_count": rest_count,
        "active_record_count": active_count,
        "dominant_sign": dominant_sign,
        "dominant_operation": dominant_operation,
        "dominant_active_fraction_percent": float(
            dominant_fraction
        ),
    }


# ============================================================
# SELECT DOMINANT ACTIVE CURRENT
# ============================================================

def select_dominant_active_records(
    dataframe,
    sign_information,
):
    """
    Select records matching the dominant active-current sign.

    The complete dataset remains available for overall current
    statistics. The selected records are used for active-current
    stability analysis and representative graphs.
    """

    dominant_sign = sign_information[
        "dominant_sign"
    ]

    if dominant_sign == "Negative":
        selected_data = dataframe[
            dataframe["current_a"]
            < -REST_CURRENT_THRESHOLD_A
        ].copy()

        message = (
            "Negative current was treated as the dominant "
            "active-current direction."
        )

    elif dominant_sign == "Positive":
        selected_data = dataframe[
            dataframe["current_a"]
            > REST_CURRENT_THRESHOLD_A
        ].copy()

        message = (
            "Positive current was treated as the dominant "
            "active-current direction."
        )

    else:
        selected_data = dataframe.copy()

        message = (
            "No dominant active-current direction was found. "
            "All valid current records were retained."
        )

    if (
        len(selected_data) < MINIMUM_POINTS_PER_CYCLE
        or selected_data["cycle"].nunique() == 0
    ):
        selected_data = dataframe.copy()

        message = (
            "Dominant-sign filtering left insufficient data. "
            "All valid current records were retained."
        )

    return selected_data, message


# ============================================================
# CALCULATE ROBUST SPIKE METRICS
# ============================================================

def calculate_robust_spike_metrics(current_values):
    """
    Detect exploratory current outliers using median absolute
    deviation.

    These values indicate statistical outliers only. They are
    not interpreted as electrical safety faults because maximum
    permitted current and C-rate are not known.
    """

    values = np.asarray(
        current_values,
        dtype=float,
    )

    if len(values) < 5:
        return {
            "median_current_a": float(
                np.median(values)
            ) if len(values) else None,
            "median_absolute_deviation_a": 0.0,
            "robust_spike_count": 0,
            "maximum_robust_z_score": 0.0,
        }

    median_current = float(
        np.median(values)
    )

    absolute_deviations = np.abs(
        values - median_current
    )

    median_absolute_deviation = float(
        np.median(
            absolute_deviations
        )
    )

    if median_absolute_deviation <= 1e-12:
        return {
            "median_current_a": median_current,
            "median_absolute_deviation_a": (
                median_absolute_deviation
            ),
            "robust_spike_count": 0,
            "maximum_robust_z_score": 0.0,
        }

    robust_z_scores = (
        0.6745
        * (
            values
            - median_current
        )
        / median_absolute_deviation
    )

    spike_flags = (
        np.abs(robust_z_scores)
        > ROBUST_SPIKE_Z_THRESHOLD
    )

    return {
        "median_current_a": median_current,
        "median_absolute_deviation_a": (
            median_absolute_deviation
        ),
        "robust_spike_count": int(
            spike_flags.sum()
        ),
        "maximum_robust_z_score": float(
            np.max(
                np.abs(
                    robust_z_scores
                )
            )
        ),
    }


# ============================================================
# CLASSIFY CURRENT STABILITY
# ============================================================

def classify_current_stability(
    absolute_mean_current,
    current_standard_deviation,
):
    """
    Classify active-current stability using coefficient of
    variation.
    """

    if (
        absolute_mean_current
        < MINIMUM_ACTIVE_CURRENT_A
    ):
        return {
            "coefficient_of_variation_percent": None,
            "stability_category": "Insufficient active current",
        }

    coefficient_of_variation = (
        current_standard_deviation
        / absolute_mean_current
        * 100
    )

    if (
        coefficient_of_variation
        <= STABLE_CURRENT_CV_PERCENT
    ):
        stability_category = "Stable"

    elif (
        coefficient_of_variation
        <= MODERATE_CURRENT_CV_PERCENT
    ):
        stability_category = "Moderately variable"

    else:
        stability_category = "Variable"

    return {
        "coefficient_of_variation_percent": float(
            coefficient_of_variation
        ),
        "stability_category": (
            stability_category
        ),
    }


# ============================================================
# ANALYZE ONE CYCLE
# ============================================================

def analyze_current_cycle(
    cycle_number,
    cycle_data,
    dominant_sign,
):
    """
    Calculate current statistics and analytical warnings for
    one cycle.
    """

    sorted_cycle, sequence_column = (
        sort_cycle_records(
            cycle_data
        )
    )

    sorted_cycle = sorted_cycle[
        sorted_cycle["current_a"].notna()
    ].reset_index(drop=True)

    current_values = (
        sorted_cycle["current_a"]
        .to_numpy(
            dtype=float
        )
    )

    point_count = int(
        len(current_values)
    )

    if point_count == 0:
        return {
            "cycle": cycle_number,
            "status": "Rejected",
            "reason": (
                "No valid current measurements were found."
            ),
            "sequence_column": sequence_column,
            "point_count": 0,
            "warning_count": 1,
        }

    minimum_current = float(
        np.min(
            current_values
        )
    )

    maximum_current = float(
        np.max(
            current_values
        )
    )

    mean_current = float(
        np.mean(
            current_values
        )
    )

    median_current = float(
        np.median(
            current_values
        )
    )

    current_standard_deviation = float(
        np.std(
            current_values,
            ddof=0,
        )
    )

    current_range = float(
        maximum_current
        - minimum_current
    )

    negative_record_count = int(
        (
            current_values
            < -REST_CURRENT_THRESHOLD_A
        ).sum()
    )

    positive_record_count = int(
        (
            current_values
            > REST_CURRENT_THRESHOLD_A
        ).sum()
    )

    rest_record_count = int(
        (
            np.abs(
                current_values
            )
            <= REST_CURRENT_THRESHOLD_A
        ).sum()
    )

    active_record_count = (
        negative_record_count
        + positive_record_count
    )

    rest_fraction_percent = float(
        rest_record_count
        / point_count
        * 100
    )

    if dominant_sign == "Negative":
        dominant_values = current_values[
            current_values
            < -REST_CURRENT_THRESHOLD_A
        ]

        opposite_sign_count = (
            positive_record_count
        )

    elif dominant_sign == "Positive":
        dominant_values = current_values[
            current_values
            > REST_CURRENT_THRESHOLD_A
        ]

        opposite_sign_count = (
            negative_record_count
        )

    else:
        dominant_values = current_values[
            np.abs(
                current_values
            )
            > REST_CURRENT_THRESHOLD_A
        ]

        opposite_sign_count = 0

    if len(dominant_values) > 0:
        dominant_mean_current = float(
            np.mean(
                dominant_values
            )
        )

        dominant_absolute_mean = float(
            np.mean(
                np.abs(
                    dominant_values
                )
            )
        )

        dominant_current_standard_deviation = float(
            np.std(
                dominant_values,
                ddof=0,
            )
        )

    else:
        dominant_mean_current = 0.0
        dominant_absolute_mean = 0.0
        dominant_current_standard_deviation = 0.0

    stability_result = classify_current_stability(
        dominant_absolute_mean,
        dominant_current_standard_deviation,
    )

    # Abrupt current changes are exploratory because known
    # charging or test protocols can contain intentional steps.
    current_differences = np.diff(
        current_values
    )

    abrupt_change_count = int(
        (
            np.abs(
                current_differences
            )
            > CURRENT_CHANGE_THRESHOLD_A
        ).sum()
    )

    maximum_positive_change = 0.0
    maximum_negative_change = 0.0
    maximum_absolute_change = 0.0

    if len(current_differences) > 0:
        maximum_positive_change = float(
            max(
                0.0,
                np.max(
                    current_differences
                ),
            )
        )

        maximum_negative_change = float(
            min(
                0.0,
                np.min(
                    current_differences
                ),
            )
        )

        maximum_absolute_change = float(
            np.max(
                np.abs(
                    current_differences
                )
            )
        )

    spike_metrics = (
        calculate_robust_spike_metrics(
            dominant_values
            if len(dominant_values) > 0
            else current_values
        )
    )

    warnings = []

    if (
        point_count
        < MINIMUM_POINTS_PER_CYCLE
    ):
        warnings.append(
            "Insufficient points for reliable current analysis."
        )

    if active_record_count == 0:
        warnings.append(
            "No active-current records were detected."
        )

    if opposite_sign_count > 0:
        warnings.append(
            f"{opposite_sign_count} records use the opposite "
            "current sign within this cycle. This may represent "
            "charge, recovery, or operation transitions."
        )

    if (
        stability_result[
            "stability_category"
        ]
        == "Variable"
    ):
        warnings.append(
            "Dominant active current is variable within the "
            "cycle. Verify whether the test protocol uses "
            "constant-current or dynamic loading."
        )

    warning_text = (
        " | ".join(
            warnings
        )
        if warnings
        else "No cycle-level current warning."
    )

    status = (
        "Accepted"
        if point_count
        >= MINIMUM_POINTS_PER_CYCLE
        else "Limited"
    )

    return {
        "cycle": cycle_number,
        "status": status,
        "reason": warning_text,
        "sequence_column": sequence_column,
        "point_count": point_count,
        "minimum_current_a": minimum_current,
        "maximum_current_a": maximum_current,
        "mean_current_a": mean_current,
        "median_current_a": median_current,
        "current_standard_deviation_a": (
            current_standard_deviation
        ),
        "current_range_a": current_range,
        "negative_record_count": (
            negative_record_count
        ),
        "positive_record_count": (
            positive_record_count
        ),
        "rest_record_count": (
            rest_record_count
        ),
        "active_record_count": (
            active_record_count
        ),
        "rest_fraction_percent": (
            rest_fraction_percent
        ),
        "opposite_sign_record_count": (
            opposite_sign_count
        ),
        "dominant_mean_current_a": (
            dominant_mean_current
        ),
        "dominant_absolute_mean_current_a": (
            dominant_absolute_mean
        ),
        "dominant_current_standard_deviation_a": (
            dominant_current_standard_deviation
        ),
        "current_coefficient_of_variation_percent": (
            stability_result[
                "coefficient_of_variation_percent"
            ]
        ),
        "current_stability_category": (
            stability_result[
                "stability_category"
            ]
        ),
        "abrupt_current_change_count_exploratory": (
            abrupt_change_count
        ),
        "maximum_positive_current_change_a_exploratory": (
            maximum_positive_change
        ),
        "maximum_negative_current_change_a_exploratory": (
            maximum_negative_change
        ),
        "maximum_absolute_current_change_a_exploratory": (
            maximum_absolute_change
        ),
        "robust_current_spike_count_exploratory": (
            spike_metrics[
                "robust_spike_count"
            ]
        ),
        "maximum_robust_current_z_score_exploratory": (
            spike_metrics[
                "maximum_robust_z_score"
            ]
        ),
        "current_spikes_used_for_safety": (
            CURRENT_SPIKES_USED_FOR_SAFETY
        ),
        "warning_count": int(
            len(
                warnings
            )
        ),
    }


# ============================================================
# SELECT REPRESENTATIVE CYCLES
# ============================================================

def select_graph_cycles(
    available_cycles,
    maximum_cycles=MAXIMUM_GRAPH_CYCLES,
):
    """
    Select evenly distributed cycles for the graph.
    """

    sorted_cycles = sorted(
        available_cycles
    )

    if (
        len(
            sorted_cycles
        )
        <= maximum_cycles
    ):
        return sorted_cycles

    selected_indices = np.linspace(
        0,
        len(
            sorted_cycles
        )
        - 1,
        maximum_cycles,
        dtype=int,
    )

    return list(
        dict.fromkeys(
            sorted_cycles[index]
            for index in selected_indices
        )
    )


# ============================================================
# RUN CURRENT DIAGNOSTICS
# ============================================================

def analyze_current_diagnostics(dataframe):
    """
    Run current diagnostics across all cycles.
    """

    validated_data = validate_current_dataset(
        dataframe
    )

    sign_information = (
        detect_current_sign_convention(
            validated_data
        )
    )

    (
        active_data,
        active_selection_message,
    ) = select_dominant_active_records(
        validated_data,
        sign_information,
    )

    cycle_records = []

    for cycle_number, cycle_data in (
        validated_data.groupby(
            "cycle",
            sort=True,
        )
    ):
        cycle_records.append(
            analyze_current_cycle(
                cycle_number,
                cycle_data,
                sign_information[
                    "dominant_sign"
                ],
            )
        )

    cycle_summary = pd.DataFrame(
        cycle_records
    )

    if cycle_summary.empty:
        raise ValueError(
            "No cycle-level current results were created."
        )

    accepted_cycles = (
        cycle_summary.loc[
            cycle_summary[
                "status"
            ].isin(
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

    current_values = (
        validated_data[
            "current_a"
        ]
        .dropna()
    )

    stable_cycles = int(
        (
            cycle_summary[
                "current_stability_category"
            ]
            == "Stable"
        ).sum()
    )

    moderately_variable_cycles = int(
        (
            cycle_summary[
                "current_stability_category"
            ]
            == "Moderately variable"
        ).sum()
    )

    variable_cycles = int(
        (
            cycle_summary[
                "current_stability_category"
            ]
            == "Variable"
        ).sum()
    )

    cycles_with_opposite_sign = int(
        (
            cycle_summary[
                "opposite_sign_record_count"
            ]
            > 0
        ).sum()
    )

    analysis_summary = {
        "original_records": int(
            len(
                dataframe
            )
        ),
        "validated_records": int(
            len(
                validated_data
            )
        ),
        "selected_active_records": int(
            len(
                active_data
            )
        ),
        "cycles_found": int(
            validated_data[
                "cycle"
            ].nunique()
        ),
        "cycles_analyzed": int(
            len(
                cycle_summary
            )
        ),
        "cycles_with_warnings": int(
            (
                cycle_summary[
                    "warning_count"
                ]
                > 0
            ).sum()
        ),
        "overall_minimum_current_a": float(
            current_values.min()
        ),
        "overall_maximum_current_a": float(
            current_values.max()
        ),
        "overall_mean_current_a": float(
            current_values.mean()
        ),
        "overall_median_current_a": float(
            current_values.median()
        ),
        "negative_record_count": (
            sign_information[
                "negative_record_count"
            ]
        ),
        "positive_record_count": (
            sign_information[
                "positive_record_count"
            ]
        ),
        "rest_record_count": (
            sign_information[
                "rest_record_count"
            ]
        ),
        "dominant_sign": (
            sign_information[
                "dominant_sign"
            ]
        ),
        "dominant_active_fraction_percent": (
            sign_information[
                "dominant_active_fraction_percent"
            ]
        ),
        "stable_cycles": stable_cycles,
        "moderately_variable_cycles": (
            moderately_variable_cycles
        ),
        "variable_cycles": (
            variable_cycles
        ),
        "cycles_with_opposite_sign_records": (
            cycles_with_opposite_sign
        ),
        "current_spikes_used_for_safety": False,
        "current_spike_exclusion_reason": (
            "Permitted current, nominal capacity, C-rate, "
            "chemistry, and test protocol are not known."
        ),
        "active_selection_message": (
            active_selection_message
        ),
        "graph_cycles": graph_cycles,
    }

    return (
        validated_data,
        active_data,
        cycle_summary,
        analysis_summary,
        sign_information,
    )


# ============================================================
# CREATE CURRENT GRAPH
# ============================================================

def create_current_graph(
    validated_data,
    graph_cycles,
):
    """
    Create representative current-versus-time curves.

    Measurement index is used when elapsed time is unavailable.
    """

    has_time = (
        "time_s"
        in validated_data.columns
        and validated_data[
            "time_s"
        ].notna().any()
    )

    figure, axis = plt.subplots(
        figsize=(13, 8),
    )

    color_map = plt.get_cmap(
        "plasma"
    )

    number_of_cycles = len(
        graph_cycles
    )

    x_label = "Measurement Index"

    for index, cycle_number in enumerate(
        graph_cycles
    ):
        cycle_data = validated_data[
            validated_data[
                "cycle"
            ]
            == cycle_number
        ].copy()

        if has_time:
            cycle_data = cycle_data.dropna(
                subset=[
                    "time_s",
                    "current_a",
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
                    "current_a",
                ]
            )

            x_values = np.arange(
                len(
                    cycle_data
                )
            )

        if cycle_data.empty:
            continue

        if number_of_cycles == 1:
            color_position = 0.5
        else:
            color_position = (
                index
                / (
                    number_of_cycles
                    - 1
                )
            )

        axis.plot(
            x_values,
            cycle_data[
                "current_a"
            ],
            linewidth=1.8,
            color=color_map(
                color_position
            ),
            label=(
                f"Cycle "
                f"{cycle_number:g}"
            ),
        )

    axis.axhline(
        REST_CURRENT_THRESHOLD_A,
        color="#777777",
        linestyle="--",
        linewidth=1.2,
        label=(
            f"Rest boundary "
            f"(+{REST_CURRENT_THRESHOLD_A:.2f} A)"
        ),
    )

    axis.axhline(
        -REST_CURRENT_THRESHOLD_A,
        color="#777777",
        linestyle="--",
        linewidth=1.2,
        label=(
            f"Rest boundary "
            f"(-{REST_CURRENT_THRESHOLD_A:.2f} A)"
        ),
    )

    axis.axhline(
        0,
        color="black",
        linewidth=1,
        alpha=0.7,
    )

    axis.set_title(
        "Battery Current Diagnostics",
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
        "Current (A)",
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
# BUILD CURRENT WARNINGS
# ============================================================

def build_current_warnings(
    cycle_summary,
    analysis_summary,
):
    """
    Create dataset-level current warnings.

    Current magnitude and spike metrics are not treated as
    safety violations because battery specifications and C-rate
    limits are unavailable.
    """

    warnings = []

    if (
        analysis_summary[
            "variable_cycles"
        ]
        > 0
    ):
        warnings.append(
            f"{analysis_summary['variable_cycles']} cycles "
            "have variable dominant active current. This may "
            "be expected for dynamic-load protocols."
        )

    if (
        analysis_summary[
            "cycles_with_opposite_sign_records"
        ]
        > 0
    ):
        warnings.append(
            f"{analysis_summary['cycles_with_opposite_sign_records']} "
            "cycles contain records using the opposite current "
            "sign. These can represent charge, recovery, or "
            "operation transitions."
        )

    limited_cycles = cycle_summary[
        cycle_summary[
            "status"
        ]
        == "Limited"
    ]

    if not limited_cycles.empty:
        warnings.append(
            f"{len(limited_cycles)} cycles contain fewer than "
            f"{MINIMUM_POINTS_PER_CYCLE} current measurements."
        )

    warnings.append(
        "Current magnitudes, abrupt changes, and robust spike "
        "metrics were not used for safety classification because "
        "nominal capacity, permitted C-rate, chemistry, and test "
        "protocol are not known."
    )

    return warnings


# ============================================================
# SAVE CURRENT RESULTS
# ============================================================

def save_current_results(
    validated_data,
    cycle_summary,
    analysis_summary,
    warnings,
    output_directory,
):
    """
    Save current records, summaries, warnings, and graph.
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
        / "current_detailed_records.csv"
    )

    cycle_summary_path = (
        output_directory
        / "current_cycle_summary.csv"
    )

    analysis_summary_path = (
        output_directory
        / "current_analysis_summary.csv"
    )

    warning_path = (
        output_directory
        / "current_warnings.csv"
    )

    graph_path = (
        output_directory
        / "current_diagnostics.png"
    )

    validated_data.to_csv(
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
                str(item)
                for item in value
            )
            if isinstance(
                value,
                list,
            )
            else value
        )
        for key, value
        in analysis_summary.items()
    }

    pd.DataFrame(
        [
            summary_for_csv
        ]
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

    figure = create_current_graph(
        validated_data,
        analysis_summary[
            "graph_cycles"
        ],
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
                str(
                    graph_path
                )
            )

    except Exception as open_error:
        print(
            "\nThe current graph was saved successfully, "
            "but it could not be opened automatically."
        )

        print(
            f"Opening error: "
            f"{open_error}"
        )

    return {
        "detailed_records": (
            detailed_path
        ),
        "cycle_summary": (
            cycle_summary_path
        ),
        "analysis_summary": (
            analysis_summary_path
        ),
        "warnings": warning_path,
        "graph": graph_path,
    }


# ============================================================
# PRINT CURRENT REPORT
# ============================================================

def print_current_report(
    analysis_summary,
    sign_information,
    warnings,
    saved_files,
    unknown_columns,
):
    """
    Print a readable current diagnostics report.
    """

    print(
        "\n"
        + "="
        * 75
    )

    print(
        "BATTERY CURRENT DIAGNOSTICS"
    )

    print(
        "="
        * 75
    )

    print(
        f"\nOriginal records: "
        f"{analysis_summary['original_records']:,}"
    )

    print(
        f"Validated records: "
        f"{analysis_summary['validated_records']:,}"
    )

    print(
        f"Selected dominant active records: "
        f"{analysis_summary['selected_active_records']:,}"
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
        f"Cycles with analytical warnings: "
        f"{analysis_summary['cycles_with_warnings']}"
    )

    print(
        "\nOVERALL CURRENT RANGE"
    )

    print(
        f"  Minimum current: "
        f"{analysis_summary['overall_minimum_current_a']:.4f} A"
    )

    print(
        f"  Maximum current: "
        f"{analysis_summary['overall_maximum_current_a']:.4f} A"
    )

    print(
        f"  Mean current: "
        f"{analysis_summary['overall_mean_current_a']:.4f} A"
    )

    print(
        f"  Median current: "
        f"{analysis_summary['overall_median_current_a']:.4f} A"
    )

    print(
        "\nCURRENT DIRECTION"
    )

    print(
        f"  Negative records: "
        f"{sign_information['negative_record_count']:,}"
    )

    print(
        f"  Positive records: "
        f"{sign_information['positive_record_count']:,}"
    )

    print(
        f"  Near-zero/rest records: "
        f"{sign_information['rest_record_count']:,}"
    )

    print(
        f"  Dominant sign: "
        f"{sign_information['dominant_sign']}"
    )

    print(
        f"  Dominant fraction of active records: "
        f"{sign_information['dominant_active_fraction_percent']:.2f}%"
    )

    print(
        "\nCURRENT STABILITY"
    )

    print(
        f"  Stable cycles: "
        f"{analysis_summary['stable_cycles']}"
    )

    print(
        f"  Moderately variable cycles: "
        f"{analysis_summary['moderately_variable_cycles']}"
    )

    print(
        f"  Variable cycles: "
        f"{analysis_summary['variable_cycles']}"
    )

    print(
        f"  Cycles with opposite-sign records: "
        f"{analysis_summary['cycles_with_opposite_sign_records']}"
    )

    print(
        "\nCURRENT SPIKE METRIC STATUS"
    )

    print(
        "  Used for safety classification: No"
    )

    print(
        "  Reason: "
        f"{analysis_summary['current_spike_exclusion_reason']}"
    )

    print(
        "\nWARNINGS"
    )

    for warning in warnings:
        print(
            f"  - {warning}"
        )

    if unknown_columns:
        print(
            "\nUNRECOGNIZED ORIGINAL COLUMNS"
        )

        for column in unknown_columns:
            print(
                f"  - {column}"
            )

    print(
        "\nSAVED RESULTS"
    )

    for result_name, result_path in (
        saved_files.items()
    ):
        print(
            f"  {result_name}: "
            f"{result_path}"
        )

    print(
        "\nIMPORTANT"
    )

    print(
        "  Current sign conventions vary between datasets. "
        "The dominant sign is detected statistically and must "
        "be confirmed against dataset documentation."
    )

    print(
        "  Current stability categories describe signal "
        "variation. They do not automatically indicate battery "
        "health or a safety condition."
    )

    print(
        "  Current magnitude cannot be judged safely without "
        "nominal capacity, permitted C-rate, chemistry, cell "
        "specifications, and test protocol."
    )

    print(
        "="
        * 75
    )


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
            active_data,
            current_cycle_summary,
            current_summary,
            sign_info,
        ) = analyze_current_diagnostics(
            standardized_data
        )

        current_warnings = build_current_warnings(
            current_cycle_summary,
            current_summary,
        )

        output_folder = (
            PLATFORM_FOLDER
            / "outputs"
            / Path(dataset_path).stem
            / "current_diagnostics"
        )

        saved_results = save_current_results(
            validated_data,
            current_cycle_summary,
            current_summary,
            current_warnings,
            output_folder,
        )

        print_current_report(
            current_summary,
            sign_info,
            current_warnings,
            saved_results,
            unknown_columns,
        )

    except Exception as error:
        print(
            f"\nCurrent diagnostics failed: "
            f"{error}"
        )