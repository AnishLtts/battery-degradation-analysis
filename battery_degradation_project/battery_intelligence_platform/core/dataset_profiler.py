from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
CORE_FOLDER = CURRENT_FILE.parent
PLATFORM_FOLDER = CORE_FOLDER.parent

DIAGNOSTICS_FOLDER = (
    PLATFORM_FOLDER
    / "analytics"
    / "diagnostics"
)

sys.path.insert(
    0,
    str(CORE_FOLDER),
)

sys.path.insert(
    0,
    str(DIAGNOSTICS_FOLDER),
)


# ============================================================
# CORE IMPORTS
# ============================================================

from file_reader import load_battery_dataset

from column_mapper import (
    apply_column_mapping,
    detect_column_mapping,
)

from unit_normalizer import (
    normalize_dataset_units,
    validate_normalized_ranges,
    save_normalized_results,
)


# ============================================================
# DIAGNOSTIC IMPORTS
# ============================================================

from voltage_diagnostics import (
    analyze_voltage_diagnostics,
    build_voltage_warnings,
    save_voltage_results,
)

from current_diagnostics import (
    analyze_current_diagnostics,
    build_current_warnings,
    save_current_results,
)

from temperature_diagnostics import (
    run_temperature_diagnostics,
)

from capacity_diagnostics import (
    run_capacity_diagnostics,
)


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_COLUMNS = [
    "battery_id",
    "cycle",
    "time_s",
    "voltage_v",
    "current_a",
    "temperature_c",
    "capacity_ah",
]

NUMERIC_COLUMNS = [
    "cycle",
    "time_s",
    "voltage_v",
    "current_a",
    "temperature_c",
    "capacity_ah",
]


# ============================================================
# NUMERIC CONVERSION
# ============================================================

def convert_numeric_columns(dataframe):
    """
    Convert recognized battery measurement columns to numeric
    values. Invalid values become missing values.
    """
    converted_data = dataframe.copy()

    for column in NUMERIC_COLUMNS:
        if column in converted_data.columns:
            converted_data[column] = pd.to_numeric(
                converted_data[column],
                errors="coerce",
            )

    return converted_data


# ============================================================
# DETECT POSSIBLE UNITS
# ============================================================

def detect_suspected_units(dataframe):
    """
    Estimate measurement units using engineering value ranges.

    These are warnings and estimates only. Unit conversion is
    handled separately by unit_normalizer.py when the unit is
    explicit in the original column name.
    """
    detected_units = {}
    warnings = []

    # Voltage
    if "voltage_v" in dataframe.columns:
        values = dataframe[
            "voltage_v"
        ].dropna()

        if values.empty:
            detected_units["voltage"] = "Unknown"

        else:
            median_value = float(
                values.abs().median()
            )

            maximum_value = float(
                values.abs().max()
            )

            if 500 <= median_value <= 6000:
                detected_units["voltage"] = (
                    "Likely millivolts (mV)"
                )

                warnings.append(
                    "Voltage values may still be in millivolts."
                )

            elif 0.5 <= median_value <= 10:
                detected_units["voltage"] = (
                    "Likely volts (V)"
                )

            elif maximum_value > 10:
                detected_units["voltage"] = (
                    "Unknown or pack-level voltage"
                )

                warnings.append(
                    "Voltage is above a typical single-cell "
                    "range. Confirm whether the dataset contains "
                    "pack-level voltage."
                )

            else:
                detected_units["voltage"] = "Unknown"

                warnings.append(
                    "Voltage units could not be identified."
                )

    else:
        detected_units["voltage"] = "Unavailable"

    # Current
    if "current_a" in dataframe.columns:
        values = dataframe[
            "current_a"
        ].dropna()

        active_values = values[
            values.abs() > 0.0001
        ]

        if active_values.empty:
            detected_units["current"] = "Unknown"

        else:
            median_value = float(
                active_values.abs().median()
            )

            if median_value >= 100:
                detected_units["current"] = (
                    "Likely milliamperes (mA)"
                )

                warnings.append(
                    "Current values may still be in "
                    "milliamperes."
                )

            elif median_value <= 50:
                detected_units["current"] = (
                    "Likely amperes (A)"
                )

            else:
                detected_units["current"] = (
                    "Unknown or high-current system"
                )

                warnings.append(
                    "Current units or system scale require "
                    "confirmation."
                )

    else:
        detected_units["current"] = "Unavailable"

    # Capacity
    if "capacity_ah" in dataframe.columns:
        values = dataframe[
            "capacity_ah"
        ].dropna()

        positive_values = values[
            values > 0
        ]

        if positive_values.empty:
            detected_units["capacity"] = "Unknown"

        else:
            median_value = float(
                positive_values.median()
            )

            maximum_value = float(
                positive_values.max()
            )

            if median_value >= 100:
                detected_units["capacity"] = (
                    "Likely milliampere-hours (mAh)"
                )

                warnings.append(
                    "Capacity values may still be in mAh."
                )

            elif maximum_value <= 20:
                detected_units["capacity"] = (
                    "Likely ampere-hours (Ah)"
                )

            else:
                detected_units["capacity"] = (
                    "Unknown or pack-level capacity"
                )

                warnings.append(
                    "Capacity units or battery scale require "
                    "confirmation."
                )

    else:
        detected_units["capacity"] = "Unavailable"

    # Time
    if "time_s" in dataframe.columns:
        values = dataframe[
            "time_s"
        ].dropna()

        if values.empty:
            detected_units["time"] = "Unknown"

        else:
            detected_units["time"] = (
                "Assumed seconds (s)"
            )

            warnings.append(
                "Time is assumed to be in seconds. Confirm this "
                "using the source dataset documentation."
            )

    else:
        detected_units["time"] = "Unavailable"

    # Temperature
    if "temperature_c" in dataframe.columns:
        values = dataframe[
            "temperature_c"
        ].dropna()

        if values.empty:
            detected_units["temperature"] = "Unknown"

        else:
            median_value = float(
                values.median()
            )

            if 200 <= median_value <= 400:
                detected_units["temperature"] = (
                    "Likely kelvin (K)"
                )

                warnings.append(
                    "Temperature values may still be in kelvin."
                )

            elif -50 <= median_value <= 150:
                detected_units["temperature"] = (
                    "Likely degrees Celsius"
                )

            else:
                detected_units["temperature"] = "Unknown"

                warnings.append(
                    "Temperature units or values appear unusual."
                )

    else:
        detected_units["temperature"] = "Unavailable"

    return detected_units, warnings


# ============================================================
# DATA QUALITY
# ============================================================

def profile_data_quality(dataframe):
    """
    Measure missing values, duplicate rows, and suspicious
    engineering values.
    """
    row_count = int(
        len(dataframe)
    )

    column_count = int(
        len(dataframe.columns)
    )

    duplicate_rows = int(
        dataframe.duplicated().sum()
    )

    total_cells = (
        row_count * column_count
    )

    total_missing = int(
        dataframe.isna().sum().sum()
    )

    if total_cells > 0:
        missing_percentage = (
            total_missing
            / total_cells
            * 100
        )

    else:
        missing_percentage = 0.0

    missing_by_column = {
        column: {
            "missing_count": int(
                dataframe[column].isna().sum()
            ),
            "missing_percent": float(
                dataframe[column].isna().mean()
                * 100
            ),
        }
        for column in dataframe.columns
    }

    invalid_counts = {}

    if "voltage_v" in dataframe.columns:
        invalid_counts[
            "nonpositive_voltage"
        ] = int(
            (
                dataframe["voltage_v"]
                <= 0
            ).sum()
        )

    if "capacity_ah" in dataframe.columns:
        invalid_counts[
            "negative_capacity"
        ] = int(
            (
                dataframe["capacity_ah"]
                < 0
            ).sum()
        )

    if "time_s" in dataframe.columns:
        invalid_counts[
            "negative_time"
        ] = int(
            (
                dataframe["time_s"]
                < 0
            ).sum()
        )

    if "temperature_c" in dataframe.columns:
        invalid_counts[
            "temperature_below_minus_50"
        ] = int(
            (
                dataframe["temperature_c"]
                < -50
            ).sum()
        )

        invalid_counts[
            "temperature_above_150"
        ] = int(
            (
                dataframe["temperature_c"]
                > 150
            ).sum()
        )

    return {
        "rows": row_count,
        "columns": column_count,
        "duplicate_rows": duplicate_rows,
        "total_missing_values": total_missing,
        "missing_percentage": float(
            missing_percentage
        ),
        "missing_by_column": missing_by_column,
        "invalid_counts": invalid_counts,
    }


# ============================================================
# DATASET STRUCTURE
# ============================================================

def detect_dataset_structure(dataframe):
    """
    Detect the dataset format, cycle count, battery count, and
    current sign behaviour.
    """
    structure = {
        "dataset_type": "Unknown",
        "battery_scope": "Unknown",
        "battery_count": None,
        "cycle_count": None,
        "operation_type": "Unknown",
        "current_sign_convention": "Unknown",
    }

    # Battery scope
    if "battery_id" in dataframe.columns:
        battery_count = int(
            dataframe["battery_id"]
            .dropna()
            .nunique()
        )

        structure["battery_count"] = battery_count

        if battery_count > 1:
            structure["battery_scope"] = (
                "Multiple batteries"
            )

        elif battery_count == 1:
            structure["battery_scope"] = (
                "Single battery"
            )

    else:
        structure["battery_scope"] = (
            "Battery identifier unavailable"
        )

    # Cycle count
    if "cycle" in dataframe.columns:
        structure["cycle_count"] = int(
            dataframe["cycle"]
            .dropna()
            .nunique()
        )

    has_time_series = all(
        column in dataframe.columns
        for column in [
            "time_s",
            "voltage_v",
            "current_a",
        ]
    )

    has_cycle = (
        "cycle" in dataframe.columns
    )

    has_capacity = (
        "capacity_ah" in dataframe.columns
    )

    if has_time_series and has_cycle:
        structure["dataset_type"] = (
            "Cycle-based time-series"
        )

    elif has_cycle and has_capacity:
        rows_per_cycle = (
            dataframe.groupby("cycle")
            .size()
            .median()
        )

        if rows_per_cycle <= 3:
            structure["dataset_type"] = (
                "Cycle-summary"
            )

        else:
            structure["dataset_type"] = (
                "Cycle-based measurements"
            )

    elif has_time_series:
        structure["dataset_type"] = (
            "Time-series without cycle index"
        )

    elif has_capacity:
        structure["dataset_type"] = (
            "Capacity dataset"
        )

    # Current sign behaviour
    if "current_a" in dataframe.columns:
        current_values = dataframe[
            "current_a"
        ].dropna()

        negative_count = int(
            (current_values < -0.001).sum()
        )

        positive_count = int(
            (current_values > 0.001).sum()
        )

        if negative_count > 0 and positive_count > 0:
            structure["operation_type"] = (
                "Mixed charge and discharge"
            )

            if negative_count > positive_count:
                structure[
                    "current_sign_convention"
                ] = (
                    "Negative current likely represents discharge"
                )

            else:
                structure[
                    "current_sign_convention"
                ] = (
                    "Positive current likely represents discharge"
                )

        elif negative_count > 0:
            structure["operation_type"] = (
                "Likely discharge"
            )

            structure[
                "current_sign_convention"
            ] = (
                "Negative current represents active operation"
            )

        elif positive_count > 0:
            structure["operation_type"] = (
                "Likely charge or positive-current discharge"
            )

            structure[
                "current_sign_convention"
            ] = (
                "Positive current represents active operation"
            )

        else:
            structure["operation_type"] = (
                "No active current detected"
            )

    return structure


# ============================================================
# NUMERIC STATISTICS
# ============================================================

def calculate_column_statistics(dataframe):
    """
    Calculate basic statistics for standardized numeric columns.
    """
    statistics = {}

    for column in NUMERIC_COLUMNS:
        if column not in dataframe.columns:
            continue

        values = dataframe[
            column
        ].dropna()

        if values.empty:
            statistics[column] = {
                "count": 0,
                "minimum": None,
                "maximum": None,
                "mean": None,
                "median": None,
            }

            continue

        statistics[column] = {
            "count": int(
                values.count()
            ),
            "minimum": float(
                values.min()
            ),
            "maximum": float(
                values.max()
            ),
            "mean": float(
                values.mean()
            ),
            "median": float(
                values.median()
            ),
        }

    return statistics


# ============================================================
# ANALYSIS CAPABILITIES
# ============================================================

def detect_analysis_capabilities(dataframe):
    """
    Determine which analyses can run from the available columns.
    """
    available_columns = set(
        dataframe.columns
    )

    capability_requirements = {
        "soh": [
            "capacity_ah",
        ],
        "degradation": [
            "cycle",
            "capacity_ah",
        ],
        "soc": [
            "time_s",
            "current_a",
            "capacity_ah",
        ],
        "ica": [
            "cycle",
            "voltage_v",
            "capacity_ah",
        ],
        "dva": [
            "cycle",
            "voltage_v",
            "capacity_ah",
        ],
        "voltage_diagnostics": [
            "cycle",
            "voltage_v",
        ],
        "current_diagnostics": [
            "cycle",
            "current_a",
        ],
        "temperature_diagnostics": [
            "temperature_c",
        ],
        "capacity_diagnostics": [
            "cycle",
            "capacity_ah",
        ],
    }

    capabilities = {}

    for analysis_name, requirements in (
        capability_requirements.items()
    ):
        missing_columns = [
            column
            for column in requirements
            if column not in available_columns
        ]

        capabilities[analysis_name] = {
            "available": (
                len(missing_columns) == 0
            ),
            "requirements": requirements,
            "missing_columns": missing_columns,
        }

    return capabilities


# ============================================================
# DATA READINESS
# ============================================================

def calculate_reliability(
    quality_report,
    capabilities,
    unit_warnings,
):
    """
    Calculate a data-readiness score.

    The result is not an ML accuracy score.
    """
    score = 100
    reasons = []

    missing_percentage = quality_report[
        "missing_percentage"
    ]

    if missing_percentage > 20:
        score -= 35

        reasons.append(
            "More than 20% of dataset values are missing."
        )

    elif missing_percentage > 5:
        score -= 15

        reasons.append(
            "More than 5% of dataset values are missing."
        )

    duplicate_rows = quality_report[
        "duplicate_rows"
    ]

    row_count = max(
        quality_report["rows"],
        1,
    )

    duplicate_percentage = (
        duplicate_rows
        / row_count
        * 100
    )

    if duplicate_percentage > 10:
        score -= 20

        reasons.append(
            "More than 10% of rows are duplicated."
        )

    elif duplicate_percentage > 1:
        score -= 5

        reasons.append(
            "More than 1% of rows are duplicated."
        )

    invalid_total = sum(
        quality_report[
            "invalid_counts"
        ].values()
    )

    if invalid_total > 0:
        score -= min(
            20,
            invalid_total,
        )

        reasons.append(
            "Potentially invalid engineering values were found."
        )

    if unit_warnings:
        score -= min(
            20,
            len(unit_warnings) * 5,
        )

        reasons.append(
            "One or more measurement units require confirmation."
        )

    available_count = sum(
        1
        for capability in capabilities.values()
        if capability["available"]
    )

    if available_count == 0:
        score -= 30

        reasons.append(
            "No supported analysis is available."
        )

    score = max(
        0,
        min(100, score),
    )

    if score >= 85:
        level = "High"

    elif score >= 65:
        level = "Moderate"

    elif score >= 40:
        level = "Limited"

    else:
        level = "Low"

    if not reasons:
        reasons.append(
            "No major data-readiness problems were detected."
        )

    return {
        "score": int(score),
        "level": level,
        "reasons": reasons,
    }


# ============================================================
# BUILD PROFILE
# ============================================================

def build_dataset_profile(dataframe):
    """
    Build the complete profile from standardized and normalized
    data.
    """
    numeric_data = convert_numeric_columns(
        dataframe
    )

    quality_report = profile_data_quality(
        numeric_data
    )

    structure_report = detect_dataset_structure(
        numeric_data
    )

    detected_units, unit_warnings = (
        detect_suspected_units(
            numeric_data
        )
    )

    statistics = calculate_column_statistics(
        numeric_data
    )

    capabilities = detect_analysis_capabilities(
        numeric_data
    )

    reliability = calculate_reliability(
        quality_report,
        capabilities,
        unit_warnings,
    )

    profile = {
        "quality": quality_report,
        "structure": structure_report,
        "detected_units": detected_units,
        "unit_warnings": unit_warnings,
        "statistics": statistics,
        "capabilities": capabilities,
        "reliability": reliability,
    }

    return profile, numeric_data


# ============================================================
# PRINT NORMALIZATION SUMMARY
# ============================================================

def print_integrated_normalization_summary(
    original_dataset_path,
    normalized_dataset_path,
    normalization_summary,
    normalization_warnings,
):
    """
    Print a concise normalization summary before profiling.
    """
    print("\n" + "=" * 75)
    print("UNIT NORMALIZATION")
    print("=" * 75)

    print(
        f"\nOriginal dataset:\n"
        f"{original_dataset_path}"
    )

    print(
        f"\nNormalized dataset:\n"
        f"{normalized_dataset_path}"
    )

    print(
        "\nAlready normalized: "
        f"{normalization_summary.get('already_normalized', False)}"
    )

    print(
        "Columns converted: "
        f"{normalization_summary.get('columns_converted', 0)}"
    )

    print(
        "Columns standardized: "
        f"{normalization_summary.get('columns_standardized', 0)}"
    )

    print(
        "Columns unchanged: "
        f"{normalization_summary.get('columns_unchanged', 0)}"
    )

    if normalization_warnings:
        print("\nNORMALIZATION WARNINGS")

        for warning in normalization_warnings:
            print(
                f"  - {warning}"
            )

    else:
        print(
            "\nNo unit-normalization warning was generated."
        )

    print("=" * 75)


# ============================================================
# PRINT DATASET PROFILE
# ============================================================

def print_dataset_profile(
    dataset_path,
    original_columns,
    mapping,
    unknown_columns,
    profile,
):
    """
    Print the complete normalized dataset profile.
    """
    print("\n" + "=" * 75)
    print("BATTERY DATASET PROFILE")
    print("=" * 75)

    print(
        f"\nDataset used for profiling:\n"
        f"{dataset_path}"
    )

    print(
        f"\nRows: "
        f"{profile['quality']['rows']:,}"
    )

    print(
        f"Columns: "
        f"{profile['quality']['columns']}"
    )

    print("\nCOLUMNS BEFORE STANDARD MAPPING")

    for column in original_columns:
        print(
            f"  - {column}"
        )

    print("\nDETECTED COLUMN MAPPING")

    if mapping:
        for original_column, standard_column in (
            mapping.items()
        ):
            print(
                f"  {original_column} -> "
                f"{standard_column}"
            )

    else:
        print(
            "  No additional column mapping was required."
        )

    if unknown_columns:
        print("\nUNRECOGNIZED COLUMNS")

        for column in unknown_columns:
            print(
                f"  - {column}"
            )

    structure = profile["structure"]

    print("\nDATASET STRUCTURE")

    print(
        f"  Dataset type: "
        f"{structure['dataset_type']}"
    )

    print(
        f"  Battery scope: "
        f"{structure['battery_scope']}"
    )

    print(
        f"  Battery count: "
        f"{structure['battery_count']}"
    )

    print(
        f"  Cycle count: "
        f"{structure['cycle_count']}"
    )

    print(
        f"  Operation type: "
        f"{structure['operation_type']}"
    )

    print(
        f"  Current convention: "
        f"{structure['current_sign_convention']}"
    )

    print("\nSUSPECTED UNITS AFTER NORMALIZATION")

    for measurement, unit in (
        profile["detected_units"].items()
    ):
        print(
            f"  {measurement}: {unit}"
        )

    if profile["unit_warnings"]:
        print("\nUNIT WARNINGS")

        for warning in profile[
            "unit_warnings"
        ]:
            print(
                f"  - {warning}"
            )

    quality = profile["quality"]

    print("\nDATA QUALITY")

    print(
        f"  Missing values: "
        f"{quality['total_missing_values']:,}"
    )

    print(
        f"  Missing percentage: "
        f"{quality['missing_percentage']:.2f}%"
    )

    print(
        f"  Duplicate rows: "
        f"{quality['duplicate_rows']:,}"
    )

    if quality["invalid_counts"]:
        print(
            "  Engineering checks:"
        )

        for check_name, count in (
            quality["invalid_counts"].items()
        ):
            print(
                f"    {check_name}: {count:,}"
            )

    print("\nNUMERIC RANGES")

    for column, statistics in (
        profile["statistics"].items()
    ):
        if statistics["count"] == 0:
            print(
                f"  {column}: no valid values"
            )

            continue

        print(
            f"  {column}: "
            f"min={statistics['minimum']:.4f}, "
            f"max={statistics['maximum']:.4f}, "
            f"median={statistics['median']:.4f}"
        )

    print("\nAVAILABLE ANALYSES")

    for analysis_name, capability in (
        profile["capabilities"].items()
    ):
        if capability["available"]:
            print(
                f"  {analysis_name.upper()}: Available"
            )

        else:
            missing_text = ", ".join(
                capability["missing_columns"]
            )

            print(
                f"  {analysis_name.upper()}: "
                f"Not available "
                f"(missing: {missing_text})"
            )

    reliability = profile["reliability"]

    print("\nDATA READINESS")

    print(
        f"  Score: "
        f"{reliability['score']}/100"
    )

    print(
        f"  Level: "
        f"{reliability['level']}"
    )

    print(
        "  Reasons:"
    )

    for reason in reliability["reasons"]:
        print(
            f"    - {reason}"
        )

    print("\nIMPORTANT")

    print(
        "  Unit conversions are applied only when the source "
        "unit is explicit in the column name."
    )

    print(
        "  Values inferred only from ranges are reported as "
        "warnings and are not silently converted."
    )

    print(
        "  The readiness score measures data quality and "
        "analysis compatibility. It is not ML model accuracy."
    )

    print("=" * 75)


# ============================================================
# OUTPUT ROOT
# ============================================================

def get_diagnostics_output_root(
    normalized_dataset_path,
):
    """
    Return the main dataset output directory.
    """
    resolved_path = Path(
        normalized_dataset_path
    ).expanduser().resolve()

    if (
        resolved_path.name.lower()
        == "normalized_battery_data.csv"
        and resolved_path.parent.name.lower()
        == "unit_normalization"
    ):
        return resolved_path.parent.parent

    return (
        PLATFORM_FOLDER
        / "outputs"
        / resolved_path.stem
    )


# ============================================================
# AUTOMATIC DIAGNOSTICS
# ============================================================

def run_automatic_diagnostics(
    normalized_dataset_path,
    standardized_data,
):
    """
    Run voltage, current, temperature, and capacity diagnostics.

    One failed diagnostic does not stop the remaining modules.
    """
    print("\n" + "=" * 75)
    print("AUTOMATIC BATTERY DIAGNOSTICS")
    print("=" * 75)

    output_root = get_diagnostics_output_root(
        normalized_dataset_path
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    available_columns = set(
        standardized_data.columns
    )

    diagnostic_results = {}

    # Voltage diagnostics
    required_columns = {
        "cycle",
        "voltage_v",
    }

    missing_columns = sorted(
        required_columns
        - available_columns
    )

    if missing_columns:
        diagnostic_results["voltage"] = {
            "status": "Skipped",
            "details": (
                "Missing columns: "
                + ", ".join(missing_columns)
            ),
        }

        print(
            "\n[VOLTAGE] Skipped. Missing columns: "
            + ", ".join(missing_columns)
        )

    else:
        try:
            print(
                "\n[VOLTAGE] Running..."
            )

            (
                validated_voltage_data,
                discharge_data,
                voltage_cycle_summary,
                voltage_summary,
                discharge_information,
            ) = analyze_voltage_diagnostics(
                standardized_data.copy()
            )

            voltage_warnings = (
                build_voltage_warnings(
                    voltage_cycle_summary,
                    voltage_summary,
                )
            )

            voltage_output_folder = (
                output_root
                / "voltage_diagnostics"
            )

            saved_files = save_voltage_results(
                discharge_data,
                voltage_cycle_summary,
                voltage_summary,
                voltage_warnings,
                voltage_output_folder,
            )

            diagnostic_results["voltage"] = {
                "status": "Completed",
                "output_folder": str(
                    voltage_output_folder
                ),
                "files": saved_files,
            }

            print(
                "[VOLTAGE] Completed successfully."
            )

            print(
                f"[VOLTAGE] Output: "
                f"{voltage_output_folder}"
            )

        except Exception as error:
            diagnostic_results["voltage"] = {
                "status": "Failed",
                "details": str(error),
            }

            print(
                f"[VOLTAGE] Failed: {error}"
            )

    # Current diagnostics
    required_columns = {
        "cycle",
        "current_a",
    }

    missing_columns = sorted(
        required_columns
        - available_columns
    )

    if missing_columns:
        diagnostic_results["current"] = {
            "status": "Skipped",
            "details": (
                "Missing columns: "
                + ", ".join(missing_columns)
            ),
        }

        print(
            "\n[CURRENT] Skipped. Missing columns: "
            + ", ".join(missing_columns)
        )

    else:
        try:
            print(
                "\n[CURRENT] Running..."
            )

            (
                validated_current_data,
                active_current_data,
                current_cycle_summary,
                current_summary,
                sign_information,
            ) = analyze_current_diagnostics(
                standardized_data.copy()
            )

            current_warnings = (
                build_current_warnings(
                    current_cycle_summary,
                    current_summary,
                )
            )

            current_output_folder = (
                output_root
                / "current_diagnostics"
            )

            saved_files = save_current_results(
                validated_current_data,
                current_cycle_summary,
                current_summary,
                current_warnings,
                current_output_folder,
            )

            diagnostic_results["current"] = {
                "status": "Completed",
                "output_folder": str(
                    current_output_folder
                ),
                "files": saved_files,
            }

            print(
                "[CURRENT] Completed successfully."
            )

            print(
                f"[CURRENT] Output: "
                f"{current_output_folder}"
            )

        except Exception as error:
            diagnostic_results["current"] = {
                "status": "Failed",
                "details": str(error),
            }

            print(
                f"[CURRENT] Failed: {error}"
            )

    # Temperature diagnostics
    required_columns = {
        "temperature_c",
    }

    missing_columns = sorted(
        required_columns
        - available_columns
    )

    if missing_columns:
        diagnostic_results["temperature"] = {
            "status": "Skipped",
            "details": (
                "Missing columns: "
                + ", ".join(missing_columns)
            ),
        }

        print(
            "\n[TEMPERATURE] Skipped. Missing columns: "
            + ", ".join(missing_columns)
        )

    else:
        try:
            print(
                "\n[TEMPERATURE] Running..."
            )

            temperature_results = (
                run_temperature_diagnostics(
                    normalized_dataset_path
                )
            )

            diagnostic_results["temperature"] = {
                "status": "Completed",
                "output_folder": str(
                    temperature_results[
                        "output_directory"
                    ]
                ),
                "files": temperature_results[
                    "output_files"
                ],
            }

            print(
                "[TEMPERATURE] Completed successfully."
            )

            print(
                "[TEMPERATURE] Output: "
                f"{temperature_results['output_directory']}"
            )

        except Exception as error:
            diagnostic_results["temperature"] = {
                "status": "Failed",
                "details": str(error),
            }

            print(
                f"[TEMPERATURE] Failed: {error}"
            )

    # Capacity diagnostics
    required_columns = {
        "cycle",
        "capacity_ah",
    }

    missing_columns = sorted(
        required_columns
        - available_columns
    )

    if missing_columns:
        diagnostic_results["capacity"] = {
            "status": "Skipped",
            "details": (
                "Missing columns: "
                + ", ".join(missing_columns)
            ),
        }

        print(
            "\n[CAPACITY] Skipped. Missing columns: "
            + ", ".join(missing_columns)
        )

    else:
        try:
            print(
                "\n[CAPACITY] Running..."
            )

            capacity_results = (
                run_capacity_diagnostics(
                    normalized_dataset_path
                )
            )

            diagnostic_results["capacity"] = {
                "status": "Completed",
                "output_folder": str(
                    capacity_results[
                        "output_directory"
                    ]
                ),
                "files": capacity_results[
                    "output_files"
                ],
            }

            print(
                "[CAPACITY] Completed successfully."
            )

            print(
                "[CAPACITY] Output: "
                f"{capacity_results['output_directory']}"
            )

        except Exception as error:
            diagnostic_results["capacity"] = {
                "status": "Failed",
                "details": str(error),
            }

            print(
                f"[CAPACITY] Failed: {error}"
            )

    completed_count = sum(
        result["status"] == "Completed"
        for result in diagnostic_results.values()
    )

    skipped_count = sum(
        result["status"] == "Skipped"
        for result in diagnostic_results.values()
    )

    failed_count = sum(
        result["status"] == "Failed"
        for result in diagnostic_results.values()
    )

    print("\n" + "-" * 75)
    print("DIAGNOSTIC EXECUTION SUMMARY")
    print("-" * 75)

    for diagnostic_name, result in (
        diagnostic_results.items()
    ):
        print(
            f"  {diagnostic_name.upper()}: "
            f"{result['status']}"
        )

        if result["status"] != "Completed":
            print(
                f"    Reason: "
                f"{result.get('details', 'Unknown')}"
            )

    print(
        f"\nCompleted: {completed_count}"
    )

    print(
        f"Skipped: {skipped_count}"
    )

    print(
        f"Failed: {failed_count}"
    )

    print(
        f"\nMain output folder:\n"
        f"{output_root}"
    )

    print("=" * 75)

    return diagnostic_results


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def run_complete_pipeline(dataset_path):
    """
    Run the complete safe battery dataset workflow.

    Flow for a raw dataset:
        Load raw data
        Normalize explicit units
        Save normalized CSV
        Map standardized columns
        Profile normalized data
        Run supported diagnostics

    Flow for an already-normalized dataset:
        Load normalized data
        Detect the normalization marker
        Reuse the same normalized CSV
        Do not create a duplicate output folder
        Profile the data
        Run supported diagnostics
    """
    original_dataset_path = Path(
        dataset_path
    ).expanduser().resolve()

    original_data = load_battery_dataset(
        original_dataset_path
    )

    (
        normalized_data,
        conversion_audit,
        normalization_warnings,
        normalization_summary,
    ) = normalize_dataset_units(
        original_data
    )

    range_warnings = validate_normalized_ranges(
        normalized_data
    )

    all_normalization_warnings = (
        normalization_warnings
        + range_warnings
    )

    normalization_summary[
        "warning_count"
    ] = len(
        all_normalization_warnings
    )

    already_normalized = bool(
        normalization_summary.get(
            "already_normalized",
            False,
        )
    )

    # --------------------------------------------------------
    # REUSE AN ALREADY-NORMALIZED DATASET
    # --------------------------------------------------------

    if already_normalized:
        normalized_dataset_path = (
            original_dataset_path
        )

        existing_output_directory = (
            normalized_dataset_path.parent
        )

        saved_normalization_files = {
            "normalized_data": (
                normalized_dataset_path
            ),
            "conversion_audit": (
                existing_output_directory
                / "unit_conversion_audit.csv"
            ),
            "warnings": (
                existing_output_directory
                / "unit_normalization_warnings.csv"
            ),
            "summary": (
                existing_output_directory
                / "unit_normalization_summary.csv"
            ),
        }

        print(
            "\nExisting unit-normalization marker detected."
        )

        print(
            "The existing normalized dataset will be reused."
        )

        print(
            "No duplicate normalized dataset folder "
            "will be created."
        )

    # --------------------------------------------------------
    # SAVE A NEWLY-NORMALIZED RAW DATASET
    # --------------------------------------------------------

    else:
        saved_normalization_files = (
            save_normalized_results(
                normalized_data,
                conversion_audit,
                all_normalization_warnings,
                normalization_summary,
                original_dataset_path,
            )
        )

        normalized_dataset_path = Path(
            saved_normalization_files[
                "normalized_data"
            ]
        ).resolve()

    # --------------------------------------------------------
    # LOAD DATA USED BY THE PROFILER AND DIAGNOSTICS
    # --------------------------------------------------------

    normalized_data_for_mapping = (
        load_battery_dataset(
            normalized_dataset_path
        )
    )

    columns_before_mapping = list(
        normalized_data_for_mapping.columns
    )

    mapping, unknown_columns = (
        detect_column_mapping(
            normalized_data_for_mapping
        )
    )

    standardized_data = apply_column_mapping(
        normalized_data_for_mapping,
        mapping,
    )

    dataset_profile, profiled_data = (
        build_dataset_profile(
            standardized_data
        )
    )

    print_integrated_normalization_summary(
        original_dataset_path,
        normalized_dataset_path,
        normalization_summary,
        all_normalization_warnings,
    )

    print_dataset_profile(
        normalized_dataset_path,
        columns_before_mapping,
        mapping,
        unknown_columns,
        dataset_profile,
    )

    diagnostic_results = (
        run_automatic_diagnostics(
            normalized_dataset_path,
            profiled_data,
        )
    )

    return {
        "original_dataset_path": (
            original_dataset_path
        ),
        "normalized_dataset_path": (
            normalized_dataset_path
        ),
        "normalization_files": (
            saved_normalization_files
        ),
        "dataset_profile": dataset_profile,
        "diagnostic_results": diagnostic_results,
    }

# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":
    dataset_path = input(
        "Paste the full path of a raw or normalized "
        "battery dataset: "
    ).strip().strip('"').strip("'")

    if not dataset_path:
        print(
            "\nBattery pipeline failed: "
            "No dataset path was supplied."
        )

        sys.exit(1)

    try:
        pipeline_results = run_complete_pipeline(
            dataset_path
        )

        print("\n" + "=" * 75)
        print("COMPLETE PIPELINE FINISHED")
        print("=" * 75)

        print(
            "\nNormalized dataset used by diagnostics:"
        )

        print(
            pipeline_results[
                "normalized_dataset_path"
            ]
        )

        print(
            "\nThe original raw dataset was not modified."
        )

        print("=" * 75)

    except Exception as error:
        print(
            f"\nBattery pipeline failed: {error}"
        )

        sys.exit(1)