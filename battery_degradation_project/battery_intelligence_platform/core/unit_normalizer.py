from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


# ============================================================
# LOCATE CORE FOLDER
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
CORE_FOLDER = CURRENT_FILE.parent
PLATFORM_FOLDER = CORE_FOLDER.parent

sys.path.insert(0, str(CORE_FOLDER))

from file_reader import load_battery_dataset


# ============================================================
# NORMALIZER CONFIGURATION
# ============================================================

NORMALIZATION_VERSION = "1.0"

STANDARD_COLUMN_NAMES = {
    "voltage": "voltage_v",
    "current": "current_a",
    "capacity": "capacity_ah",
    "time": "time_s",
    "temperature": "temperature_c",
}


# Explicit column-name aliases.
#
# Automatic conversion is performed only when the source unit
# is clear from the original column name.

COLUMN_UNIT_ALIASES = {
    # --------------------------------------------------------
    # VOLTAGE
    # --------------------------------------------------------
    "voltage_v": {
        "measurement": "voltage",
        "source_unit": "V",
        "target_unit": "V",
        "conversion_type": "identity",
        "confidence": "High",
    },
    "voltage_mv": {
        "measurement": "voltage",
        "source_unit": "mV",
        "target_unit": "V",
        "conversion_type": "divide_1000",
        "confidence": "High",
    },
    "cell_voltage_v": {
        "measurement": "voltage",
        "source_unit": "V",
        "target_unit": "V",
        "conversion_type": "identity",
        "confidence": "High",
    },
    "cell_voltage_mv": {
        "measurement": "voltage",
        "source_unit": "mV",
        "target_unit": "V",
        "conversion_type": "divide_1000",
        "confidence": "High",
    },
    "terminal_voltage_v": {
        "measurement": "voltage",
        "source_unit": "V",
        "target_unit": "V",
        "conversion_type": "identity",
        "confidence": "High",
    },
    "terminal_voltage_mv": {
        "measurement": "voltage",
        "source_unit": "mV",
        "target_unit": "V",
        "conversion_type": "divide_1000",
        "confidence": "High",
    },

    # --------------------------------------------------------
    # CURRENT
    # --------------------------------------------------------
    "current_a": {
        "measurement": "current",
        "source_unit": "A",
        "target_unit": "A",
        "conversion_type": "identity",
        "confidence": "High",
    },
    "current_ma": {
        "measurement": "current",
        "source_unit": "mA",
        "target_unit": "A",
        "conversion_type": "divide_1000",
        "confidence": "High",
    },
    "cell_current_a": {
        "measurement": "current",
        "source_unit": "A",
        "target_unit": "A",
        "conversion_type": "identity",
        "confidence": "High",
    },
    "cell_current_ma": {
        "measurement": "current",
        "source_unit": "mA",
        "target_unit": "A",
        "conversion_type": "divide_1000",
        "confidence": "High",
    },
    "load_current_a": {
        "measurement": "current",
        "source_unit": "A",
        "target_unit": "A",
        "conversion_type": "identity",
        "confidence": "High",
    },
    "load_current_ma": {
        "measurement": "current",
        "source_unit": "mA",
        "target_unit": "A",
        "conversion_type": "divide_1000",
        "confidence": "High",
    },

    # --------------------------------------------------------
    # CAPACITY
    # --------------------------------------------------------
    "capacity_ah": {
        "measurement": "capacity",
        "source_unit": "Ah",
        "target_unit": "Ah",
        "conversion_type": "identity",
        "confidence": "High",
    },
    "capacity_mah": {
        "measurement": "capacity",
        "source_unit": "mAh",
        "target_unit": "Ah",
        "conversion_type": "divide_1000",
        "confidence": "High",
    },
    "discharge_capacity_ah": {
        "measurement": "capacity",
        "source_unit": "Ah",
        "target_unit": "Ah",
        "conversion_type": "identity",
        "confidence": "High",
    },
    "discharge_capacity_mah": {
        "measurement": "capacity",
        "source_unit": "mAh",
        "target_unit": "Ah",
        "conversion_type": "divide_1000",
        "confidence": "High",
    },
    "charge_capacity_ah": {
        "measurement": "capacity",
        "source_unit": "Ah",
        "target_unit": "Ah",
        "conversion_type": "identity",
        "confidence": "High",
    },
    "charge_capacity_mah": {
        "measurement": "capacity",
        "source_unit": "mAh",
        "target_unit": "Ah",
        "conversion_type": "divide_1000",
        "confidence": "High",
    },

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------
    "time_s": {
        "measurement": "time",
        "source_unit": "s",
        "target_unit": "s",
        "conversion_type": "identity",
        "confidence": "High",
    },
    "time_sec": {
        "measurement": "time",
        "source_unit": "s",
        "target_unit": "s",
        "conversion_type": "identity",
        "confidence": "High",
    },
    "time_seconds": {
        "measurement": "time",
        "source_unit": "s",
        "target_unit": "s",
        "conversion_type": "identity",
        "confidence": "High",
    },
    "elapsed_time_s": {
        "measurement": "time",
        "source_unit": "s",
        "target_unit": "s",
        "conversion_type": "identity",
        "confidence": "High",
    },
    "time_min": {
        "measurement": "time",
        "source_unit": "min",
        "target_unit": "s",
        "conversion_type": "multiply_60",
        "confidence": "High",
    },
    "time_minutes": {
        "measurement": "time",
        "source_unit": "min",
        "target_unit": "s",
        "conversion_type": "multiply_60",
        "confidence": "High",
    },
    "elapsed_time_min": {
        "measurement": "time",
        "source_unit": "min",
        "target_unit": "s",
        "conversion_type": "multiply_60",
        "confidence": "High",
    },
    "time_h": {
        "measurement": "time",
        "source_unit": "h",
        "target_unit": "s",
        "conversion_type": "multiply_3600",
        "confidence": "High",
    },
    "time_hr": {
        "measurement": "time",
        "source_unit": "h",
        "target_unit": "s",
        "conversion_type": "multiply_3600",
        "confidence": "High",
    },
    "time_hours": {
        "measurement": "time",
        "source_unit": "h",
        "target_unit": "s",
        "conversion_type": "multiply_3600",
        "confidence": "High",
    },
    "elapsed_time_h": {
        "measurement": "time",
        "source_unit": "h",
        "target_unit": "s",
        "conversion_type": "multiply_3600",
        "confidence": "High",
    },

    # --------------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------------
    "temperature_c": {
        "measurement": "temperature",
        "source_unit": "degC",
        "target_unit": "degC",
        "conversion_type": "identity",
        "confidence": "High",
    },
    "temp_c": {
        "measurement": "temperature",
        "source_unit": "degC",
        "target_unit": "degC",
        "conversion_type": "identity",
        "confidence": "High",
    },
    "temperature_deg_c": {
        "measurement": "temperature",
        "source_unit": "degC",
        "target_unit": "degC",
        "conversion_type": "identity",
        "confidence": "High",
    },
    "temperature_k": {
        "measurement": "temperature",
        "source_unit": "K",
        "target_unit": "degC",
        "conversion_type": "kelvin_to_celsius",
        "confidence": "High",
    },
    "temp_k": {
        "measurement": "temperature",
        "source_unit": "K",
        "target_unit": "degC",
        "conversion_type": "kelvin_to_celsius",
        "confidence": "High",
    },
}


# ============================================================
# NORMALIZE A COLUMN NAME
# ============================================================

def normalize_column_name(column_name):
    """
    Convert a column name into a comparison-safe format.

    Examples:
        Voltage (mV) -> voltage_mv
        Current[A]   -> current_a
        Time-min     -> time_min
    """

    normalized_name = str(
        column_name
    ).strip().lower()

    normalized_name = normalized_name.replace(
        "°",
        "deg",
    )

    normalized_name = normalized_name.replace(
        "%",
        "percent",
    )

    normalized_name = re.sub(
        r"[\(\)\[\]\{\}/\\\-]+",
        "_",
        normalized_name,
    )

    normalized_name = re.sub(
        r"[^a-z0-9_]+",
        "_",
        normalized_name,
    )

    normalized_name = re.sub(
        r"_+",
        "_",
        normalized_name,
    )

    return normalized_name.strip("_")


# ============================================================
# CREATE UNIQUE TARGET COLUMN NAME
# ============================================================

def create_target_column_name(
    measurement,
    normalized_source_name,
):
    """
    Select the standardized output column name.

    Charge and discharge capacity are kept separate.
    """

    if measurement == "capacity":
        if "discharge" in normalized_source_name:
            return "discharge_capacity_ah"

        if "charge" in normalized_source_name:
            return "charge_capacity_ah"

        return "capacity_ah"

    return STANDARD_COLUMN_NAMES[
        measurement
    ]


# ============================================================
# APPLY ONE CONVERSION
# ============================================================

def apply_conversion(
    numeric_values,
    conversion_type,
):
    """
    Apply a known unit conversion.
    """

    if conversion_type == "identity":
        return numeric_values.copy()

    if conversion_type == "divide_1000":
        return numeric_values / 1000.0

    if conversion_type == "multiply_60":
        return numeric_values * 60.0

    if conversion_type == "multiply_3600":
        return numeric_values * 3600.0

    if conversion_type == "kelvin_to_celsius":
        return numeric_values - 273.15

    raise ValueError(
        f"Unsupported conversion type: {conversion_type}"
    )


# ============================================================
# CREATE RANGE-BASED WARNING
# ============================================================

def create_range_warning(
    column_name,
    normalized_name,
    numeric_values,
):
    """
    Inspect an unmapped numeric column for suspicious units.

    This function creates warnings only. It never converts a
    column using value range alone.
    """

    valid_values = numeric_values.dropna()

    if valid_values.empty:
        return None

    absolute_median = float(
        valid_values.abs().median()
    )

    median_value = float(
        valid_values.median()
    )

    if "volt" in normalized_name:
        if 500 <= absolute_median <= 6000:
            return (
                f"{column_name} has voltage-like values with "
                f"median magnitude {absolute_median:.4f}. "
                "The values may be millivolts, but no automatic "
                "conversion was applied because the unit is not "
                "explicit in the column name."
            )

    if "current" in normalized_name:
        if absolute_median >= 100:
            return (
                f"{column_name} has current-like values with "
                f"median magnitude {absolute_median:.4f}. "
                "The values may be milliamperes, but no automatic "
                "conversion was applied because the unit is not "
                "explicit in the column name."
            )

    if "capacity" in normalized_name:
        if absolute_median >= 100:
            return (
                f"{column_name} has capacity-like values with "
                f"median magnitude {absolute_median:.4f}. "
                "The values may be milliampere-hours, but no "
                "automatic conversion was applied because the "
                "unit is not explicit in the column name."
            )

    if (
        "temp" in normalized_name
        or "temperature" in normalized_name
    ):
        if 200 <= median_value <= 400:
            return (
                f"{column_name} has temperature-like values "
                f"with median {median_value:.4f}. The values may "
                "be kelvin, but no automatic conversion was "
                "applied because the unit is not explicit in "
                "the column name."
            )

    if "time" in normalized_name:
        return (
            f"{column_name} appears to contain time values, "
            "but the unit is not explicit. The values were "
            "left unchanged."
        )

    return None


# ============================================================
# DETECT WHETHER DATA WAS ALREADY NORMALIZED
# ============================================================

def detect_existing_normalization(dataframe):
    """
    Check for the internal normalization marker.

    This protects against accidental repeated conversion when
    the output CSV is passed through this module again.
    """

    marker_column = "_unit_normalization_version"

    if marker_column not in dataframe.columns:
        return False, None

    marker_values = (
        dataframe[marker_column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not marker_values:
        return False, None

    return True, marker_values


# ============================================================
# NORMALIZE DATASET UNITS
# ============================================================

def normalize_dataset_units(dataframe):
    """
    Normalize explicit battery measurement units.

    Conversion policy:
        Explicit unit in column name -> convert automatically.
        Unit inferred only from range -> warn and leave unchanged.
        Existing normalization marker -> do not convert again.
    """

    already_normalized, marker_values = (
        detect_existing_normalization(
            dataframe
        )
    )

    if already_normalized:
        return (
            dataframe.copy(),
            pd.DataFrame(),
            [
                "The dataset already contains a unit-normalization "
                "marker. No additional conversions were applied. "
                f"Detected marker values: {marker_values}."
            ],
            {
                "already_normalized": True,
                "normalization_version": (
                    NORMALIZATION_VERSION
                ),
                "columns_converted": 0,
                "columns_standardized": 0,
                "columns_unchanged": int(
                    len(dataframe.columns)
                ),
            },
        )

    normalized_data = dataframe.copy()

    audit_records = []
    warnings = []

    source_columns = list(
        dataframe.columns
    )

    columns_converted = 0
    columns_standardized = 0
    columns_unchanged = 0

    for source_column in source_columns:
        normalized_name = normalize_column_name(
            source_column
        )

        alias_information = COLUMN_UNIT_ALIASES.get(
            normalized_name
        )

        numeric_values = pd.to_numeric(
            dataframe[source_column],
            errors="coerce",
        )

        valid_numeric_count = int(
            numeric_values.notna().sum()
        )

        if alias_information is None:
            range_warning = create_range_warning(
                source_column,
                normalized_name,
                numeric_values,
            )

            if range_warning is not None:
                warnings.append(
                    range_warning
                )

            columns_unchanged += 1

            audit_records.append(
                {
                    "source_column": source_column,
                    "normalized_source_name": (
                        normalized_name
                    ),
                    "target_column": source_column,
                    "measurement": "Unknown",
                    "source_unit": "Unknown",
                    "target_unit": "Unchanged",
                    "conversion_type": "none",
                    "conversion_applied": False,
                    "confidence": "Uncertain",
                    "valid_numeric_values": (
                        valid_numeric_count
                    ),
                    "status": (
                        "Unrecognized or ambiguous column"
                    ),
                }
            )

            continue

        measurement = alias_information[
            "measurement"
        ]

        target_column = create_target_column_name(
            measurement,
            normalized_name,
        )

        conversion_type = alias_information[
            "conversion_type"
        ]

        if valid_numeric_count == 0:
            warnings.append(
                f"{source_column} matched a known unit pattern, "
                "but it contains no valid numeric values."
            )

            columns_unchanged += 1

            audit_records.append(
                {
                    "source_column": source_column,
                    "normalized_source_name": (
                        normalized_name
                    ),
                    "target_column": target_column,
                    "measurement": measurement,
                    "source_unit": alias_information[
                        "source_unit"
                    ],
                    "target_unit": alias_information[
                        "target_unit"
                    ],
                    "conversion_type": (
                        conversion_type
                    ),
                    "conversion_applied": False,
                    "confidence": alias_information[
                        "confidence"
                    ],
                    "valid_numeric_values": 0,
                    "status": (
                        "No valid numeric values"
                    ),
                }
            )

            continue

        if (
            target_column in normalized_data.columns
            and target_column != source_column
        ):
            existing_target = pd.to_numeric(
                normalized_data[target_column],
                errors="coerce",
            )

            if existing_target.notna().any():
                warnings.append(
                    f"{source_column} was not converted because "
                    f"{target_column} already exists with valid "
                    "values. This prevents silent overwriting."
                )

                columns_unchanged += 1

                audit_records.append(
                    {
                        "source_column": source_column,
                        "normalized_source_name": (
                            normalized_name
                        ),
                        "target_column": target_column,
                        "measurement": measurement,
                        "source_unit": alias_information[
                            "source_unit"
                        ],
                        "target_unit": alias_information[
                            "target_unit"
                        ],
                        "conversion_type": (
                            conversion_type
                        ),
                        "conversion_applied": False,
                        "confidence": alias_information[
                            "confidence"
                        ],
                        "valid_numeric_values": (
                            valid_numeric_count
                        ),
                        "status": (
                            "Skipped because target exists"
                        ),
                    }
                )

                continue

        converted_values = apply_conversion(
            numeric_values,
            conversion_type,
        )

        normalized_data[target_column] = (
            converted_values
        )

        if target_column != source_column:
            normalized_data = (
                normalized_data.drop(
                    columns=[source_column]
                )
            )

        conversion_applied = (
            conversion_type != "identity"
        )

        if conversion_applied:
            columns_converted += 1
            status = "Converted and standardized"
        else:
            columns_standardized += 1
            status = "Standardized without conversion"

        audit_records.append(
            {
                "source_column": source_column,
                "normalized_source_name": (
                    normalized_name
                ),
                "target_column": target_column,
                "measurement": measurement,
                "source_unit": alias_information[
                    "source_unit"
                ],
                "target_unit": alias_information[
                    "target_unit"
                ],
                "conversion_type": conversion_type,
                "conversion_applied": (
                    conversion_applied
                ),
                "confidence": alias_information[
                    "confidence"
                ],
                "valid_numeric_values": (
                    valid_numeric_count
                ),
                "status": status,
            }
        )

    normalized_data[
        "_unit_normalization_version"
    ] = NORMALIZATION_VERSION

    audit_table = pd.DataFrame(
        audit_records
    )

    summary = {
        "already_normalized": False,
        "normalization_version": (
            NORMALIZATION_VERSION
        ),
        "input_columns": int(
            len(dataframe.columns)
        ),
        "output_columns": int(
            len(normalized_data.columns)
        ),
        "columns_converted": int(
            columns_converted
        ),
        "columns_standardized": int(
            columns_standardized
        ),
        "columns_unchanged": int(
            columns_unchanged
        ),
        "warning_count": int(
            len(warnings)
        ),
    }

    return (
        normalized_data,
        audit_table,
        warnings,
        summary,
    )


# ============================================================
# BUILD ENGINEERING VALIDATION WARNINGS
# ============================================================

def validate_normalized_ranges(dataframe):
    """
    Check standardized values after conversion.

    These checks generate warnings. Records are not deleted.
    """

    warnings = []

    if "voltage_v" in dataframe.columns:
        voltage_values = pd.to_numeric(
            dataframe["voltage_v"],
            errors="coerce",
        )

        nonpositive_voltage_count = int(
            (voltage_values <= 0).sum()
        )

        high_voltage_count = int(
            (voltage_values > 10).sum()
        )

        if nonpositive_voltage_count > 0:
            warnings.append(
                f"{nonpositive_voltage_count} standardized "
                "voltage values are zero or negative."
            )

        if high_voltage_count > 0:
            warnings.append(
                f"{high_voltage_count} standardized voltage "
                "values exceed 10 V. The dataset may contain "
                "pack-level measurements."
            )

    if "capacity_ah" in dataframe.columns:
        capacity_values = pd.to_numeric(
            dataframe["capacity_ah"],
            errors="coerce",
        )

        negative_capacity_count = int(
            (capacity_values < 0).sum()
        )

        if negative_capacity_count > 0:
            warnings.append(
                f"{negative_capacity_count} standardized "
                "capacity values are negative."
            )

    if "time_s" in dataframe.columns:
        time_values = pd.to_numeric(
            dataframe["time_s"],
            errors="coerce",
        )

        negative_time_count = int(
            (time_values < 0).sum()
        )

        if negative_time_count > 0:
            warnings.append(
                f"{negative_time_count} standardized time "
                "values are negative."
            )

    if "temperature_c" in dataframe.columns:
        temperature_values = pd.to_numeric(
            dataframe["temperature_c"],
            errors="coerce",
        )

        unusual_temperature_count = int(
            (
                (temperature_values < -50)
                | (temperature_values > 150)
            ).sum()
        )

        if unusual_temperature_count > 0:
            warnings.append(
                f"{unusual_temperature_count} standardized "
                "temperature values are outside the broad "
                "-50°C to 150°C engineering screening range."
            )

    return warnings


# ============================================================
# SAVE NORMALIZED RESULTS
# ============================================================

def save_normalized_results(
    normalized_data,
    audit_table,
    warnings,
    summary,
    dataset_path,
):
    """
    Save normalized data, audit records, warnings, and summary.
    """

    output_directory = (
        PLATFORM_FOLDER
        / "outputs"
        / Path(dataset_path).stem
        / "unit_normalization"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalized_path = (
        output_directory
        / "normalized_battery_data.csv"
    )

    audit_path = (
        output_directory
        / "unit_conversion_audit.csv"
    )

    warning_path = (
        output_directory
        / "unit_normalization_warnings.csv"
    )

    summary_path = (
        output_directory
        / "unit_normalization_summary.csv"
    )

    normalized_data.to_csv(
        normalized_path,
        index=False,
    )

    audit_table.to_csv(
        audit_path,
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

    pd.DataFrame(
        [summary]
    ).to_csv(
        summary_path,
        index=False,
    )

    return {
        "normalized_data": normalized_path,
        "conversion_audit": audit_path,
        "warnings": warning_path,
        "summary": summary_path,
    }


# ============================================================
# PRINT NORMALIZATION REPORT
# ============================================================

def print_normalization_report(
    dataset_path,
    normalized_data,
    audit_table,
    warnings,
    summary,
    saved_files,
):
    """
    Print a readable unit-normalization report.
    """

    print("\n" + "=" * 75)
    print("BATTERY DATASET UNIT NORMALIZATION")
    print("=" * 75)

    print(
        f"\nDataset: {dataset_path}"
    )

    print(
        f"Normalization version: "
        f"{summary['normalization_version']}"
    )

    if summary["already_normalized"]:
        print(
            "\nThe dataset was already normalized. "
            "No additional conversion was performed."
        )

    else:
        print("\nNORMALIZATION SUMMARY")

        print(
            f"  Input columns: "
            f"{summary['input_columns']}"
        )

        print(
            f"  Output columns: "
            f"{summary['output_columns']}"
        )

        print(
            f"  Columns converted: "
            f"{summary['columns_converted']}"
        )

        print(
            f"  Columns standardized without conversion: "
            f"{summary['columns_standardized']}"
        )

        print(
            f"  Columns left unchanged: "
            f"{summary['columns_unchanged']}"
        )

    if not audit_table.empty:
        print("\nUNIT CONVERSION AUDIT")

        display_columns = [
            "source_column",
            "target_column",
            "source_unit",
            "target_unit",
            "conversion_applied",
            "confidence",
            "status",
        ]

        print(
            audit_table[
                display_columns
            ].to_string(
                index=False
            )
        )

    print("\nOUTPUT COLUMNS")

    for column in normalized_data.columns:
        print(f"  - {column}")

    print("\nWARNINGS")

    if warnings:
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print(
            "  No unit-normalization warnings were generated."
        )

    print("\nSAVED RESULTS")

    for result_name, result_path in (
        saved_files.items()
    ):
        print(
            f"  {result_name}: {result_path}"
        )

    print("\nIMPORTANT")

    print(
        "  Automatic conversions are performed only when the "
        "unit is explicit in the original column name."
    )

    print(
        "  Value-range detection creates warnings only and "
        "does not silently change ambiguous measurements."
    )

    print(
        "  The internal normalization marker prevents the "
        "same normalized output from being converted twice."
    )

    print("=" * 75)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":
    dataset_path = input(
        "Paste the full path of a battery dataset: "
    ).strip().strip('"')

    try:
        original_data = load_battery_dataset(
            dataset_path
        )

        (
            normalized_data,
            conversion_audit,
            normalization_warnings,
            normalization_summary,
        ) = normalize_dataset_units(
            original_data
        )

        range_warnings = (
            validate_normalized_ranges(
                normalized_data
            )
        )

        all_warnings = (
            normalization_warnings
            + range_warnings
        )

        normalization_summary[
            "warning_count"
        ] = len(all_warnings)

        saved_results = save_normalized_results(
            normalized_data,
            conversion_audit,
            all_warnings,
            normalization_summary,
            dataset_path,
        )

        print_normalization_report(
            dataset_path,
            normalized_data,
            conversion_audit,
            all_warnings,
            normalization_summary,
            saved_results,
        )

    except Exception as error:
        print(
            f"\nUnit normalization failed: {error}"
        )