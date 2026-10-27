from pathlib import Path
import sys

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
# SOC CAPABILITY REQUIREMENTS
# ============================================================

SOC_REQUIRED_SIGNALS = {
    "current_a": (
        "Current is required to measure charge entering "
        "or leaving the battery."
    ),
    "time_s": (
        "Time is required to integrate current using "
        "coulomb counting."
    ),
}

SOC_SUPPORTING_SIGNALS = {
    "voltage_v": (
        "Voltage helps validate and correct SOC estimates."
    ),
    "temperature_c": (
        "Temperature helps account for changes in battery "
        "behavior and usable capacity."
    ),
    "capacity_ah": (
        "Capacity provides the battery charge reference."
    ),
    "soc_percent": (
        "Existing SOC labels can be used to train and "
        "validate a machine-learning SOC model."
    ),
}


# ============================================================
# FIND AVAILABLE SIGNALS
# ============================================================

def find_available_signals(dataframe):
    """
    Identify which standardized battery signals are available.
    """

    available_signals = []

    for column in dataframe.columns:
        if dataframe[column].notna().any():
            available_signals.append(column)

    return available_signals


# ============================================================
# CHECK DATA QUALITY
# ============================================================

def inspect_signal_quality(dataframe, signal_name):
    """
    Return basic data-quality information for one signal.
    """

    if signal_name not in dataframe.columns:
        return {
            "signal": signal_name,
            "available": False,
            "valid_records": 0,
            "missing_records": len(dataframe),
            "unique_values": 0,
            "status": "Missing",
        }

    numeric_signal = pd.to_numeric(
        dataframe[signal_name],
        errors="coerce",
    )

    valid_records = int(
        numeric_signal.notna().sum()
    )

    missing_records = int(
        numeric_signal.isna().sum()
    )

    unique_values = int(
        numeric_signal.nunique()
    )

    if valid_records == 0:
        status = "Invalid"

    elif unique_values <= 1:
        status = "Constant"

    elif missing_records > valid_records:
        status = "Poor Quality"

    else:
        status = "Usable"

    return {
        "signal": signal_name,
        "available": True,
        "valid_records": valid_records,
        "missing_records": missing_records,
        "unique_values": unique_values,
        "status": status,
    }


# ============================================================
# CHECK SOC CAPABILITY
# ============================================================

def check_soc_capability(dataframe):
    """
    Decide which SOC estimation methods can be supported
    by the uploaded battery dataset.
    """

    available_signals = find_available_signals(
        dataframe
    )

    signal_quality = []

    all_signals = list(
        SOC_REQUIRED_SIGNALS.keys()
    ) + list(
        SOC_SUPPORTING_SIGNALS.keys()
    )

    for signal_name in all_signals:
        signal_quality.append(
            inspect_signal_quality(
                dataframe,
                signal_name,
            )
        )

    quality_table = pd.DataFrame(
        signal_quality
    )

    usable_signals = set(
        quality_table.loc[
            quality_table["status"] == "Usable",
            "signal",
        ]
    )

    has_current = "current_a" in usable_signals
    has_time = "time_s" in usable_signals
    has_voltage = "voltage_v" in usable_signals
    has_capacity = "capacity_ah" in usable_signals
    has_soc_label = "soc_percent" in usable_signals

    coulomb_counting_ready = (
        has_current
        and has_time
        and has_capacity
    )

    label_based_ml_ready = (
        has_soc_label
        and has_voltage
        and has_current
    )

    voltage_based_analysis_ready = (
        has_voltage
    )

    if coulomb_counting_ready:
        overall_status = "SOC estimation is ready"

    elif label_based_ml_ready:
        overall_status = "ML-based SOC modelling is ready"

    elif has_current and has_time:
        overall_status = (
            "Partially ready: battery capacity is required"
        )

    else:
        overall_status = (
            "Not ready: required SOC signals are missing"
        )

    return {
        "available_signals": available_signals,
        "quality_table": quality_table,
        "coulomb_counting_ready": coulomb_counting_ready,
        "label_based_ml_ready": label_based_ml_ready,
        "voltage_based_analysis_ready": (
            voltage_based_analysis_ready
        ),
        "overall_status": overall_status,
    }


# ============================================================
# PRINT REPORT
# ============================================================

def print_soc_capability_report(
    capability_result,
    original_mapping,
    unknown_columns,
):
    """
    Print an understandable SOC capability report.
    """

    print("\n" + "=" * 75)
    print("BATTERY SOC CAPABILITY CHECK")
    print("=" * 75)

    print("\nDETECTED STANDARD BATTERY SIGNALS")

    if original_mapping:
        for original_column, standard_column in (
            original_mapping.items()
        ):
            print(
                f"  {original_column} -> {standard_column}"
            )

    else:
        print(
            "  No recognized battery signals were detected."
        )

    print("\nSIGNAL QUALITY")

    print(
        capability_result["quality_table"].to_string(
            index=False
        )
    )

    print("\nSUPPORTED SOC METHODS")

    print(
        "  Coulomb counting: "
        f"{'READY' if capability_result['coulomb_counting_ready'] else 'NOT READY'}"
    )

    print(
        "  Label-based ML model: "
        f"{'READY' if capability_result['label_based_ml_ready'] else 'NOT READY'}"
    )

    print(
        "  Voltage-based analysis: "
        f"{'READY' if capability_result['voltage_based_analysis_ready'] else 'NOT READY'}"
    )

    print("\nOVERALL RESULT")

    print(
        f"  {capability_result['overall_status']}"
    )

    if unknown_columns:
        print("\nUNRECOGNIZED ORIGINAL COLUMNS")

        for column in unknown_columns:
            print(f"  - {column}")

        print(
            "\nUnrecognized columns are not deleted. "
            "They are simply not required by the current "
            "SOC capability checker."
        )

    print("\nSOC REQUIREMENTS")

    print(
        "  Coulomb counting requires current, time, "
        "and battery capacity."
    )

    print(
        "  ML-based SOC modelling normally requires "
        "voltage, current, and existing SOC labels."
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

        capability_result = check_soc_capability(
            standardized_data
        )

        print_soc_capability_report(
            capability_result,
            mapping,
            unknown_columns,
        )

    except Exception as error:
        print(
            f"\nSOC capability check failed: {error}"
        )