from pathlib import Path
import sys

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
# ICA/DVA REQUIREMENTS
# ============================================================

MINIMUM_POINTS_PER_CYCLE = 20
RECOMMENDED_POINTS_PER_CYCLE = 50
MINIMUM_VOLTAGE_RANGE = 0.10
MINIMUM_CAPACITY_RANGE = 0.01


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_ica_dva_data(dataframe):
    """
    Convert required columns to numeric values and remove
    invalid voltage-capacity records.
    """

    prepared_data = dataframe.copy()

    numeric_columns = [
        "cycle",
        "voltage_v",
        "capacity_ah",
        "time_s",
        "current_a",
    ]

    for column in numeric_columns:
        if column in prepared_data.columns:
            prepared_data[column] = pd.to_numeric(
                prepared_data[column],
                errors="coerce",
            )

    required_columns = [
        "voltage_v",
        "capacity_ah",
    ]

    available_required_columns = [
        column
        for column in required_columns
        if column in prepared_data.columns
    ]

    if available_required_columns:
        prepared_data = prepared_data.dropna(
            subset=available_required_columns
        )

    if "capacity_ah" in prepared_data.columns:
        prepared_data = prepared_data[
            prepared_data["capacity_ah"] >= 0
        ]

    return prepared_data.reset_index(drop=True)


# ============================================================
# DETECT DISCHARGE RECORDS
# ============================================================

def detect_discharge_data(dataframe):
    """
    Try to identify discharge records from the current column.

    The function does not remove data when the current-sign
    convention cannot be identified reliably.
    """

    result = {
        "current_available": False,
        "current_sign_detected": False,
        "discharge_sign": None,
        "discharge_records": None,
        "message": (
            "Current is unavailable. The checker will inspect "
            "all voltage-capacity records."
        ),
    }

    if "current_a" not in dataframe.columns:
        return dataframe.copy(), result

    valid_current = dataframe["current_a"].dropna()

    if valid_current.empty:
        result["message"] = (
            "Current exists but contains no valid numeric values."
        )
        return dataframe.copy(), result

    result["current_available"] = True

    negative_count = int(
        (valid_current < -0.001).sum()
    )

    positive_count = int(
        (valid_current > 0.001).sum()
    )

    if negative_count == 0 and positive_count == 0:
        result["message"] = (
            "Current values are approximately zero. "
            "Discharge records cannot be identified."
        )
        return dataframe.copy(), result

    if negative_count >= positive_count:
        discharge_data = dataframe[
            dataframe["current_a"] < -0.001
        ].copy()

        result["discharge_sign"] = "negative"
        result["message"] = (
            "Negative current was treated as discharge."
        )
    else:
        discharge_data = dataframe[
            dataframe["current_a"] > 0.001
        ].copy()

        result["discharge_sign"] = "positive"
        result["message"] = (
            "Positive current was treated as discharge."
        )

    if len(discharge_data) < MINIMUM_POINTS_PER_CYCLE:
        result["message"] = (
            "Current was found, but too few records remained "
            "after discharge filtering. All valid records were "
            "used for capability checking."
        )
        return dataframe.copy(), result

    result["current_sign_detected"] = True
    result["discharge_records"] = int(
        len(discharge_data)
    )

    return discharge_data, result


# ============================================================
# ANALYZE CYCLE QUALITY
# ============================================================

def analyze_cycle_quality(dataframe):
    """
    Examine the number of points and the voltage-capacity
    ranges available in each cycle.
    """

    if "cycle" not in dataframe.columns:
        working_data = dataframe.copy()
        working_data["cycle"] = 1
        cycle_was_created = True
    else:
        working_data = dataframe.dropna(
            subset=["cycle"]
        ).copy()
        cycle_was_created = False

    cycle_records = []

    for cycle_value, cycle_data in working_data.groupby(
        "cycle",
        sort=True,
    ):
        cycle_data = cycle_data.dropna(
            subset=[
                "voltage_v",
                "capacity_ah",
            ]
        )

        point_count = int(len(cycle_data))

        if cycle_data.empty:
            voltage_range = 0.0
            capacity_range = 0.0
            unique_voltage_points = 0
            unique_capacity_points = 0
        else:
            voltage_range = float(
                cycle_data["voltage_v"].max()
                - cycle_data["voltage_v"].min()
            )

            capacity_range = float(
                cycle_data["capacity_ah"].max()
                - cycle_data["capacity_ah"].min()
            )

            unique_voltage_points = int(
                cycle_data["voltage_v"].nunique()
            )

            unique_capacity_points = int(
                cycle_data["capacity_ah"].nunique()
            )

        sufficient_points = (
            point_count >= MINIMUM_POINTS_PER_CYCLE
        )

        recommended_points = (
            point_count >= RECOMMENDED_POINTS_PER_CYCLE
        )

        sufficient_voltage_range = (
            voltage_range >= MINIMUM_VOLTAGE_RANGE
        )

        sufficient_capacity_range = (
            capacity_range >= MINIMUM_CAPACITY_RANGE
        )

        enough_unique_values = (
            unique_voltage_points >= 10
            and unique_capacity_points >= 10
        )

        reliable_for_ica_dva = all(
            [
                sufficient_points,
                sufficient_voltage_range,
                sufficient_capacity_range,
                enough_unique_values,
            ]
        )

        cycle_records.append(
            {
                "cycle": cycle_value,
                "point_count": point_count,
                "unique_voltage_points": (
                    unique_voltage_points
                ),
                "unique_capacity_points": (
                    unique_capacity_points
                ),
                "voltage_range_v": voltage_range,
                "capacity_range_ah": capacity_range,
                "minimum_points_passed": (
                    sufficient_points
                ),
                "recommended_points_passed": (
                    recommended_points
                ),
                "voltage_range_passed": (
                    sufficient_voltage_range
                ),
                "capacity_range_passed": (
                    sufficient_capacity_range
                ),
                "reliable_for_ica_dva": (
                    reliable_for_ica_dva
                ),
            }
        )

    cycle_quality = pd.DataFrame(
        cycle_records
    )

    return cycle_quality, cycle_was_created


# ============================================================
# CHECK ICA/DVA CAPABILITY
# ============================================================

def check_ica_dva_capability(dataframe):
    """
    Determine whether the dataset supports ICA and DVA.

    ICA requires voltage and capacity to calculate dQ/dV.
    DVA requires voltage and capacity to calculate dV/dQ.
    """

    original_rows = int(len(dataframe))

    required_columns = [
        "voltage_v",
        "capacity_ah",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    report = {
        "ica_available": False,
        "dva_available": False,
        "reliability": "Unavailable",
        "reason": "",
        "original_rows": original_rows,
        "valid_rows": 0,
        "cycles_found": 0,
        "reliable_cycles": 0,
        "recommended_cycles": 0,
        "cycle_column_available": (
            "cycle" in dataframe.columns
        ),
        "voltage_available": (
            "voltage_v" in dataframe.columns
        ),
        "capacity_available": (
            "capacity_ah" in dataframe.columns
        ),
    }

    if missing_columns:
        report["reason"] = (
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

        return report, pd.DataFrame(), None

    prepared_data = prepare_ica_dva_data(
        dataframe
    )

    report["valid_rows"] = int(
        len(prepared_data)
    )

    if prepared_data.empty:
        report["reason"] = (
            "Voltage and capacity columns were found, "
            "but no valid numeric records remained."
        )

        return report, pd.DataFrame(), None

    discharge_data, discharge_information = (
        detect_discharge_data(
            prepared_data
        )
    )

    cycle_quality, cycle_was_created = (
        analyze_cycle_quality(
            discharge_data
        )
    )

    if cycle_quality.empty:
        report["reason"] = (
            "No valid cycle-level voltage-capacity records "
            "were found."
        )

        return (
            report,
            cycle_quality,
            discharge_information,
        )

    report["cycles_found"] = int(
        len(cycle_quality)
    )

    report["reliable_cycles"] = int(
        cycle_quality[
            "reliable_for_ica_dva"
        ].sum()
    )

    report["recommended_cycles"] = int(
        cycle_quality[
            "recommended_points_passed"
        ].sum()
    )

    if report["reliable_cycles"] == 0:
        report["reason"] = (
            "Voltage and capacity are available, but no cycle "
            "has enough valid voltage-capacity measurements "
            "and sufficient signal range for reliable ICA/DVA."
        )

        report["reliability"] = "Not reliable"

        return (
            report,
            cycle_quality,
            discharge_information,
        )

    report["ica_available"] = True
    report["dva_available"] = True

    reliable_fraction = (
        report["reliable_cycles"]
        / report["cycles_found"]
    )

    if (
        report["recommended_cycles"] >= 1
        and reliable_fraction >= 0.75
    ):
        report["reliability"] = "High"
    elif reliable_fraction >= 0.40:
        report["reliability"] = "Moderate"
    else:
        report["reliability"] = "Limited"

    if cycle_was_created:
        report["reason"] = (
            "ICA and DVA are possible, but the dataset has no "
            "cycle column. The complete dataset will be treated "
            "as one analysis sequence."
        )
    else:
        report["reason"] = (
            "Voltage-capacity curves contain enough valid "
            "measurements for ICA and DVA analysis."
        )

    return (
        report,
        cycle_quality,
        discharge_information,
    )


# ============================================================
# PRINT CAPABILITY REPORT
# ============================================================

def print_capability_report(
    report,
    cycle_quality,
    discharge_information,
    unknown_columns,
):
    """
    Print a readable ICA/DVA capability report.
    """

    print("\n" + "=" * 75)
    print("ICA AND DVA CAPABILITY CHECK")
    print("=" * 75)

    print(
        f"\nOriginal records: "
        f"{report['original_rows']:,}"
    )

    print(
        f"Valid voltage-capacity records: "
        f"{report['valid_rows']:,}"
    )

    print(
        f"Cycles or sequences found: "
        f"{report['cycles_found']}"
    )

    print(
        f"Reliable cycles: "
        f"{report['reliable_cycles']}"
    )

    print(
        f"Recommended-quality cycles: "
        f"{report['recommended_cycles']}"
    )

    print("\nCOLUMN AVAILABILITY")

    print(
        "  Voltage: "
        + (
            "Available"
            if report["voltage_available"]
            else "Missing"
        )
    )

    print(
        "  Capacity: "
        + (
            "Available"
            if report["capacity_available"]
            else "Missing"
        )
    )

    print(
        "  Cycle: "
        + (
            "Available"
            if report["cycle_column_available"]
            else "Missing"
        )
    )

    print("\nANALYSIS AVAILABILITY")

    print(
        "  ICA (dQ/dV): "
        + (
            "Available"
            if report["ica_available"]
            else "Not available"
        )
    )

    print(
        "  DVA (dV/dQ): "
        + (
            "Available"
            if report["dva_available"]
            else "Not available"
        )
    )

    print(
        f"  Reliability: "
        f"{report['reliability']}"
    )

    print(
        f"\nReason: {report['reason']}"
    )

    if discharge_information is not None:
        print("\nDISCHARGE DETECTION")

        print(
            "  "
            + discharge_information["message"]
        )

        if (
            discharge_information[
                "discharge_records"
            ]
            is not None
        ):
            print(
                "  Discharge records used: "
                f"{discharge_information['discharge_records']:,}"
            )

    if not cycle_quality.empty:
        print("\nCYCLE QUALITY SUMMARY")

        display_columns = [
            "cycle",
            "point_count",
            "voltage_range_v",
            "capacity_range_ah",
            "reliable_for_ica_dva",
        ]

        print(
            cycle_quality[
                display_columns
            ]
            .head(10)
            .to_string(index=False)
        )

        if len(cycle_quality) > 10:
            print(
                f"\n  Showing 10 of "
                f"{len(cycle_quality)} cycles."
            )

    if unknown_columns:
        print("\nUNRECOGNIZED ORIGINAL COLUMNS")

        for column in unknown_columns:
            print(f"  - {column}")

    print("\nMINIMUM CHECKS")

    print(
        f"  Minimum points per cycle: "
        f"{MINIMUM_POINTS_PER_CYCLE}"
    )

    print(
        f"  Recommended points per cycle: "
        f"{RECOMMENDED_POINTS_PER_CYCLE}"
    )

    print(
        f"  Minimum voltage range: "
        f"{MINIMUM_VOLTAGE_RANGE:.2f} V"
    )

    print(
        f"  Minimum capacity range: "
        f"{MINIMUM_CAPACITY_RANGE:.3f} Ah"
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

        capability_report, cycle_quality, discharge_info = (
            check_ica_dva_capability(
                standardized_data
            )
        )

        print_capability_report(
            capability_report,
            cycle_quality,
            discharge_info,
            unknown_columns,
        )

    except Exception as error:
        print(
            f"\nICA/DVA capability check failed: {error}"
        )