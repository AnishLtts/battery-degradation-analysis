from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# LOCATE CORE FOLDER
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
CORE_FOLDER = CURRENT_FILE.parent

sys.path.insert(0, str(CORE_FOLDER))

from file_reader import load_battery_dataset


# ============================================================
# CYCLE-FEATURE SCHEMA
# ============================================================

CYCLE_FEATURE_COLUMN_MAPPING = {
    "battery_id": "battery_id",
    "cycle": "cycle",
    "IR": "internal_resistance_ohm",
    "QC": "charge_capacity_ah",
    "QD": "discharge_capacity_ah",
    "Tavg": "average_temperature_c",
    "Tmin": "minimum_temperature_c",
    "Tmax": "maximum_temperature_c",
    "chargetime": "charge_time",
    "cycle_life": "cycle_life",
}


# Retain these columns without assigning a meaning.
AMBIGUOUS_COLUMNS = [
    "C1",
    "Q1",
    "C2",
]


REQUIRED_SCHEMA_COLUMNS = [
    "battery_id",
    "cycle",
    "QC",
    "QD",
]


NUMERIC_STANDARD_COLUMNS = [
    "cycle",
    "internal_resistance_ohm",
    "charge_capacity_ah",
    "discharge_capacity_ah",
    "average_temperature_c",
    "minimum_temperature_c",
    "maximum_temperature_c",
    "charge_time",
    "cycle_life",
]


# ============================================================
# DETECT CYCLE-FEATURE DATASET
# ============================================================

def detect_cycle_feature_schema(dataframe):
    """
    Detect whether the dataset matches a multi-battery,
    cycle-level feature schema.

    Required original columns:
        battery_id
        cycle
        QC
        QD
    """

    available_columns = set(
        dataframe.columns
    )

    missing_required_columns = [
        column
        for column in REQUIRED_SCHEMA_COLUMNS
        if column not in available_columns
    ]

    matched_known_columns = [
        column
        for column in CYCLE_FEATURE_COLUMN_MAPPING
        if column in available_columns
    ]

    matched_ambiguous_columns = [
        column
        for column in AMBIGUOUS_COLUMNS
        if column in available_columns
    ]

    required_match_count = (
        len(REQUIRED_SCHEMA_COLUMNS)
        - len(missing_required_columns)
    )

    schema_detected = (
        len(missing_required_columns) == 0
    )

    confidence_score = (
        required_match_count
        / len(REQUIRED_SCHEMA_COLUMNS)
        * 70
    )

    optional_known_columns = [
        column
        for column in CYCLE_FEATURE_COLUMN_MAPPING
        if column not in REQUIRED_SCHEMA_COLUMNS
    ]

    optional_match_count = sum(
        1
        for column in optional_known_columns
        if column in available_columns
    )

    if optional_known_columns:
        confidence_score += (
            optional_match_count
            / len(optional_known_columns)
            * 30
        )

    confidence_score = int(
        round(
            min(100, confidence_score)
        )
    )

    if confidence_score >= 90:
        confidence_level = "High"
    elif confidence_score >= 70:
        confidence_level = "Moderate"
    elif confidence_score >= 40:
        confidence_level = "Limited"
    else:
        confidence_level = "Low"

    return {
        "schema_detected": schema_detected,
        "confidence_score": confidence_score,
        "confidence_level": confidence_level,
        "matched_known_columns": matched_known_columns,
        "matched_ambiguous_columns": (
            matched_ambiguous_columns
        ),
        "missing_required_columns": (
            missing_required_columns
        ),
    }


# ============================================================
# APPLY SAFE COLUMN MAPPING
# ============================================================

def apply_cycle_feature_mapping(dataframe):
    """
    Rename only columns with sufficiently clear meanings.

    C1, Q1, and C2 remain unchanged because their exact
    definitions have not been verified.
    """

    schema_report = detect_cycle_feature_schema(
        dataframe
    )

    if not schema_report["schema_detected"]:
        missing_text = ", ".join(
            schema_report[
                "missing_required_columns"
            ]
        )

        raise ValueError(
            "This dataset does not match the supported "
            "cycle-feature schema. Missing required columns: "
            + missing_text
        )

    applicable_mapping = {
        original_column: standard_column
        for original_column, standard_column
        in CYCLE_FEATURE_COLUMN_MAPPING.items()
        if original_column in dataframe.columns
    }

    adapted_data = dataframe.rename(
        columns=applicable_mapping
    ).copy()

    return (
        adapted_data,
        applicable_mapping,
        schema_report,
    )


# ============================================================
# CLEAN CYCLE-FEATURE DATA
# ============================================================

def clean_cycle_feature_data(dataframe):
    """
    Convert supported fields to numeric values and remove rows
    that cannot be assigned to a battery and cycle.
    """

    cleaned_data = dataframe.copy()

    for column in NUMERIC_STANDARD_COLUMNS:
        if column in cleaned_data.columns:
            cleaned_data[column] = pd.to_numeric(
                cleaned_data[column],
                errors="coerce",
            )

    cleaned_data = cleaned_data.dropna(
        subset=[
            "battery_id",
            "cycle",
        ]
    )

    if "charge_capacity_ah" in cleaned_data.columns:
        cleaned_data.loc[
            cleaned_data["charge_capacity_ah"] < 0,
            "charge_capacity_ah",
        ] = np.nan

    if "discharge_capacity_ah" in cleaned_data.columns:
        cleaned_data.loc[
            cleaned_data["discharge_capacity_ah"] < 0,
            "discharge_capacity_ah",
        ] = np.nan

    if "internal_resistance_ohm" in cleaned_data.columns:
        cleaned_data.loc[
            cleaned_data[
                "internal_resistance_ohm"
            ] < 0,
            "internal_resistance_ohm",
        ] = np.nan

    cleaned_data = cleaned_data.sort_values(
        by=[
            "battery_id",
            "cycle",
        ],
        kind="stable",
    ).reset_index(drop=True)

    if cleaned_data.empty:
        raise ValueError(
            "No valid battery and cycle records remained "
            "after cleaning."
        )

    return cleaned_data


# ============================================================
# CALCULATE DERIVED FEATURES
# ============================================================

def calculate_cycle_features(dataframe):
    """
    Calculate safe cycle-level battery features.

    Derived features:
        coulombic_efficiency_percent
        temperature_spread_c
        reference_discharge_capacity_ah
        discharge_capacity_retention_percent
        cycle_soh_percent
        cycle_life_target_consistent
        cycle_life_ml_eligible

    SOH is calculated relative to the median discharge capacity
    from the first five available cycles for each battery.

    A battery is eligible for future supervised cycle-life
    modelling only when it has one nonmissing cycle_life target
    shared consistently across its rows.
    """

    results = dataframe.copy()

    # --------------------------------------------------------
    # COULOMBIC EFFICIENCY
    # --------------------------------------------------------

    if all(
        column in results.columns
        for column in [
            "charge_capacity_ah",
            "discharge_capacity_ah",
        ]
    ):
        valid_charge_capacity = (
            results["charge_capacity_ah"] > 0
        )

        results[
            "coulombic_efficiency_percent"
        ] = np.nan

        results.loc[
            valid_charge_capacity,
            "coulombic_efficiency_percent",
        ] = (
            results.loc[
                valid_charge_capacity,
                "discharge_capacity_ah",
            ]
            / results.loc[
                valid_charge_capacity,
                "charge_capacity_ah",
            ]
            * 100
        )

    # --------------------------------------------------------
    # TEMPERATURE SPREAD
    # --------------------------------------------------------

    if all(
        column in results.columns
        for column in [
            "minimum_temperature_c",
            "maximum_temperature_c",
        ]
    ):
        results["temperature_spread_c"] = (
            results["maximum_temperature_c"]
            - results["minimum_temperature_c"]
        )

    # --------------------------------------------------------
    # REFERENCE CAPACITY, RETENTION, AND SOH
    # --------------------------------------------------------

    if "discharge_capacity_ah" in results.columns:
        reference_capacity_table = (
            results.dropna(
                subset=[
                    "discharge_capacity_ah",
                ]
            )
            .sort_values(
                [
                    "battery_id",
                    "cycle",
                ]
            )
            .groupby(
                "battery_id",
                group_keys=False,
            )
            .head(5)
            .groupby(
                "battery_id"
            )[
                "discharge_capacity_ah"
            ]
            .median()
            .rename(
                "reference_discharge_capacity_ah"
            )
        )

        results = results.merge(
            reference_capacity_table,
            left_on="battery_id",
            right_index=True,
            how="left",
        )

        valid_reference = (
            results[
                "reference_discharge_capacity_ah"
            ] > 0
        )

        results[
            "discharge_capacity_retention_percent"
        ] = np.nan

        results[
            "cycle_soh_percent"
        ] = np.nan

        results.loc[
            valid_reference,
            "discharge_capacity_retention_percent",
        ] = (
            results.loc[
                valid_reference,
                "discharge_capacity_ah",
            ]
            / results.loc[
                valid_reference,
                "reference_discharge_capacity_ah",
            ]
            * 100
        )

        results.loc[
            valid_reference,
            "cycle_soh_percent",
        ] = results.loc[
            valid_reference,
            "discharge_capacity_retention_percent",
        ]

    # --------------------------------------------------------
    # CYCLE-LIFE TARGET CONSISTENCY
    # --------------------------------------------------------

    if "cycle_life" in results.columns:
        cycle_life_unique_count = (
            results.groupby(
                "battery_id"
            )["cycle_life"]
            .transform(
                lambda values: (
                    values.dropna().nunique()
                )
            )
        )

        results[
            "cycle_life_unique_target_count"
        ] = cycle_life_unique_count.astype(
            int
        )

        results[
            "cycle_life_target_consistent"
        ] = (
            cycle_life_unique_count == 1
        )

        results[
            "cycle_life_ml_eligible"
        ] = (
            results[
                "cycle_life_target_consistent"
            ]
            & results["cycle_life"].notna()
        )

    return results


# ============================================================
# BUILD CYCLE-LIFE CONSISTENCY REPORT
# ============================================================

def build_cycle_life_consistency_report(dataframe):
    """
    Create one cycle-life target consistency record per battery.

    Batteries containing multiple cycle_life values are retained
    for descriptive analysis but marked as ineligible for future
    supervised cycle-life model training.
    """

    if "cycle_life" not in dataframe.columns:
        return pd.DataFrame(
            columns=[
                "battery_id",
                "row_count",
                "nonmissing_target_count",
                "unique_target_count",
                "minimum_cycle_life",
                "maximum_cycle_life",
                "cycle_life_range",
                "target_consistent",
                "ml_eligible",
                "target_values",
            ]
        )

    report_records = []

    for battery_id, battery_data in dataframe.groupby(
        "battery_id",
        sort=True,
    ):
        target_values = (
            battery_data["cycle_life"]
            .dropna()
        )

        unique_target_values = sorted(
            target_values.unique().tolist()
        )

        unique_target_count = int(
            len(unique_target_values)
        )

        target_consistent = (
            unique_target_count == 1
        )

        ml_eligible = (
            target_consistent
            and len(target_values) > 0
        )

        if target_values.empty:
            minimum_cycle_life = None
            maximum_cycle_life = None
            cycle_life_range = None
        else:
            minimum_cycle_life = float(
                target_values.min()
            )

            maximum_cycle_life = float(
                target_values.max()
            )

            cycle_life_range = float(
                maximum_cycle_life
                - minimum_cycle_life
            )

        target_values_text = ", ".join(
            f"{float(value):g}"
            for value in unique_target_values
        )

        report_records.append(
            {
                "battery_id": battery_id,
                "row_count": int(
                    len(battery_data)
                ),
                "nonmissing_target_count": int(
                    len(target_values)
                ),
                "unique_target_count": (
                    unique_target_count
                ),
                "minimum_cycle_life": (
                    minimum_cycle_life
                ),
                "maximum_cycle_life": (
                    maximum_cycle_life
                ),
                "cycle_life_range": (
                    cycle_life_range
                ),
                "target_consistent": (
                    target_consistent
                ),
                "ml_eligible": ml_eligible,
                "target_values": (
                    target_values_text
                ),
            }
        )

    return pd.DataFrame(
        report_records
    )


# ============================================================
# DETECT SUPPORTED ANALYSES
# ============================================================

def detect_cycle_feature_capabilities(dataframe):
    """
    Identify analyses supported by the cycle-feature dataset.

    Raw time-series analyses are explicitly unavailable.
    """

    available_columns = set(
        dataframe.columns
    )

    if "cycle_life_ml_eligible" in dataframe.columns:
        eligible_battery_count = int(
            dataframe.loc[
                dataframe[
                    "cycle_life_ml_eligible"
                ],
                "battery_id",
            ].nunique()
        )
    else:
        eligible_battery_count = 0

    cycle_life_modelling_available = (
        "cycle_life" in available_columns
        and eligible_battery_count >= 2
    )

    capabilities = {
        "battery_comparison": {
            "available": all(
                column in available_columns
                for column in [
                    "battery_id",
                    "cycle",
                ]
            ),
            "reason": (
                "Battery identifier and cycle are available."
            ),
        },
        "cycle_level_soh": {
            "available": (
                "discharge_capacity_ah"
                in available_columns
            ),
            "reason": (
                "Cycle-level discharge capacity is available."
            ),
        },
        "capacity_degradation": {
            "available": all(
                column in available_columns
                for column in [
                    "battery_id",
                    "cycle",
                    "discharge_capacity_ah",
                ]
            ),
            "reason": (
                "Battery, cycle, and discharge capacity "
                "are available."
            ),
        },
        "coulombic_efficiency": {
            "available": all(
                column in available_columns
                for column in [
                    "charge_capacity_ah",
                    "discharge_capacity_ah",
                ]
            ),
            "reason": (
                "Charge and discharge capacities are available."
            ),
        },
        "resistance_trend": {
            "available": (
                "internal_resistance_ohm"
                in available_columns
            ),
            "reason": (
                "Cycle-level internal resistance is available."
            ),
        },
        "temperature_trend": {
            "available": any(
                column in available_columns
                for column in [
                    "average_temperature_c",
                    "minimum_temperature_c",
                    "maximum_temperature_c",
                ]
            ),
            "reason": (
                "One or more cycle-level temperature "
                "features are available."
            ),
        },
        "charge_time_trend": {
            "available": (
                "charge_time" in available_columns
            ),
            "reason": (
                "Charge-time values are available, but "
                "their unit must be confirmed."
            ),
        },
        "cycle_life_modelling": {
            "available": (
                cycle_life_modelling_available
            ),
            "reason": (
                f"{eligible_battery_count} batteries have "
                "consistent, nonmissing cycle-life targets. "
                "Ineligible batteries must be excluded, and "
                "train-test splitting must be performed by "
                "battery_id to prevent leakage."
            ),
        },
        "soc_coulomb_counting": {
            "available": False,
            "reason": (
                "Raw current and elapsed-time measurements "
                "are unavailable."
            ),
        },
        "ica": {
            "available": False,
            "reason": (
                "Raw voltage-capacity curve measurements "
                "are unavailable."
            ),
        },
        "dva": {
            "available": False,
            "reason": (
                "Raw voltage-capacity curve measurements "
                "are unavailable."
            ),
        },
        "voltage_curve_analysis": {
            "available": False,
            "reason": (
                "Raw voltage measurements are unavailable."
            ),
        },
    }

    return capabilities


# ============================================================
# BUILD DATASET SUMMARY
# ============================================================

def build_cycle_feature_summary(dataframe):
    """
    Build a summary for the adapted cycle-feature dataset.
    """

    battery_count = int(
        dataframe["battery_id"]
        .dropna()
        .nunique()
    )

    total_rows = int(
        len(dataframe)
    )

    unique_cycle_values = int(
        dataframe["cycle"]
        .dropna()
        .nunique()
    )

    records_per_battery = (
        dataframe.groupby(
            "battery_id"
        )
        .size()
    )

    cycles_per_battery = (
        dataframe.groupby(
            "battery_id"
        )["cycle"]
        .nunique()
    )

    duplicate_battery_cycle_rows = int(
        dataframe.duplicated(
            subset=[
                "battery_id",
                "cycle",
            ]
        ).sum()
    )

    eligible_batteries = 0
    ineligible_batteries = 0

    if "cycle_life_ml_eligible" in dataframe.columns:
        eligibility_by_battery = (
            dataframe.groupby(
                "battery_id"
            )[
                "cycle_life_ml_eligible"
            ]
            .all()
        )

        eligible_batteries = int(
            eligibility_by_battery.sum()
        )

        ineligible_batteries = int(
            (~eligibility_by_battery).sum()
        )

    summary = {
        "dataset_type": (
            "Multi-battery cycle-level feature dataset"
        ),
        "total_rows": total_rows,
        "battery_count": battery_count,
        "unique_cycle_values": unique_cycle_values,
        "minimum_records_per_battery": int(
            records_per_battery.min()
        ),
        "maximum_records_per_battery": int(
            records_per_battery.max()
        ),
        "median_records_per_battery": float(
            records_per_battery.median()
        ),
        "minimum_cycles_per_battery": int(
            cycles_per_battery.min()
        ),
        "maximum_cycles_per_battery": int(
            cycles_per_battery.max()
        ),
        "median_cycles_per_battery": float(
            cycles_per_battery.median()
        ),
        "duplicate_battery_cycle_rows": (
            duplicate_battery_cycle_rows
        ),
        "cycle_life_ml_eligible_batteries": (
            eligible_batteries
        ),
        "cycle_life_ml_ineligible_batteries": (
            ineligible_batteries
        ),
    }

    return summary


# ============================================================
# BUILD QUALITY WARNINGS
# ============================================================

def build_cycle_feature_warnings(dataframe):
    """
    Create human-readable warnings for cycle-level features.
    """

    warnings = []

    if "charge_time" in dataframe.columns:
        warnings.append(
            "The charge-time unit is not confirmed. "
            "The values are retained without conversion."
        )

    ambiguous_present = [
        column
        for column in AMBIGUOUS_COLUMNS
        if column in dataframe.columns
    ]

    if ambiguous_present:
        warnings.append(
            "Ambiguous columns were preserved without mapping: "
            + ", ".join(ambiguous_present)
            + "."
        )

    if "cycle_life" in dataframe.columns:
        cycle_life_variation = (
            dataframe.groupby(
                "battery_id"
            )["cycle_life"]
            .nunique(
                dropna=True
            )
        )

        varying_battery_ids = (
            cycle_life_variation[
                cycle_life_variation > 1
            ]
            .index
            .astype(str)
            .tolist()
        )

        if varying_battery_ids:
            warnings.append(
                "Conflicting cycle_life targets were found for "
                f"{len(varying_battery_ids)} battery or batteries: "
                + ", ".join(varying_battery_ids)
                + ". These batteries remain available for "
                "descriptive analytics but are marked ineligible "
                "for supervised cycle-life ML training."
            )

    if "coulombic_efficiency_percent" in dataframe.columns:
        efficiency_values = dataframe[
            "coulombic_efficiency_percent"
        ]

        unusual_efficiency = int(
            (
                (
                    efficiency_values < 80
                )
                | (
                    efficiency_values > 120
                )
            ).sum()
        )

        if unusual_efficiency > 0:
            warnings.append(
                f"{unusual_efficiency} rows have coulombic "
                "efficiency outside the broad 80% to 120% "
                "screening range."
            )

    if "average_temperature_c" in dataframe.columns:
        temperature_values = dataframe[
            "average_temperature_c"
        ]

        unusual_temperature = int(
            (
                (
                    temperature_values < -20
                )
                | (
                    temperature_values > 80
                )
            ).sum()
        )

        if unusual_temperature > 0:
            warnings.append(
                f"{unusual_temperature} rows have average "
                "temperature outside -20°C to 80°C."
            )

    if not warnings:
        warnings.append(
            "No major cycle-feature warnings were detected."
        )

    return warnings


# ============================================================
# ADAPT COMPLETE DATASET
# ============================================================

def adapt_cycle_feature_dataset(dataframe):
    """
    Run the complete cycle-feature adaptation process.
    """

    (
        mapped_data,
        applied_mapping,
        schema_report,
    ) = apply_cycle_feature_mapping(
        dataframe
    )

    cleaned_data = clean_cycle_feature_data(
        mapped_data
    )

    feature_data = calculate_cycle_features(
        cleaned_data
    )

    cycle_life_report = (
        build_cycle_life_consistency_report(
            feature_data
        )
    )

    capabilities = detect_cycle_feature_capabilities(
        feature_data
    )

    summary = build_cycle_feature_summary(
        feature_data
    )

    warnings = build_cycle_feature_warnings(
        feature_data
    )

    report = {
        "schema": schema_report,
        "mapping": applied_mapping,
        "summary": summary,
        "capabilities": capabilities,
        "warnings": warnings,
        "cycle_life_consistency": (
            cycle_life_report
        ),
    }

    return feature_data, report


# ============================================================
# SAVE ADAPTED DATA
# ============================================================

def save_adapted_dataset(
    adapted_data,
    report,
    dataset_path,
):
    """
    Save the standardized cycle-feature dataset and reports.
    """

    platform_folder = (
        CORE_FOLDER.parent
    )

    output_directory = (
        platform_folder
        / "outputs"
        / Path(dataset_path).stem
        / "cycle_feature_adapter"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    adapted_path = (
        output_directory
        / "adapted_cycle_features.csv"
    )

    summary_path = (
        output_directory
        / "cycle_feature_summary.csv"
    )

    capability_path = (
        output_directory
        / "cycle_feature_capabilities.csv"
    )

    warning_path = (
        output_directory
        / "cycle_feature_warnings.csv"
    )

    cycle_life_report_path = (
        output_directory
        / "cycle_life_consistency_report.csv"
    )

    inconsistent_target_path = (
        output_directory
        / "inconsistent_cycle_life_records.csv"
    )

    adapted_data.to_csv(
        adapted_path,
        index=False,
    )

    pd.DataFrame(
        [report["summary"]]
    ).to_csv(
        summary_path,
        index=False,
    )

    capability_records = []

    for analysis_name, capability in (
        report["capabilities"].items()
    ):
        capability_records.append(
            {
                "analysis": analysis_name,
                "available": capability[
                    "available"
                ],
                "reason": capability[
                    "reason"
                ],
            }
        )

    pd.DataFrame(
        capability_records
    ).to_csv(
        capability_path,
        index=False,
    )

    pd.DataFrame(
        {
            "warning": report["warnings"],
        }
    ).to_csv(
        warning_path,
        index=False,
    )

    cycle_life_report = report[
        "cycle_life_consistency"
    ]

    cycle_life_report.to_csv(
        cycle_life_report_path,
        index=False,
    )

    if (
        "cycle_life_ml_eligible"
        in adapted_data.columns
    ):
        inconsistent_records = adapted_data[
            ~adapted_data[
                "cycle_life_ml_eligible"
            ]
        ].copy()
    else:
        inconsistent_records = pd.DataFrame()

    inconsistent_records.to_csv(
        inconsistent_target_path,
        index=False,
    )

    return {
        "adapted_data": adapted_path,
        "summary": summary_path,
        "capabilities": capability_path,
        "warnings": warning_path,
        "cycle_life_consistency": (
            cycle_life_report_path
        ),
        "inconsistent_cycle_life_records": (
            inconsistent_target_path
        ),
    }


# ============================================================
# PRINT ADAPTER REPORT
# ============================================================

def print_adapter_report(
    dataset_path,
    report,
    adapted_data,
    saved_files,
):
    """
    Print a readable cycle-feature adaptation report.
    """

    print("\n" + "=" * 75)
    print("CYCLE-FEATURE DATASET ADAPTER")
    print("=" * 75)

    print(
        f"\nDataset: {dataset_path}"
    )

    schema = report["schema"]

    print("\nSCHEMA DETECTION")

    print(
        f"  Detected: "
        f"{schema['schema_detected']}"
    )

    print(
        f"  Confidence: "
        f"{schema['confidence_score']}/100 "
        f"({schema['confidence_level']})"
    )

    print("\nAPPLIED COLUMN MAPPING")

    for original_column, standard_column in (
        report["mapping"].items()
    ):
        print(
            f"  {original_column} -> "
            f"{standard_column}"
        )

    if schema["matched_ambiguous_columns"]:
        print("\nPRESERVED AMBIGUOUS COLUMNS")

        for column in schema[
            "matched_ambiguous_columns"
        ]:
            print(
                f"  {column} -> unchanged"
            )

    summary = report["summary"]

    print("\nDATASET CLASSIFICATION")

    print(
        f"  Type: {summary['dataset_type']}"
    )

    print(
        f"  Rows: {summary['total_rows']:,}"
    )

    print(
        f"  Batteries: "
        f"{summary['battery_count']}"
    )

    print(
        f"  Unique cycle values: "
        f"{summary['unique_cycle_values']}"
    )

    print(
        f"  Records per battery: "
        f"{summary['minimum_records_per_battery']} "
        f"to {summary['maximum_records_per_battery']} "
        f"(median "
        f"{summary['median_records_per_battery']:.1f})"
    )

    print(
        f"  Cycles per battery: "
        f"{summary['minimum_cycles_per_battery']} "
        f"to {summary['maximum_cycles_per_battery']} "
        f"(median "
        f"{summary['median_cycles_per_battery']:.1f})"
    )

    print(
        f"  Duplicate battery-cycle rows: "
        f"{summary['duplicate_battery_cycle_rows']}"
    )

    print("\nCYCLE-LIFE ML ELIGIBILITY")

    print(
        "  Eligible batteries: "
        f"{summary['cycle_life_ml_eligible_batteries']}"
    )

    print(
        "  Ineligible batteries: "
        f"{summary['cycle_life_ml_ineligible_batteries']}"
    )

    cycle_life_report = report[
        "cycle_life_consistency"
    ]

    if not cycle_life_report.empty:
        inconsistent_batteries = (
            cycle_life_report[
                ~cycle_life_report[
                    "ml_eligible"
                ]
            ]
        )

        if not inconsistent_batteries.empty:
            print(
                "  Ineligible battery details:"
            )

            for _, row in (
                inconsistent_batteries.iterrows()
            ):
                print(
                    f"    {row['battery_id']}: "
                    f"{row['unique_target_count']} "
                    "unique cycle_life values, "
                    f"range={row['cycle_life_range']:.2f}"
                )

    print("\nDERIVED FEATURES")

    derived_features = [
        "coulombic_efficiency_percent",
        "temperature_spread_c",
        "reference_discharge_capacity_ah",
        "discharge_capacity_retention_percent",
        "cycle_soh_percent",
        "cycle_life_unique_target_count",
        "cycle_life_target_consistent",
        "cycle_life_ml_eligible",
    ]

    for feature in derived_features:
        if feature in adapted_data.columns:
            print(f"  {feature}")

    print("\nANALYSIS CAPABILITIES")

    for analysis_name, capability in (
        report["capabilities"].items()
    ):
        status = (
            "Available"
            if capability["available"]
            else "Not available"
        )

        print(
            f"  {analysis_name.upper()}: {status}"
        )

        print(
            f"    Reason: "
            f"{capability['reason']}"
        )

    print("\nWARNINGS")

    for warning in report["warnings"]:
        print(f"  - {warning}")

    print("\nSAVED RESULTS")

    for result_name, result_path in (
        saved_files.items()
    ):
        print(
            f"  {result_name}: {result_path}"
        )

    print("\nIMPORTANT")

    print(
        "  This adapter supports cycle-level feature data. "
        "It does not create missing raw voltage, current, or "
        "time-series measurements."
    )

    print(
        "  SOC, ICA, DVA, and voltage-curve analysis remain "
        "unavailable for this dataset."
    )

    print(
        "  C1, Q1, C2, and the charge-time unit remain "
        "uninterpreted until dataset documentation confirms "
        "their meanings."
    )

    print(
        "  Batteries with inconsistent cycle_life targets "
        "remain usable for descriptive analytics but must be "
        "excluded from supervised cycle-life ML training."
    )

    print("=" * 75)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":
    dataset_path = input(
        "Paste the full path of a cycle-feature dataset: "
    ).strip().strip('"')

    try:
        original_data = load_battery_dataset(
            dataset_path
        )

        adapted_data, adapter_report = (
            adapt_cycle_feature_dataset(
                original_data
            )
        )

        saved_results = save_adapted_dataset(
            adapted_data,
            adapter_report,
            dataset_path,
        )

        print_adapter_report(
            dataset_path,
            adapter_report,
            adapted_data,
            saved_results,
        )

    except Exception as error:
        print(
            f"\nCycle-feature adaptation failed: {error}"
        )