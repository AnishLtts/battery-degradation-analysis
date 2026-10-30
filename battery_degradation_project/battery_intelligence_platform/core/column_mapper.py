import re


COLUMN_ALIASES = {
    "time_s": [
        "time",
        "time_s",
        "time_sec",
        "time_seconds",
        "elapsed_time",
        "test_time",
        "timestamp",
    ],

    "voltage_v": [
        "voltage",
        "voltage_v",
        "voltage_measured",
        "terminal_voltage",
        "cell_voltage",
        "voltage_load",
    ],

    "current_a": [
        "current",
        "current_a",
        "current_measured",
        "battery_current",
        "current_load",
        "current_charge",
    ],

    "temperature_c": [
        "temperature",
        "temperature_c",
        "temperature_measured",
        "battery_temperature",
        "cell_temperature",
        "temp",
        "temp_c",
    ],

    "ambient_temperature_c": [
        "ambient_temperature",
        "ambient_temperature_c",
        "ambient_temp",
        "ambient_temp_c",
    ],

    "capacity_ah": [
        "capacity",
        "capacity_ah",
        "discharge_capacity",
        "charge_capacity",
        "measured_capacity",
    ],

    "cycle": [
        "cycle",
        "cycle_number",
        "cycle_index",
        "cycle_count",
    ],

    "soc_percent": [
        "soc",
        "soc_percent",
        "state_of_charge",
        "state_of_charge_percent",
    ],

    "soh_percent": [
        "soh",
        "soh_percent",
        "state_of_health",
        "state_of_health_percent",
    ],

    "battery_id": [
        "battery_id",
        "battery",
        "cell_id",
        "cell",
    ],

    "test_id": [
        "test_id",
        "test_number",
        "experiment_id",
    ],

    "test_type": [
        "type",
        "test_type",
        "operation",
        "operation_type",
        "step_type",
    ],

        "drive_cycle": [
        "drive_cycle",
        "drivecycle",
        "driving_cycle",
        "drive_profile",
        "test_profile",
        "profile",
    ],

    "source_file": [
        "filename",
        "file_name",
        "source_file",
        "source",
    ],

    "electrolyte_resistance_ohm": [
        "re",
        "electrolyte_resistance",
        "ohmic_resistance",
    ],

    "charge_transfer_resistance_ohm": [
        "rct",
        "charge_transfer_resistance",
    ],

    "internal_resistance_ohm": [
        "resistance",
        "internal_resistance",
        "dcir",
        "resistance_ohm",
    ],

    "rul_cycles": [
        "rul",
        "rul_cycles",
        "remaining_useful_life",
        "remaining_cycles",
    ],
}


def normalize_column_name(column_name):
    normalized = str(column_name).strip().lower()

    normalized = normalized.replace("%", "percent")
    normalized = normalized.replace("°", "")
    normalized = normalized.replace("(", "_")
    normalized = normalized.replace(")", "")
    normalized = normalized.replace("/", "_")

    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        normalized,
    )

    normalized = re.sub(
        r"_+",
        "_",
        normalized,
    )

    return normalized.strip("_")


def create_alias_lookup():
    alias_lookup = {}

    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            normalized_alias = normalize_column_name(alias)
            alias_lookup[normalized_alias] = standard_name

    return alias_lookup


def detect_column_mapping(dataframe):
    alias_lookup = create_alias_lookup()

    mapping = {}
    unknown_columns = []
    used_standard_names = set()

    for original_column in dataframe.columns:
        normalized_column = normalize_column_name(
            original_column
        )

        standard_name = alias_lookup.get(
            normalized_column
        )

        if standard_name is None:
            unknown_columns.append(original_column)
            continue

        if standard_name in used_standard_names:
            unknown_columns.append(original_column)
            continue

        mapping[original_column] = standard_name
        used_standard_names.add(standard_name)

    return mapping, unknown_columns


def apply_column_mapping(dataframe, mapping):
    standardized_data = dataframe.copy()

    standardized_data = standardized_data.rename(
        columns=mapping
    )

    return standardized_data


def print_mapping_report(
    mapping,
    unknown_columns,
    standardized_data,
):
    print("\n" + "=" * 70)
    print("UNIVERSAL BATTERY COLUMN MAPPING")
    print("=" * 70)

    print("\nRECOGNIZED COLUMNS")

    if mapping:
        for original, standard in mapping.items():
            print(f"  {original} -> {standard}")
    else:
        print("  No recognized battery columns.")

    print("\nUNRECOGNIZED COLUMNS")

    if unknown_columns:
        for column in unknown_columns:
            print(f"  - {column}")
    else:
        print("  All columns were recognized.")

    print("\nSTANDARDIZED COLUMNS")

    for column in standardized_data.columns:
        print(f"  - {column}")

    print("=" * 70)


if __name__ == "__main__":
    from file_reader import load_battery_dataset

    dataset_path = input(
        "Paste the full path of a battery dataset: "
    ).strip().strip('"')

    try:
        battery_data = load_battery_dataset(
            dataset_path
        )

        mapping, unknown = detect_column_mapping(
            battery_data
        )

        standardized_data = apply_column_mapping(
            battery_data,
            mapping,
        )

        print_mapping_report(
            mapping,
            unknown,
            standardized_data,
        )

        print("\nFIRST FIVE STANDARDIZED ROWS")
        print(standardized_data.head())

    except Exception as error:
        print(f"\nColumn mapping failed: {error}")