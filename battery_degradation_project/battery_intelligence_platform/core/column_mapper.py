from pathlib import Path
import json
import re


# ============================================================
# COLUMN ALIASES
# ============================================================

COLUMN_ALIASES = {
    "time_s": [
        "time",
        "time_s",
        "time_sec",
        "time_secs",
        "time_second",
        "time_seconds",
        "test_time",
        "test_time_s",
        "elapsed_time",
        "elapsed_time_s",
        "elapsed_seconds",
        "relative_time",
        "relative_time_s",
        "record_time",
        "sample_time",
        "duration_s",
        "seconds",
        "sec",
        "t_s",
    ],

    "voltage_v": [
        "voltage",
        "voltage_v",
        "volt",
        "volts",
        "v",
        "v_cell",
        "vcell",
        "cell_v",
        "u_cell",
        "ucell",
        "voltage_measured",
        "measured_voltage",
        "terminal_voltage",
        "terminal_voltage_v",
        "cell_voltage",
        "cell_voltage_v",
        "battery_voltage",
        "battery_voltage_v",
        "voltage_load",
        "load_voltage",
        "pack_voltage",
        "pack_voltage_v",
    ],

    "current_a": [
        "current",
        "current_a",
        "amp",
        "amps",
        "ampere",
        "amperes",
        "i",
        "i_cell",
        "icell",
        "cell_i",
        "i_load",
        "iload",
        "current_measured",
        "measured_current",
        "battery_current",
        "battery_current_a",
        "cell_current",
        "cell_current_a",
        "current_load",
        "load_current",
        "current_charge",
        "charge_current",
        "discharge_current",
        "pack_current",
        "pack_current_a",
    ],

    "temperature_c": [
        "temperature",
        "temperature_c",
        "temp",
        "temp_c",
        "t_cell",
        "tcell",
        "cell_t",
        "temperature_measured",
        "measured_temperature",
        "battery_temperature",
        "battery_temperature_c",
        "cell_temperature",
        "cell_temperature_c",
        "surface_temperature",
        "surface_temperature_c",
        "case_temperature",
        "case_temperature_c",
    ],

    "ambient_temperature_c": [
        "ambient_temperature",
        "ambient_temperature_c",
        "ambient_temp",
        "ambient_temp_c",
        "room_temperature",
        "room_temperature_c",
        "environment_temperature",
        "environment_temperature_c",
        "chamber_temperature",
        "chamber_temperature_c",
    ],

    "capacity_ah": [
        "capacity",
        "capacity_ah",
        "capacity_amp_hour",
        "capacity_amp_hours",
        "q",
        "q_ah",
        "q_discharge",
        "q_charge",
        "qdischarge",
        "qcharge",
        "discharge_capacity",
        "discharge_capacity_ah",
        "charge_capacity",
        "charge_capacity_ah",
        "measured_capacity",
        "measured_capacity_ah",
        "available_capacity",
        "available_capacity_ah",
        "cell_capacity",
        "cell_capacity_ah",
        "battery_capacity",
        "battery_capacity_ah",
        "cumulative_capacity",
        "cumulative_capacity_ah",
    ],

    "cycle": [
        "cycle",
        "cycles",
        "cycle_no",
        "cycle_num",
        "cycle_number",
        "cycle_index",
        "cycle_count",
        "cycle_id",
        "cycleno",
        "cyclenum",
        "cycleindex",
        "cyclecount",
        "n_cycle",
    ],

    "soc_percent": [
        "soc",
        "soc_percent",
        "soc_percentage",
        "state_of_charge",
        "state_of_charge_percent",
        "state_of_charge_percentage",
    ],

    "soh_percent": [
        "soh",
        "soh_percent",
        "soh_percentage",
        "state_of_health",
        "state_of_health_percent",
        "state_of_health_percentage",
    ],

    "battery_id": [
        "battery_id",
        "battery",
        "battery_name",
        "battery_number",
        "battery_no",
        "battery_index",
        "cell_id",
        "cell",
        "cell_name",
        "cell_number",
        "cell_no",
        "cell_index",
        "sample_id",
        "device_id",
    ],

    "test_id": [
        "test_id",
        "test_number",
        "test_no",
        "test_index",
        "experiment_id",
        "experiment_number",
        "experiment_no",
        "run_id",
        "run_number",
    ],

    "test_type": [
        "type",
        "test_type",
        "operation",
        "operation_type",
        "step_type",
        "step_name",
        "mode",
        "status",
        "phase",
        "battery_state",
        "charge_discharge",
        "charge_discharge_state",
    ],

    "drive_cycle": [
        "drive_cycle",
        "drivecycle",
        "driving_cycle",
        "drive_profile",
        "driving_profile",
        "test_profile",
        "profile",
        "load_profile",
        "current_profile",
    ],

    "source_file": [
        "filename",
        "file_name",
        "source_file",
        "source_filename",
        "source",
        "input_file",
        "data_file",
    ],

    "electrolyte_resistance_ohm": [
        "re",
        "re_ohm",
        "electrolyte_resistance",
        "electrolyte_resistance_ohm",
        "ohmic_resistance",
        "ohmic_resistance_ohm",
    ],

    "charge_transfer_resistance_ohm": [
        "rct",
        "rct_ohm",
        "charge_transfer_resistance",
        "charge_transfer_resistance_ohm",
    ],

    "internal_resistance_ohm": [
        "resistance",
        "resistance_ohm",
        "internal_resistance",
        "internal_resistance_ohm",
        "dcir",
        "dcir_ohm",
        "cell_resistance",
        "cell_resistance_ohm",
    ],

    "rul_cycles": [
        "rul",
        "rul_cycles",
        "remaining_useful_life",
        "remaining_useful_life_cycles",
        "remaining_cycles",
        "cycles_remaining",
    ],
}


# ============================================================
# INTERACTIVE MAPPING CONFIGURATION
# ============================================================

INTERACTIVE_TARGETS = [
    (
        "cycle",
        "Cycle number or cycle index",
    ),
    (
        "time_s",
        "Time in seconds",
    ),
    (
        "voltage_v",
        "Voltage in volts",
    ),
    (
        "current_a",
        "Current in amperes",
    ),
    (
        "temperature_c",
        "Cell or battery temperature in Celsius",
    ),
    (
        "ambient_temperature_c",
        "Ambient temperature in Celsius",
    ),
    (
        "capacity_ah",
        "Capacity in ampere-hours",
    ),
    (
        "battery_id",
        "Battery or cell identifier",
    ),
    (
        "test_type",
        "Charge, discharge, rest, or operation type",
    ),
    (
        "soc_percent",
        "State of charge percentage",
    ),
    (
        "soh_percent",
        "State of health percentage",
    ),
    (
        "internal_resistance_ohm",
        "Internal resistance in ohms",
    ),
    (
        "rul_cycles",
        "Remaining useful life in cycles",
    ),
]

MAPPING_CONFIG_FILENAME = (
    "saved_column_mappings.json"
)


# ============================================================
# COLUMN-NAME NORMALIZATION
# ============================================================

def normalize_column_name(column_name):
    """
    Convert a source column name into a consistent format.

    Examples:
        Cycle Index -> cycle_index
        Voltage (V) -> voltage_v
        SOC (%) -> soc_percent
    """
    normalized = str(
        column_name
    ).strip().lower()

    replacements = {
        "%": "percent",
        "°": "",
        "Â°": "",
        "µ": "u",
        "μ": "u",
        "(": "_",
        ")": "",
        "[": "_",
        "]": "",
        "{": "_",
        "}": "",
        "/": "_",
        "\\": "_",
    }

    for old_value, new_value in (
        replacements.items()
    ):
        normalized = normalized.replace(
            old_value,
            new_value,
        )

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


# ============================================================
# ALIAS LOOKUP
# ============================================================

def create_alias_lookup():
    """
    Create a normalized alias-to-standard-name lookup.
    """
    alias_lookup = {}

    for standard_name, aliases in (
        COLUMN_ALIASES.items()
    ):
        all_aliases = list(aliases)

        if standard_name not in all_aliases:
            all_aliases.append(
                standard_name
            )

        for alias in all_aliases:
            normalized_alias = (
                normalize_column_name(alias)
            )

            existing_standard = (
                alias_lookup.get(
                    normalized_alias
                )
            )

            if (
                existing_standard is not None
                and existing_standard
                != standard_name
            ):
                raise ValueError(
                    "Alias conflict detected: "
                    f"'{alias}' maps to both "
                    f"'{existing_standard}' and "
                    f"'{standard_name}'."
                )

            alias_lookup[
                normalized_alias
            ] = standard_name

    return alias_lookup


# ============================================================
# SAVED MAPPING FILE
# ============================================================

def get_default_mapping_file():
    """
    Return the default location for saved manual mappings.
    """
    return (
        Path(__file__).resolve().parent
        / MAPPING_CONFIG_FILENAME
    )


def load_saved_mappings(mapping_file=None):
    """
    Load previously saved manual column mappings.

    The JSON keys are normalized source names and the values are
    standard target names or null for ignored columns.
    """
    if mapping_file is None:
        mapping_file = (
            get_default_mapping_file()
        )

    mapping_file = Path(
        mapping_file
    )

    if not mapping_file.exists():
        return {}

    try:
        with mapping_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            loaded_data = json.load(file)

    except (
        json.JSONDecodeError,
        OSError,
    ) as error:
        print(
            "\nWarning: Saved column mappings "
            f"could not be loaded: {error}"
        )

        return {}

    if not isinstance(
        loaded_data,
        dict,
    ):
        print(
            "\nWarning: Saved column mapping file "
            "does not contain a JSON object."
        )

        return {}

    valid_targets = set(
        COLUMN_ALIASES.keys()
    )

    valid_mappings = {}

    for source_name, target_name in (
        loaded_data.items()
    ):
        normalized_source = (
            normalize_column_name(
                source_name
            )
        )

        if target_name is None:
            valid_mappings[
                normalized_source
            ] = None

        elif target_name in valid_targets:
            valid_mappings[
                normalized_source
            ] = target_name

    return valid_mappings


def save_manual_mappings(
    manual_mappings,
    mapping_file=None,
):
    """
    Merge and save manual mappings as JSON.
    """
    if mapping_file is None:
        mapping_file = (
            get_default_mapping_file()
        )

    mapping_file = Path(
        mapping_file
    )

    mapping_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_mappings = (
        load_saved_mappings(
            mapping_file
        )
    )

    for source_name, target_name in (
        manual_mappings.items()
    ):
        normalized_source = (
            normalize_column_name(
                source_name
            )
        )

        existing_mappings[
            normalized_source
        ] = target_name

    with mapping_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            existing_mappings,
            file,
            indent=4,
            sort_keys=True,
        )

    return mapping_file


# ============================================================
# AUTOMATIC COLUMN MAPPING
# ============================================================

def detect_column_mapping(
    dataframe,
    saved_mappings=None,
):
    """
    Automatically map known aliases and previously saved manual
    mappings.

    Returns:
        mapping:
            Dictionary of original names to standard names.

        unknown_columns:
            Columns that remain unrecognized, were ignored, or
            conflicted with a target already in use.
    """
    alias_lookup = (
        create_alias_lookup()
    )

    if saved_mappings is None:
        saved_mappings = (
            load_saved_mappings()
        )

    mapping = {}
    unknown_columns = []
    used_standard_names = set()

    for original_column in (
        dataframe.columns
    ):
        normalized_column = (
            normalize_column_name(
                original_column
            )
        )

        standard_name = (
            alias_lookup.get(
                normalized_column
            )
        )

        if standard_name is None:
            standard_name = (
                saved_mappings.get(
                    normalized_column
                )
            )

        if standard_name is None:
            unknown_columns.append(
                original_column
            )

            continue

        if (
            standard_name
            in used_standard_names
        ):
            unknown_columns.append(
                original_column
            )

            continue

        mapping[
            original_column
        ] = standard_name

        used_standard_names.add(
            standard_name
        )

    return mapping, unknown_columns


# ============================================================
# INTERACTIVE COLUMN MAPPING
# ============================================================

def print_interactive_choices(
    used_standard_names,
):
    """
    Print available mapping targets and return a number-to-target
    lookup.
    """
    choice_lookup = {}
    choice_number = 1

    print(
        "\nAvailable meanings:"
    )

    for standard_name, description in (
        INTERACTIVE_TARGETS
    ):
        if (
            standard_name
            in used_standard_names
        ):
            continue

        print(
            f"  {choice_number}. "
            f"{standard_name} - "
            f"{description}"
        )

        choice_lookup[
            choice_number
        ] = standard_name

        choice_number += 1

    print(
        f"  {choice_number}. Ignore this column"
    )

    choice_lookup[
        choice_number
    ] = None

    return choice_lookup


def collect_interactive_mapping(
    dataframe,
    automatic_mapping,
    unknown_columns,
    save_choices=True,
    mapping_file=None,
):
    """
    Ask the user to map unknown columns safely.

    Existing automatic mappings are preserved. A standard target
    cannot be assigned to two source columns.
    """
    final_mapping = dict(
        automatic_mapping
    )

    remaining_unknown = []

    used_standard_names = set(
        final_mapping.values()
    )

    manual_choices = {}

    if not unknown_columns:
        return (
            final_mapping,
            remaining_unknown,
        )

    print("\n" + "=" * 70)
    print("INTERACTIVE COLUMN MAPPING")
    print("=" * 70)

    print(
        "\nSome columns were not recognized automatically."
    )

    print(
        "Map only columns whose meaning and unit are known."
    )

    print(
        "Choose Ignore for metadata, comments, version markers, "
        "or unclear columns."
    )

    for original_column in (
        unknown_columns
    ):
        normalized_column = (
            normalize_column_name(
                original_column
            )
        )

        print(
            "\n" + "-" * 70
        )

        print(
            f"Unrecognized column: "
            f"{original_column}"
        )

        try:
            sample_values = (
                dataframe[
                    original_column
                ]
                .dropna()
                .astype(str)
                .head(5)
                .tolist()
            )

        except Exception:
            sample_values = []

        if sample_values:
            print(
                "Sample values: "
                + ", ".join(sample_values)
            )

        else:
            print(
                "Sample values: No non-empty values"
            )

        choice_lookup = (
            print_interactive_choices(
                used_standard_names
            )
        )

        selected_target = None
        valid_choice_received = False

        while not valid_choice_received:
            user_choice = input(
                "Enter the option number: "
            ).strip()

            try:
                choice_number = int(
                    user_choice
                )

            except ValueError:
                print(
                    "Please enter a valid option number."
                )

                continue

            if choice_number not in (
                choice_lookup
            ):
                print(
                    "The selected option is not available."
                )

                continue

            selected_target = (
                choice_lookup[
                    choice_number
                ]
            )

            valid_choice_received = True

        manual_choices[
            normalized_column
        ] = selected_target

        if selected_target is None:
            remaining_unknown.append(
                original_column
            )

            print(
                f"Ignored: {original_column}"
            )

            continue

        final_mapping[
            original_column
        ] = selected_target

        used_standard_names.add(
            selected_target
        )

        print(
            f"Mapped: {original_column} "
            f"-> {selected_target}"
        )

    if (
        save_choices
        and manual_choices
    ):
        saved_path = (
            save_manual_mappings(
                manual_choices,
                mapping_file,
            )
        )

        print(
            "\nManual mapping choices saved to:"
        )

        print(
            saved_path
        )

    print("=" * 70)

    return (
        final_mapping,
        remaining_unknown,
    )


def detect_column_mapping_with_fallback(
    dataframe,
    interactive=True,
    save_choices=True,
    mapping_file=None,
):
    """
    Run automatic detection and optionally launch interactive
    mapping for unrecognized columns.

    Use this function in command-line pipelines that support user
    input. For automated tests and web applications, use
    interactive=False.
    """
    saved_mappings = (
        load_saved_mappings(
            mapping_file
        )
    )

    mapping, unknown_columns = (
        detect_column_mapping(
            dataframe,
            saved_mappings,
        )
    )

    if (
        interactive
        and unknown_columns
    ):
        mapping, unknown_columns = (
            collect_interactive_mapping(
                dataframe,
                mapping,
                unknown_columns,
                save_choices=save_choices,
                mapping_file=mapping_file,
            )
        )

    return mapping, unknown_columns


# ============================================================
# APPLY COLUMN MAPPING
# ============================================================

def apply_column_mapping(
    dataframe,
    mapping,
):
    """
    Rename source columns using the supplied mapping.
    """
    standardized_data = (
        dataframe.copy()
    )

    duplicate_targets = {
        target_name
        for target_name in mapping.values()
        if list(
            mapping.values()
        ).count(target_name) > 1
    }

    if duplicate_targets:
        raise ValueError(
            "Multiple columns were mapped to the same "
            "standard name: "
            + ", ".join(
                sorted(
                    duplicate_targets
                )
            )
        )

    standardized_data = (
        standardized_data.rename(
            columns=mapping
        )
    )

    return standardized_data


# ============================================================
# MAPPING REPORT
# ============================================================

def print_mapping_report(
    mapping,
    unknown_columns,
    standardized_data,
):
    """
    Print recognized, ignored, and standardized columns.
    """
    print("\n" + "=" * 70)
    print("UNIVERSAL BATTERY COLUMN MAPPING")
    print("=" * 70)

    print("\nRECOGNIZED AND MAPPED COLUMNS")

    if mapping:
        for original, standard in (
            mapping.items()
        ):
            print(
                f"  {original} -> {standard}"
            )

    else:
        print(
            "  No recognized battery columns."
        )

    print("\nUNMAPPED OR IGNORED COLUMNS")

    if unknown_columns:
        for column in unknown_columns:
            print(
                f"  - {column}"
            )

    else:
        print(
            "  No unmapped columns remain."
        )

    print("\nFINAL STANDARDIZED COLUMNS")

    for column in (
        standardized_data.columns
    ):
        print(
            f"  - {column}"
        )

    print("=" * 70)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":
    from file_reader import (
        load_battery_dataset,
    )

    dataset_path = input(
        "Paste the full path of a battery dataset: "
    ).strip().strip('"').strip("'")

    if not dataset_path:
        print(
            "\nColumn mapping failed: "
            "No dataset path was supplied."
        )

        raise SystemExit(1)

    try:
        battery_data = (
            load_battery_dataset(
                dataset_path
            )
        )

        mapping, unknown_columns = (
            detect_column_mapping_with_fallback(
                battery_data,
                interactive=True,
                save_choices=True,
            )
        )

        standardized_data = (
            apply_column_mapping(
                battery_data,
                mapping,
            )
        )

        print_mapping_report(
            mapping,
            unknown_columns,
            standardized_data,
        )

    except Exception as error:
        print(
            f"\nColumn mapping failed: {error}"
        )

        raise SystemExit(1)