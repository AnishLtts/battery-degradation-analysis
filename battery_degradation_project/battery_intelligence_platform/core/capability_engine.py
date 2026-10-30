ANALYSIS_REQUIREMENTS = {
    "Dataset Overview": {
        "required_all": [],
        "required_any": [],
        "description": "Displays dataset size, columns, missing values, and statistics.",
    },

    "Voltage Analysis": {
        "required_all": ["voltage_v"],
        "required_any": [],
        "description": "Analyzes battery voltage values and trends.",
    },

    "Current Analysis": {
        "required_all": ["current_a"],
        "required_any": [],
        "description": "Analyzes charge and discharge current behavior.",
    },

    "Thermal Analysis": {
        "required_all": [],
        "required_any": [
            "temperature_c",
            "ambient_temperature_c",
        ],
        "description": "Analyzes battery and ambient temperature behavior.",
    },

    "Existing SOC Analysis": {
        "required_all": ["soc_percent"],
        "required_any": [],
        "description": "Analyzes SOC values already provided by the dataset.",
    },

    "Coulomb-Counting SOC Estimation": {
        "required_all": [
            "time_s",
            "current_a",
        ],
        "required_any": [],
        "additional_input": [
            "initial_soc",
            "rated_capacity_ah",
        ],
        "description": (
            "Estimates SOC by integrating current over time. "
            "Initial SOC and rated capacity must be supplied."
        ),
    },

    "Capacity-Based SOH Analysis": {
        "required_all": ["capacity_ah"],
        "required_any": [],
        "additional_input": ["reference_capacity_ah"],
        "description": (
            "Calculates SOH from measured capacity and "
            "reference capacity."
        ),
    },

    "Existing SOH Analysis": {
        "required_all": ["soh_percent"],
        "required_any": [],
        "description": "Analyzes SOH values already provided by the dataset.",
    },

    "Capacity Degradation Analysis": {
        "required_all": ["capacity_ah"],
        "required_any": [
            "cycle",
            "test_id",
            "battery_id",
        ],
        "description": "Analyzes battery capacity reduction across tests or cycles.",
    },

    "Electrolyte Resistance Analysis": {
        "required_all": ["electrolyte_resistance_ohm"],
        "required_any": [],
        "description": "Analyzes electrolyte and ohmic resistance trends.",
    },

    "Charge-Transfer Resistance Analysis": {
        "required_all": ["charge_transfer_resistance_ohm"],
        "required_any": [],
        "description": "Analyzes charge-transfer resistance trends.",
    },

    "Internal Resistance Analysis": {
        "required_all": ["internal_resistance_ohm"],
        "required_any": [],
        "description": "Analyzes internal resistance and resistance growth.",
    },

    "Incremental Capacity Analysis": {
        "required_all": [
            "voltage_v",
            "capacity_ah",
        ],
        "required_any": [],
        "description": "Calculates the incremental-capacity curve dQ/dV.",
    },

    "Differential Voltage Analysis": {
        "required_all": [
            "voltage_v",
            "capacity_ah",
        ],
        "required_any": [],
        "description": "Calculates the differential-voltage curve dV/dQ.",
    },

    "Energy Analysis": {
        "required_all": [
            "time_s",
            "voltage_v",
            "current_a",
        ],
        "required_any": [],
        "description": "Calculates power and integrates energy over time.",
    },

    "Time-Series Analysis": {
        "required_all": ["time_s"],
        "required_any": [
            "voltage_v",
            "current_a",
            "temperature_c",
            "soc_percent",
        ],
        "description": "Analyzes available battery measurements over time.",
    },

    "Drive-Cycle Comparison": {
        "required_all": ["drive_cycle"],
        "required_any": [
            "voltage_v",
            "current_a",
            "temperature_c",
            "soc_percent",
        ],
        "description": "Compares battery behavior across drive-cycle profiles.",
    },

    "Battery Comparison": {
        "required_all": ["battery_id"],
        "required_any": [
            "capacity_ah",
            "electrolyte_resistance_ohm",
            "charge_transfer_resistance_ohm",
            "soh_percent",
        ],
        "description": "Compares measurements from multiple batteries.",
    },

    "RUL Analysis": {
        "required_all": ["rul_cycles"],
        "required_any": [],
        "description": "Analyzes an existing remaining-useful-life label.",
    },

    "Fault-Label Analysis": {
        "required_all": ["fault_label"],
        "required_any": [],
        "description": "Analyzes existing fault or anomaly labels.",
    },
}


def check_capability(available_columns, requirements):
    """
    Check whether a dataset has the columns required
    for one battery analysis.
    """

    available_columns = set(available_columns)

    required_all = requirements.get("required_all", [])
    required_any = requirements.get("required_any", [])

    missing_required = [
        column
        for column in required_all
        if column not in available_columns
    ]

    found_optional = [
        column
        for column in required_any
        if column in available_columns
    ]

    missing_optional = [
        column
        for column in required_any
        if column not in available_columns
    ]

    has_all_required = not missing_required

    has_any_optional = (
        not required_any
        or bool(found_optional)
    )

    is_available = (
        has_all_required
        and has_any_optional
    )

    return {
        "available": is_available,
        "missing_required": missing_required,
        "found_optional": found_optional,
        "missing_optional": missing_optional,
        "additional_input": requirements.get(
            "additional_input",
            [],
        ),
    }


def evaluate_dataset_capabilities(dataframe):
    """
    Determine which analyses are supported by
    the standardized battery dataset.
    """

    available_columns = dataframe.columns.tolist()

    report = {
        "available_columns": available_columns,
        "available_analyses": {},
        "unavailable_analyses": {},
    }

    for analysis_name, requirements in ANALYSIS_REQUIREMENTS.items():
        result = check_capability(
            available_columns,
            requirements,
        )

        result["description"] = requirements["description"]

        if result["available"]:
            report["available_analyses"][analysis_name] = result
        else:
            report["unavailable_analyses"][analysis_name] = result

    return report


def create_missing_data_reason(result):
    """
    Explain why an analysis cannot be performed.
    """

    reasons = []

    if result["missing_required"]:
        missing_columns = ", ".join(
            result["missing_required"]
        )

        reasons.append(
            f"Missing required columns: {missing_columns}."
        )

    if (
        not result["found_optional"]
        and result["missing_optional"]
    ):
        possible_columns = ", ".join(
            result["missing_optional"]
        )

        reasons.append(
            "At least one of these columns is required: "
            f"{possible_columns}."
        )

    if not reasons:
        reasons.append(
            "Dataset requirements were not satisfied."
        )

    return " ".join(reasons)


def print_capability_report(report):
    """
    Display available and unavailable analyses.
    """

    print("\n" + "=" * 75)
    print("UNIVERSAL BATTERY DATASET CAPABILITY REPORT")
    print("=" * 75)

    print("\nSTANDARDIZED COLUMNS")

    for column in report["available_columns"]:
        print(f"  - {column}")

    print("\nAVAILABLE ANALYSES")

    for analysis_name, result in report[
        "available_analyses"
    ].items():
        print(f"\n  [AVAILABLE] {analysis_name}")
        print(f"      {result['description']}")

        if result["additional_input"]:
            additional = ", ".join(
                result["additional_input"]
            )

            print(
                "      Additional user input required: "
                f"{additional}"
            )

    print("\nUNAVAILABLE ANALYSES")

    for analysis_name, result in report[
        "unavailable_analyses"
    ].items():
        reason = create_missing_data_reason(result)

        print(f"\n  [UNAVAILABLE] {analysis_name}")
        print(f"      Reason: {reason}")

    print("\nSUMMARY")
    print(
        "  Available analyses   : "
        f"{len(report['available_analyses'])}"
    )
    print(
        "  Unavailable analyses : "
        f"{len(report['unavailable_analyses'])}"
    )

    print("=" * 75)


if __name__ == "__main__":
    from file_reader import load_battery_dataset

    from column_mapper import (
        apply_column_mapping,
        detect_column_mapping,
    )

    dataset_path = input(
        "Paste the full path of a battery dataset: "
    ).strip().strip('"')

    try:
        battery_data = load_battery_dataset(
            dataset_path
        )

        mapping, unknown_columns = detect_column_mapping(
            battery_data
        )

        standardized_data = apply_column_mapping(
            battery_data,
            mapping,
        )

        capability_report = evaluate_dataset_capabilities(
            standardized_data
        )

        print_capability_report(
            capability_report
        )

        if unknown_columns:
            print("\nUNRECOGNIZED ORIGINAL COLUMNS")

            for column in unknown_columns:
                print(f"  - {column}")

    except Exception as error:
        print(f"\nCapability analysis failed: {error}")