def get_available_analysis_names(
    capability_report
):
    """
    Extract analysis names that are supported
    by the current dataset.
    """

    return list(
        capability_report[
            "available_analyses"
        ].keys()
    )


def print_execution_plan(
    capability_report
):
    """
    Show which analyses will be executed.
    """

    available = (
        get_available_analysis_names(
            capability_report
        )
    )

    print("\n" + "=" * 60)
    print("AUTOMATIC ANALYSIS EXECUTION PLAN")
    print("=" * 60)

    for analysis in available:
        print(f"✅ {analysis}")

    print("=" * 60)

    return available


def should_run_soh(
    capability_report
):
    available = (
        capability_report[
            "available_analyses"
        ]
    )

    return (
        "Capacity-Based SOH Analysis"
        in available
    )


def should_run_degradation(
    capability_report
):
    available = (
        capability_report[
            "available_analyses"
        ]
    )

    return (
        "Capacity Degradation Analysis"
        in available
    )


def should_run_ica(
    capability_report
):
    available = (
        capability_report[
            "available_analyses"
        ]
    )

    return (
        "Incremental Capacity Analysis"
        in available
    )


def should_run_dva(
    capability_report
):
    available = (
        capability_report[
            "available_analyses"
        ]
    )

    return (
        "Differential Voltage Analysis"
        in available
    )


def should_run_coulomb_capacity(
    capability_report
):
    available = (
        capability_report[
            "available_analyses"
        ]
    )

    return (
        "Coulomb-Counting Capacity Estimate"
        in available
    )


def should_run_soc(
    capability_report
):
    available = (
        capability_report[
            "available_analyses"
        ]
    )

    return (
        "Coulomb-Counting SOC Estimation"
        in available
    )