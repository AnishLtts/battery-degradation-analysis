from analysis_router import (
    should_run_soh,
    should_run_degradation,
    should_run_ica,
    should_run_dva,
)
from analysis_runner import (
    run_soh_analysis,
    run_degradation_analysis,
    run_ica_analysis,
    run_dva_analysis,
)

def execute_analysis_plan(
    capability_report,
):
    """
    Decide which analyses should run.
    """

    print("\n" + "=" * 60)
    print("EXECUTION ENGINE")
    print("=" * 60)

    if should_run_soh(
      capability_report
):
      run_soh_analysis()

    if should_run_degradation(
        capability_report
    ):
        run_degradation_analysis()

    if should_run_ica(
        capability_report
    ):
        run_ica_analysis()

    if should_run_dva(
        capability_report
    ):
        run_dva_analysis()

    print("=" * 60)