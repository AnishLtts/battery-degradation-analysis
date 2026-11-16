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
    normalized_data,
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
       soh_results = run_soh_analysis(
          normalized_data,
        )

       print("\n" + "=" * 60)
       print("BATTERY INTELLIGENCE SUMMARY")
       print("=" * 60)

       print(
        f"Cycles Analyzed: {len(soh_results)}"
       )

       latest_soh = (
        soh_results["soh_percent"].iloc[-1]
       )

       health_score = round(latest_soh)
       print(
        f"Latest SOH: {latest_soh:.2f}%"
       )
       if latest_soh >= 90:
        battery_grade = "A"
       elif latest_soh >= 80:
        battery_grade = "B"
       elif latest_soh >= 70:
        battery_grade = "C"
       elif latest_soh >= 60:
        battery_grade = "D"
       else:
        battery_grade = "F"

       if latest_soh >= 90:
         lifecycle_status = "New"
       elif latest_soh >= 80:
         lifecycle_status = "Healthy"
       elif latest_soh >= 70:
         lifecycle_status = "Aging"
       elif latest_soh >= 60:
         lifecycle_status = "Near End of Life"
       else:
         lifecycle_status = "End of Life"
       print(
    f"Battery Health Score: {health_score}/100"
    )
    print(
    f"Battery Grade: {battery_grade}"
    )
    print(
    f"Lifecycle Status: {lifecycle_status}"
    )

    latest_health = (
        soh_results["health_category"].iloc[-1]
       )

    print(
        f"Health Category: {latest_health}"
       )

    capacity_loss = (
        100 - latest_soh
       )

    print(
        f"Capacity Loss: {capacity_loss:.2f}%"
       )

    initial_capacity = (
        soh_results["capacity_ah"].iloc[0]
       )

    print(
        f"Initial Capacity: {initial_capacity:.3f} Ah"
       )

    final_capacity = (
        soh_results["capacity_ah"].iloc[-1]
       )

    print(
        f"Final Capacity: {final_capacity:.3f} Ah"
       )

    capacity_fade = (
        initial_capacity - final_capacity
      )

    print(
        f"Capacity Fade: {capacity_fade:.3f} Ah"
      )

    if latest_soh >= 90:
         recommendation = "Battery is in excellent condition."
    elif latest_soh >= 80:
         recommendation = "Battery is healthy and suitable for continued use."
    elif latest_soh >= 70:
         recommendation = "Battery is degraded and should be monitored."
    else:
         recommendation = (
            "Battery is approaching end-of-life "
            "and replacement should be evaluated."
        )

    print(
        f"Recommendation: {recommendation}"
       )
    if should_run_degradation(
        capability_report
    ):
        run_degradation_analysis(
            normalized_data,
        )

    if should_run_ica(
        capability_report
    ):
        run_ica_analysis()

    if should_run_dva(
        capability_report
    ):
        run_dva_analysis()

    print("=" * 60)