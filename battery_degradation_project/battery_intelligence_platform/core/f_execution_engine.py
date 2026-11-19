from pathlib import Path
from datetime import datetime

PLATFORM_FOLDER = (
    Path(__file__).resolve().parent.parent
)

from e_analysis_router import(
    should_run_soh,
    should_run_degradation,
    should_run_ica,
    should_run_dva,
    should_run_soc,

)
from g_analysis_runner import (
    run_soh_analysis,
    run_degradation_analysis,
    run_ica_analysis,
    run_dva_analysis,
    run_soc_analysis,
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
    soh_results = None
    #print(capability_report)

    health_score = "N/A"
    battery_grade = "N/A"
    lifecycle_status = "N/A"
    latest_soh = 0
    capacity_loss = 0
    initial_capacity = 0
    final_capacity = 0
    capacity_fade = 0
    recommendation = (
    "SOH analysis unavailable for this dataset."
)

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
        
    if soh_results is not None:

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
    if soh_results is not None:
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

    ica_summary = None
    ica_error = None

    if should_run_ica(
        capability_report
    ):
        try:
            ica_summary = run_ica_analysis(
               normalized_data,
            )
        except Exception as error:
            ica_error = str(error)
            print(f"⚠️  ICA analysis failed: {ica_error}")

    dva_summary = None
    dva_error = None

    if should_run_dva(
        capability_report
    ):
        try:
            dva_summary = run_dva_analysis(
               normalized_data,
            )
        except Exception as error:
            dva_error = str(error)
            print(f"⚠️  DVA analysis failed: {dva_error}")

    degradation_summary = None
    degradation_error = None

    if should_run_degradation(
        capability_report
    ):
        try:
            _, degradation_summary = run_degradation_analysis(
                normalized_data,
            )
        except Exception as error:
            degradation_error = str(error)
            print(f"⚠️  Degradation analysis failed: {degradation_error}")

    soc_cycle_summary = None
    soc_error = None

    if should_run_soc(
        capability_report
    ):
        try:
            soc_cycle_summary = run_soc_analysis(
               normalized_data,
            )
        except Exception as error:
            soc_error = str(error)
            print(f"⚠️  SOC estimation failed: {soc_error}")

    if degradation_summary is not None:
        degradation_section = (
            "Degradation Analysis: Completed\n"
            f"    Degradation Cycles Analyzed: {degradation_summary['cycles_analyzed']}\n"
            f"    Latest SOH: {degradation_summary['latest_soh_percent']:.2f}%\n"
            f"    Total Capacity Loss: {degradation_summary['total_capacity_loss_percent']:.2f}%\n"
            f"    Degradation Rate: {degradation_summary['degradation_rate_percent_per_cycle']:.4f}%/cycle"
        )
    elif degradation_error is not None:
        degradation_section = (
            f"Degradation Analysis: Failed - {degradation_error}"
        )
    else:
        degradation_section = (
            "Degradation Analysis: Not available for this dataset."
        )

    if soc_cycle_summary is not None:
        soc_section = (
            "SOC Estimation: Completed\n"
            f"    SOC Cycles Analyzed: {soc_cycle_summary['cycle'].nunique()}\n"
            "    SOC Graph:\n"
            "    outputs/soc_analysis/soc_analysis.png"
        )
    elif soc_error is not None:
        soc_section = (
            f"SOC Estimation: Failed - {soc_error}"
        )
    else:
        soc_section = (
            "SOC Estimation: Not available for this dataset."
        )

    if ica_summary is not None:
        ica_section = (
            "ICA Analysis: Completed\n"
            f"    ICA Cycles Analyzed: {ica_summary['cycles_analyzed']}\n"
            "    ICA Graph:\n"
            "    outputs/ica_analysis/ica_analysis.png"
        )
    elif ica_error is not None:
        ica_section = (
            f"ICA Analysis: Failed - {ica_error}"
        )
    else:
        ica_section = (
            "ICA Analysis: Not available for this dataset."
        )

    if dva_summary is not None:
        dva_section = (
            "DVA Analysis: Completed\n"
            f"    DVA Cycles Analyzed: {dva_summary['cycles_analyzed']}\n"
            "    DVA Graph:\n"
            "    outputs/dva_analysis/dva_analysis.png"
        )
    elif dva_error is not None:
        dva_section = (
            f"DVA Analysis: Failed - {dva_error}"
        )
    else:
        dva_section = (
            "DVA Analysis: Not available for this dataset."
        )

    run_timestamp = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)
    available_analysis_text = ""

    for analysis in capability_report[
    "available_analyses"
    ]:
     available_analysis_text += (
        f"[AVAILABLE] {analysis}\n"
     )
    unavailable_analysis_text = ""

    for analysis, details in capability_report[
    "unavailable_analyses"
    ].items():

     unavailable_analysis_text += (
        f"[UNAVAILABLE] {analysis}\n"
    )

     missing_items = []

     if details["missing_required"]:
        missing_items.extend(
            details["missing_required"]
        )

     if missing_items:
        unavailable_analysis_text += (
            f"   Reason: Missing "
            f"{', '.join(missing_items)}\n\n"
        )
    
    report_text = f"""

    DATASET INFORMATION

    Rows: {len(normalized_data):,}

   Cycles: {normalized_data['cycle'].nunique() if 'cycle' in normalized_data.columns else 'N/A'}

   ANALYSES COMPLETED

   {available_analysis_text}

   ANALYSES NOT AVAILABLE
   {unavailable_analysis_text}
   
    BATTERY INTELLIGENCE REPORT

    Battery Health Score: {health_score}/100
    Battery Grade: {battery_grade}
    Lifecycle Status: {lifecycle_status}

    Latest SOH: {latest_soh:.2f}%
    Capacity Loss: {capacity_loss:.2f}%

    Initial Capacity: {initial_capacity:.3f} Ah
    Final Capacity: {final_capacity:.3f} Ah
    Capacity Fade: {capacity_fade:.3f} Ah
    SOH Graph:
    outputs/graphs/soh_vs_cycle.png

    Capacity Graph:
    outputs/graphs/capacity_vs_cycle.png

    {ica_section}

    {dva_section}

    {degradation_section}

    {soc_section}

    Recommendation: {recommendation}
    """
    reports_folder = (
        PLATFORM_FOLDER / "outputs" / "reports"
    )

    reports_folder.mkdir(
    parents=True,
    exist_ok=True,
    )

    report_file = (
    reports_folder / "battery_report.txt"
    )

    with open(
    report_file,
    "w",
    ) as file:
     file.write(report_text)

    print(
    f"Report saved: {report_file}"
    )
    import pandas as pd

    summary_df = pd.DataFrame(
    {
        "Metric": [
            "Health Score",
            "Battery Grade",
            "Lifecycle Status",
            "Latest SOH",
            "Capacity Loss",
            "Initial Capacity",
            "Final Capacity",
            "Capacity Fade",
            "Recommendation",
        ],
        "Value": [
            health_score,
            battery_grade,
            lifecycle_status,
            latest_soh,
            capacity_loss,
            initial_capacity,
            final_capacity,
            capacity_fade,
            recommendation,
        ],
    }
)
    summary_df.to_csv(
    reports_folder / "battery_summary.csv",
    index=False,
)

    print(
    f"Summary saved: {reports_folder / 'battery_summary.csv'}"
)

    print(
        f"Recommendation: {recommendation}"
       )

    print("=" * 60)