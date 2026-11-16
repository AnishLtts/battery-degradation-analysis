
#from analytics.degradation.degradation_analyzer import (
 #   create_cycle_capacity_table,
  #  calculate_degradation_metrics,
#)
import sys
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from analytics.ica_dva.ica_analyzer import (
    analyze_ica,
    save_ica_results,
)


from analytics.soh_engine import (
    calculate_capacity_soh,
)
from analytics.degradation.degradation_analyzer import (
   create_cycle_capacity_table,
   calculate_degradation_metrics,
)
    
def run_soh_analysis(
    dataframe,
):
    print("🚀 Starting SOH Analysis")

    soh_results = calculate_capacity_soh(
        dataframe
    )

    print(
        f"✅ SOH records generated: "
        f"{len(soh_results)}"
    )
    print(soh_results.head())
    print(soh_results.columns)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))

    plt.plot(
    soh_results["cycle"],
    soh_results["soh_percent"],
    linewidth=2,
    )

    plt.title(
    "State of Health vs Cycle"
    )
    plt.xlabel(
    "Cycle"
    )
    plt.ylabel(
    "SOH (%)"
    )

    plt.grid(True)
    plt.savefig(
    "battery_degradation_project/battery_intelligence_platform/outputs/graphs/soh_vs_cycle.png"
    )

    plt.show()

    # Capacity vs Cycle Graph
    plt.figure(figsize=(10, 5))

    plt.plot(
    soh_results["cycle"],
    soh_results["capacity_ah"],
    linewidth=2,
    )

    plt.title(
    "Capacity vs Cycle"
    )
    plt.xlabel(
    "Cycle"
    )
    plt.ylabel(
    "Capacity (Ah)"
    )

    plt.grid(True)
    plt.savefig(
    "battery_degradation_project/battery_intelligence_platform/outputs/graphs/capacity_vs_cycle.png"
)
    plt.show()
    return soh_results 
def run_degradation_analysis(
    dataframe,
):
    print("🚀 Starting Degradation Analysis")

    cycle_capacity = (
        create_cycle_capacity_table(
            dataframe
        )
    )

    degradation_results, degradation_summary = (
        calculate_degradation_metrics(
            cycle_capacity
        )
    )

    print(
        f"✅ Degradation records generated: "
        f"{len(degradation_results)}"
    )

    return degradation_results


def run_ica_analysis(
    dataframe,
):
    print(
        "🚀 Starting Incremental Capacity Analysis"
    )

    (
        ica_results,
        ica_cycle_summary,
        ica_summary,
        discharge_info,
    ) = analyze_ica(
        dataframe
    )

    output_folder = (
        "battery_degradation_project/"
        "battery_intelligence_platform/"
        "outputs/ica_analysis"
    )

    save_ica_results(
        ica_results,
        ica_cycle_summary,
        ica_summary,
        output_folder,
    )

    print(
        f"✅ ICA cycles analyzed: "
        f"{ica_summary['cycles_analyzed']}"
    )


def run_dva_analysis():
    print(
        "✅ Differential Voltage Analysis executed"
    )
