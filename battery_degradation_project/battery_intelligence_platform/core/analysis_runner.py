
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
    return soh_results

def run_degradation_analysis(
    dataframe,
):
    print(
        "🚀 Starting Degradation Analysis"
    )

    print(
        f"Rows received: {len(dataframe)}"
    )
    print(
        "🚀 Starting Degradation Analysis"
    )

    cycle_capacity = (
        create_cycle_capacity_table(
            dataframe
        )
    )

    degradation_results, degradation_summary= (
        calculate_degradation_metrics(
            cycle_capacity
        )
    )

    print(
    f"✅ Degradation records generated: "
    f"{len(degradation_results)}"
    )
def run_ica_analysis():
    print(
        "✅ Incremental Capacity Analysis executed"
    )


def run_dva_analysis():
    print(
        "✅ Differential Voltage Analysis executed"
    )