
#from analytics.degradation.degradation_analyzer import (
 #   create_cycle_capacity_table,
  #  calculate_degradation_metrics,
#)
import os
import sys
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def get_output_root():
    """
    Folder where this run writes its results. The GUI and the command
    line set BATTERY_OUTPUT_ROOT to outputs/<dataset name>, so each
    dataset keeps its own charts and reports. Falls back to outputs/.
    """
    configured = os.environ.get("BATTERY_OUTPUT_ROOT")
    return Path(configured) if configured else PROJECT_ROOT / "outputs"

from analytics.ica_dva.ica_analyzer import (
    analyze_ica,
    save_ica_results,
)
from analytics.ica_dva.dva_analyzer import (
    analyze_dva,
    save_dva_results,
)

from analytics.soh_engine import (
    calculate_capacity_soh,
)
from analytics.degradation.degradation_analyzer import (
   validate_degradation_dataset,
   create_cycle_capacity_table,
   calculate_degradation_metrics,
)
from analytics.soc.soc_estimator import (
    validate_soc_dataset,
    calculate_coulomb_counting,
    estimate_soc,
    save_soc_results,
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

    graphs_folder = (
        get_output_root() /"graphs"
    )

    graphs_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_of_range_count = (
        soh_results["soh_percent"] > 100
    ).sum()

    if out_of_range_count > 0:
        print(
            f"⚠️  {out_of_range_count} SOH record(s) exceed 100% "
            f"(that battery's reference capacity was derived from "
            f"its own noisiest readings, not a true rated capacity)."
        )

    plt.figure(figsize=(10, 5))

    if "battery_id" in soh_results.columns:
        for _, battery_data in soh_results.groupby(
            "battery_id"
        ):
            plt.plot(
                battery_data["cycle"],
                battery_data["soh_percent"],
                linewidth=1,
            )
    else:
        plt.plot(
            soh_results["cycle"],
            soh_results["soh_percent"],
            linewidth=2,
        )

    plt.axhline(
        100,
        color="black",
        linewidth=0.8,
        linestyle="--",
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
        graphs_folder / "soh_vs_cycle.png"
    )

    plt.show()

    # Capacity vs Cycle Graph
    plt.figure(figsize=(10, 5))

    if "battery_id" in soh_results.columns:
        for _, battery_data in soh_results.groupby(
            "battery_id"
        ):
            plt.plot(
                battery_data["cycle"],
                battery_data["capacity_ah"],
                linewidth=1,
            )
    else:
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
        graphs_folder / "capacity_vs_cycle.png"
    )
    plt.show()
    return soh_results
def run_degradation_analysis(
    dataframe,
):
    print("🚀 Starting Degradation Analysis")

    validated_data = validate_degradation_dataset(
        dataframe
    )

    cycle_capacity = (
        create_cycle_capacity_table(
            validated_data
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

    return degradation_results, degradation_summary


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
        get_output_root() /"ica_analysis"
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

    return ica_summary


def run_dva_analysis(
    dataframe,
):
    print(
        "🚀 Starting Differential Voltage Analysis"
    )

    (
        dva_results,
        dva_cycle_summary,
        dva_summary,
        discharge_info,
    ) = analyze_dva(
        dataframe
    )

    output_folder = (
        get_output_root() /"dva_analysis"
    )

    save_dva_results(
        dva_results,
        dva_cycle_summary,
        dva_summary,
        output_folder,
    )

    print(
        f"✅ DVA cycles analyzed: "
        f"{dva_summary['cycles_analyzed']}"
    )

    return dva_summary


def run_soc_analysis(
    dataframe,
):
    print(
        "🚀 Starting Coulomb-Counting SOC Estimation"
    )

    validated_data = validate_soc_dataset(
        dataframe
    )

    integrated_data = calculate_coulomb_counting(
        validated_data
    )

    soc_results, cycle_summary = estimate_soc(
        integrated_data
    )

    output_folder = (
        get_output_root() /"soc_analysis"
    )

    save_soc_results(
        soc_results,
        cycle_summary,
        output_folder,
    )

    print(
        f"✅ SOC cycles analyzed: "
        f"{cycle_summary['cycle'].nunique()}"
    )

    return cycle_summary

# ============================================================
# COULOMB-COUNTING CAPACITY ESTIMATE
# ============================================================

def run_coulomb_capacity_analysis(
    dataframe,
):
    """
    Estimate capacity per recording (or cycle) by coulomb counting
    against SOC change. Results are reported separately; they do not
    feed the SOH analysis, because a partial recording is not a full
    capacity measurement.
    """
    from analytics.capacity.coulomb_capacity import (
        estimate_capacity_from_soc,
    )

    print(
        "🚀 Starting Coulomb-Counting Capacity Estimate"
    )

    group_column = next(
        (
            column
            for column in ("source_file", "cycle")
            if column in dataframe.columns
        ),
        None,
    )

    estimates = estimate_capacity_from_soc(
        dataframe,
        group_column=group_column,
    )

    output_folder = (
        get_output_root() /"capacity_estimate"
    )
    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )
    estimates.to_csv(
        output_folder / "coulomb_capacity_estimates.csv",
        index=False,
    )

    accepted = estimates[
        estimates["status"] == "Accepted"
    ]

    summary = {
        "recordings_analyzed": int(len(estimates)),
        "accepted": int(len(accepted)),
        "rejected": int(len(estimates) - len(accepted)),
        "mean_capacity_ah": (
            float(accepted["capacity_ah"].mean())
            if len(accepted) else None
        ),
        "min_capacity_ah": (
            float(accepted["capacity_ah"].min())
            if len(accepted) else None
        ),
        "max_capacity_ah": (
            float(accepted["capacity_ah"].max())
            if len(accepted) else None
        ),
    }

    print(
        f"✅ Coulomb capacity: {summary['accepted']} accepted, "
        f"{summary['rejected']} rejected"
    )

    return summary
