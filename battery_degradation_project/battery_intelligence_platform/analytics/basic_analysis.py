from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


# Allow this file to import modules from the core folder
CURRENT_FILE = Path(__file__).resolve()
PLATFORM_FOLDER = CURRENT_FILE.parent.parent
CORE_FOLDER = PLATFORM_FOLDER / "core"

sys.path.insert(0, str(CORE_FOLDER))

from file_reader import load_battery_dataset
from column_mapper import (
    apply_column_mapping,
    detect_column_mapping,
)


def prepare_plot_data(dataframe, maximum_points=10000):
    """
    Reduce very large datasets for faster graph display.

    The original dataset is not changed.
    """

    if len(dataframe) <= maximum_points:
        return dataframe.copy()

    sampling_step = max(
        1,
        len(dataframe) // maximum_points,
    )

    sampled_data = dataframe.iloc[
        ::sampling_step
    ].copy()

    return sampled_data


def create_time_series_graphs(dataframe):
    """
    Create graphs for time-series battery datasets.
    """

    figures = []

    if "time_s" not in dataframe.columns:
        return figures

    plot_data = prepare_plot_data(dataframe)

    time_series_signals = {
        "voltage_v": {
            "title": "Battery Voltage Over Time",
            "ylabel": "Voltage (V)",
            "color": "blue",
        },
        "current_a": {
            "title": "Battery Current Over Time",
            "ylabel": "Current (A)",
            "color": "orange",
        },
        "temperature_c": {
            "title": "Battery Temperature Over Time",
            "ylabel": "Temperature (°C)",
            "color": "red",
        },
        "soc_percent": {
            "title": "State of Charge Over Time",
            "ylabel": "SOC (%)",
            "color": "green",
        },
    }

    for column, settings in time_series_signals.items():
        if column not in plot_data.columns:
            continue

        figure, axis = plt.subplots(
            figsize=(11, 5)
        )

        axis.plot(
            plot_data["time_s"],
            plot_data[column],
            color=settings["color"],
            linewidth=1,
        )

        axis.set_title(settings["title"])
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(settings["ylabel"])
        axis.grid(True, alpha=0.3)

        figure.tight_layout()

        figures.append(
            {
                "name": column,
                "figure": figure,
            }
        )

    return figures


def create_power_graph(dataframe):
    """
    Calculate and plot battery power over time.
    """

    required_columns = {
        "time_s",
        "voltage_v",
        "current_a",
    }

    if not required_columns.issubset(
        dataframe.columns
    ):
        return None

    plot_data = prepare_plot_data(dataframe)

    power_w = (
        plot_data["voltage_v"]
        * plot_data["current_a"]
    )

    figure, axis = plt.subplots(
        figsize=(11, 5)
    )

    axis.plot(
        plot_data["time_s"],
        power_w,
        color="purple",
        linewidth=1,
    )

    axis.axhline(
        0,
        color="black",
        linewidth=0.8,
    )

    axis.set_title("Battery Power Over Time")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Power (W)")
    axis.grid(True, alpha=0.3)

    figure.tight_layout()

    return {
        "name": "power_w",
        "figure": figure,
    }


def create_capacity_graphs(dataframe):
    """
    Create capacity and degradation graphs.
    """

    figures = []

    if "capacity_ah" not in dataframe.columns:
        return figures

    valid_data = dataframe.dropna(
        subset=["capacity_ah"]
    ).copy()

    if valid_data.empty:
        return figures

    figure, axis = plt.subplots(
        figsize=(11, 5)
    )

    if "test_id" in valid_data.columns:
        x_values = valid_data["test_id"]
        x_label = "Test ID"

    elif "cycle" in valid_data.columns:
        x_values = valid_data["cycle"]
        x_label = "Cycle"

    else:
        x_values = range(len(valid_data))
        x_label = "Record Order"

    axis.plot(
        x_values,
        valid_data["capacity_ah"],
        color="green",
        linewidth=1.2,
    )

    axis.set_title("Battery Capacity Trend")
    axis.set_xlabel(x_label)
    axis.set_ylabel("Capacity (Ah)")
    axis.grid(True, alpha=0.3)

    figure.tight_layout()

    figures.append(
        {
            "name": "capacity_trend",
            "figure": figure,
        }
    )

    if "battery_id" in valid_data.columns:
        grouped_capacity = (
            valid_data.groupby("battery_id")[
                "capacity_ah"
            ]
            .mean()
            .sort_values(ascending=False)
        )

        figure, axis = plt.subplots(
            figsize=(11, 5)
        )

        grouped_capacity.plot(
            kind="bar",
            ax=axis,
            color="teal",
        )

        axis.set_title(
            "Average Capacity by Battery"
        )
        axis.set_xlabel("Battery ID")
        axis.set_ylabel("Average Capacity (Ah)")
        axis.grid(
            True,
            axis="y",
            alpha=0.3,
        )

        figure.tight_layout()

        figures.append(
            {
                "name": "capacity_by_battery",
                "figure": figure,
            }
        )

    return figures


def create_resistance_graphs(dataframe):
    """
    Create electrolyte and charge-transfer
    resistance graphs.
    """

    figures = []

    resistance_columns = {
        "electrolyte_resistance_ohm": (
            "Electrolyte Resistance Trend",
            "Electrolyte Resistance (Ω)",
            "blue",
        ),
        "charge_transfer_resistance_ohm": (
            "Charge-Transfer Resistance Trend",
            "Charge-Transfer Resistance (Ω)",
            "red",
        ),
        "internal_resistance_ohm": (
            "Internal Resistance Trend",
            "Internal Resistance (Ω)",
            "purple",
        ),
    }

    for column, settings in resistance_columns.items():
        if column not in dataframe.columns:
            continue

        valid_data = dataframe.dropna(
            subset=[column]
        ).copy()

        if valid_data.empty:
            continue

        if "test_id" in valid_data.columns:
            x_values = valid_data["test_id"]
            x_label = "Test ID"

        elif "cycle" in valid_data.columns:
            x_values = valid_data["cycle"]
            x_label = "Cycle"

        else:
            x_values = range(len(valid_data))
            x_label = "Record Order"

        figure, axis = plt.subplots(
            figsize=(11, 5)
        )

        axis.plot(
            x_values,
            valid_data[column],
            color=settings[2],
            linewidth=1.2,
        )

        axis.set_title(settings[0])
        axis.set_xlabel(x_label)
        axis.set_ylabel(settings[1])
        axis.grid(True, alpha=0.3)

        figure.tight_layout()

        figures.append(
            {
                "name": column,
                "figure": figure,
            }
        )

    return figures


def create_temperature_comparison(dataframe):
    """
    Compare available battery measurements
    across ambient temperatures.
    """

    if "ambient_temperature_c" not in dataframe.columns:
        return None

    measurement_column = None
    ylabel = None

    if "capacity_ah" in dataframe.columns:
        measurement_column = "capacity_ah"
        ylabel = "Average Capacity (Ah)"

    elif "soc_percent" in dataframe.columns:
        measurement_column = "soc_percent"
        ylabel = "Average SOC (%)"

    elif "voltage_v" in dataframe.columns:
        measurement_column = "voltage_v"
        ylabel = "Average Voltage (V)"

    if measurement_column is None:
        return None

    valid_data = dataframe.dropna(
        subset=[
            "ambient_temperature_c",
            measurement_column,
        ]
    )

    if valid_data.empty:
        return None

    temperature_summary = (
        valid_data.groupby(
            "ambient_temperature_c"
        )[measurement_column]
        .mean()
        .sort_index()
    )

    figure, axis = plt.subplots(
        figsize=(11, 5)
    )

    temperature_summary.plot(
        kind="bar",
        ax=axis,
        color="darkorange",
    )

    axis.set_title(
        "Battery Measurement by Ambient Temperature"
    )
    axis.set_xlabel("Ambient Temperature (°C)")
    axis.set_ylabel(ylabel)
    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()

    return {
        "name": "ambient_temperature_comparison",
        "figure": figure,
    }


def generate_automatic_graphs(dataframe):
    """
    Generate all valid graphs for the available columns.
    """

    figures = []

    figures.extend(
        create_time_series_graphs(dataframe)
    )

    power_figure = create_power_graph(dataframe)

    if power_figure is not None:
        figures.append(power_figure)

    figures.extend(
        create_capacity_graphs(dataframe)
    )

    figures.extend(
        create_resistance_graphs(dataframe)
    )

    temperature_figure = (
        create_temperature_comparison(dataframe)
    )

    if temperature_figure is not None:
        figures.append(temperature_figure)

    return figures


def save_graphs(
    figures,
    output_directory,
):
    """
    Save generated graphs as PNG files.
    """

    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_files = []

    for graph in figures:
        output_path = (
            output_directory
            / f"{graph['name']}.png"
        )

        graph["figure"].savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

        saved_files.append(output_path)

        plt.close(graph["figure"])

    return saved_files


if __name__ == "__main__":
    dataset_path = input(
        "Paste the full path of a battery dataset: "
    ).strip().strip('"')

    try:
        battery_data = load_battery_dataset(
            dataset_path
        )

        mapping, unknown_columns = (
            detect_column_mapping(
                battery_data
            )
        )

        standardized_data = apply_column_mapping(
            battery_data,
            mapping,
        )

        generated_figures = (
            generate_automatic_graphs(
                standardized_data
            )
        )

        if not generated_figures:
            print(
                "\nNo supported graphs could be generated "
                "from the available columns."
            )

        else:
            output_folder = (
                PLATFORM_FOLDER
                / "outputs"
                / Path(dataset_path).stem
            )

            saved_files = save_graphs(
                generated_figures,
                output_folder,
            )

            print("\n" + "=" * 70)
            print("AUTOMATIC BATTERY GRAPH GENERATION")
            print("=" * 70)

            print(
                f"Graphs generated: {len(saved_files)}"
            )

            for saved_file in saved_files:
                print(f"  - {saved_file}")

            print("=" * 70)

        if unknown_columns:
            print("\nUnrecognized columns:")

            for column in unknown_columns:
                print(f"  - {column}")

    except Exception as error:
        print(
            f"\nAutomatic graph generation failed: {error}"
        )