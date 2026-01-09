"""
CLI tool to read a sensor data CSV file and generate a plot of the recorded values.
Visualizes CO2, Temperature, Humidity, Gas Resistance, and Door Status.

Usage:
    uv run tools/plot_data.py <path_to_csv>
"""

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot E-Nose Sensor Data from CSV")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file to plot")
    return parser.parse_args()


def read_data(csv_file):
    """
    Reads the CSV file and returns a dictionary of data arrays.
    """
    data = {
        "datetime": [],
        "scd_co2": [],
        "scd_temp": [],
        "scd_hum": [],
        "bme_temp": [],
        "bme_hum": [],
        "bme_pres": [],
        "bme_gas": [],
        "door_open": [],
    }

    logger.info("Reading file: %s", csv_file)
    csv_path = Path(csv_file)

    def parse_float(row, key):
        val = row.get(key, "")
        if val and val.strip():
            try:
                return float(val)
            except ValueError:
                return np.nan
        return np.nan

    def parse_int(row, key):
        val = row.get(key, "")
        if val and val.strip():
            try:
                return int(float(val))  # Handle 0.0 or 1.0
            except ValueError:
                return 0
        return 0

    try:
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse datetime
                try:
                    dt = datetime.fromisoformat(row["datetime"])
                except (ValueError, KeyError):
                    continue  # Skip invalid rows

                data["datetime"].append(dt)

                data["scd_co2"].append(parse_float(row, "scd_co2"))
                data["scd_temp"].append(parse_float(row, "scd_temp"))
                data["scd_hum"].append(parse_float(row, "scd_hum"))
                data["bme_temp"].append(parse_float(row, "bme_temp"))
                data["bme_hum"].append(parse_float(row, "bme_hum"))
                data["bme_pres"].append(parse_float(row, "bme_pres"))
                data["bme_gas"].append(parse_float(row, "bme_gas"))
                data["door_open"].append(parse_int(row, "door_open"))

    except FileNotFoundError:
        logger.error("File not found: %s", csv_file)
        sys.exit(1)
    except Exception as e:
        logger.error("Error reading CSV: %s", e)
        sys.exit(1)

    return data


def plot_data(data, filename):
    """
    Generates a matplotlib plot from the data.
    """
    timestamps = data["datetime"]
    if not timestamps:
        logger.warning("No data found to plot.")
        return

    # Convert lists to numpy arrays for easier plotting (handling NaNs)
    scd_co2 = np.array(data["scd_co2"])
    scd_temp = np.array(data["scd_temp"])
    bme_temp = np.array(data["bme_temp"])
    scd_hum = np.array(data["scd_hum"])
    bme_hum = np.array(data["bme_hum"])
    bme_pres = np.array(data["bme_pres"])
    bme_gas = np.array(data["bme_gas"])
    door_open = np.array(data["door_open"])

    fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"Sensor Data Analysis: {filename}", fontsize=16)

    # Find door open regions for highlighting
    is_door_open = door_open > 0

    def highlight_door_regions(ax, timestamps, is_open):
        """Add red shading for door open periods."""
        in_region = False
        start_idx = 0
        for i, open_state in enumerate(is_open):
            if open_state and not in_region:
                start_idx = i
                in_region = True
            elif not open_state and in_region:
                ax.axvspan(
                    timestamps[start_idx], timestamps[i - 1], color="red", alpha=0.1
                )
                in_region = False
        # Handle case where door is still open at end
        if in_region:
            ax.axvspan(timestamps[start_idx], timestamps[-1], color="red", alpha=0.1)

    # 1. CO2
    axs[0].plot(timestamps, scd_co2, label="SCD30 CO2 (ppm)", color="tab:green")
    axs[0].set_ylabel("CO2 (ppm)")
    axs[0].legend(loc="upper left")
    axs[0].grid(True, alpha=0.3)
    highlight_door_regions(axs[0], timestamps, is_door_open)

    # 2. Temperature
    axs[1].plot(
        timestamps, scd_temp, label="SCD30 Temp (°C)", color="tab:red", linestyle="--"
    )
    axs[1].plot(timestamps, bme_temp, label="BME688 Temp (°C)", color="tab:orange")
    axs[1].set_ylabel("Temp (°C)")
    axs[1].legend(loc="upper left")
    axs[1].grid(True, alpha=0.3)
    highlight_door_regions(axs[1], timestamps, is_door_open)

    # 3. Humidity
    axs[2].plot(
        timestamps, scd_hum, label="SCD30 Hum (%)", color="tab:blue", linestyle="--"
    )
    axs[2].plot(timestamps, bme_hum, label="BME688 Hum (%)", color="tab:cyan")
    axs[2].set_ylabel("Humidity (%)")
    axs[2].legend(loc="upper left")
    axs[2].grid(True, alpha=0.3)
    highlight_door_regions(axs[2], timestamps, is_door_open)

    # 4. Pressure
    axs[3].plot(timestamps, bme_pres, label="BME688 Pressure (hPa)", color="tab:brown")
    axs[3].set_ylabel("Pressure (hPa)")
    axs[3].legend(loc="upper left")
    axs[3].grid(True, alpha=0.3)
    highlight_door_regions(axs[3], timestamps, is_door_open)

    # 5. Gas Resistance
    axs[4].plot(timestamps, bme_gas, label="BME688 Gas (Ohm)", color="tab:purple")
    axs[4].set_ylabel("Gas Resistance (Ω)")
    axs[4].set_xlabel("Time")
    axs[4].legend(loc="upper left")
    axs[4].grid(True, alpha=0.3)
    highlight_door_regions(axs[4], timestamps, is_door_open)

    plt.tight_layout()

    # Save the plot
    output_file = filename.replace(".csv", ".png")
    plt.savefig(output_file)
    logger.info("Plot saved to: %s", output_file)

    logger.info("Attempting to display plot...")
    try:
        plt.show()
    except Exception as e:
        logger.warning("Could not display plot window: %s", e)
        logger.info("You can view the generated PNG file instead.")


def main():
    args = parse_arguments()
    data = read_data(args.csv_file)
    plot_data(data, args.csv_file)


if __name__ == "__main__":
    main()
