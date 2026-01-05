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

import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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
        "bme_gas": [],
        "door_open": [],
    }

    logger.info(f"Reading file: {csv_file}")
    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse datetime
                try:
                    dt = datetime.fromisoformat(row["datetime"])
                except (ValueError, KeyError):
                    continue  # Skip invalid rows

                data["datetime"].append(dt)

                # Helper to safely parse float
                def parse_float(key):
                    val = row.get(key, "")
                    if val and val.strip():
                        try:
                            return float(val)
                        except ValueError:
                            return np.nan
                    return np.nan

                # Helper to safely parse int (for door_open)
                def parse_int(key):
                    val = row.get(key, "")
                    if val and val.strip():
                        try:
                            return int(float(val)) # Handle 0.0 or 1.0
                        except ValueError:
                            return 0
                    return 0

                data["scd_co2"].append(parse_float("scd_co2"))
                data["scd_temp"].append(parse_float("scd_temp"))
                data["scd_hum"].append(parse_float("scd_hum"))
                data["bme_temp"].append(parse_float("bme_temp"))
                data["bme_hum"].append(parse_float("bme_hum"))
                data["bme_gas"].append(parse_float("bme_gas"))
                data["door_open"].append(parse_int("door_open"))

    except FileNotFoundError:
        logger.error(f"File not found: {csv_file}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
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
    bme_gas = np.array(data["bme_gas"])
    door_open = np.array(data["door_open"])

    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Sensor Data Analysis: {filename}", fontsize=16)

    # 1. CO2
    axs[0].plot(timestamps, scd_co2, label="SCD30 CO2 (ppm)", color="tab:green")
    axs[0].set_ylabel("CO2 (ppm)")
    axs[0].legend(loc="upper left")
    axs[0].grid(True, alpha=0.3)
    
    # Highlight Door Open regions
    # Create boolean mask
    is_door_open = door_open > 0
    axs[0].fill_between(timestamps, axs[0].get_ylim()[0], axs[0].get_ylim()[1], 
                        where=is_door_open, color='red', alpha=0.1, label="Door Open", transform=axs[0].get_xaxis_transform())


    # 2. Temperature
    axs[1].plot(timestamps, scd_temp, label="SCD30 Temp (°C)", color="tab:red", linestyle="--")
    axs[1].plot(timestamps, bme_temp, label="BME688 Temp (°C)", color="tab:orange")
    axs[1].set_ylabel("Temp (°C)")
    axs[1].legend(loc="upper left")
    axs[1].grid(True, alpha=0.3)
    axs[1].fill_between(timestamps, axs[1].get_ylim()[0], axs[1].get_ylim()[1], 
                        where=is_door_open, color='red', alpha=0.1, transform=axs[1].get_xaxis_transform())

    # 3. Humidity
    axs[2].plot(timestamps, scd_hum, label="SCD30 Hum (%)", color="tab:blue", linestyle="--")
    axs[2].plot(timestamps, bme_hum, label="BME688 Hum (%)", color="tab:cyan")
    axs[2].set_ylabel("Humidity (%)")
    axs[2].legend(loc="upper left")
    axs[2].grid(True, alpha=0.3)
    axs[2].fill_between(timestamps, axs[2].get_ylim()[0], axs[2].get_ylim()[1], 
                        where=is_door_open, color='red', alpha=0.1, transform=axs[2].get_xaxis_transform())

    # 4. Gas Resistance
    axs[3].plot(timestamps, bme_gas, label="BME688 Gas (Ohm)", color="tab:purple")
    axs[3].set_ylabel("Gas Resistance (Ω)")
    axs[3].set_xlabel("Time")
    axs[3].legend(loc="upper left")
    axs[3].grid(True, alpha=0.3)
    axs[3].fill_between(timestamps, axs[3].get_ylim()[0], axs[3].get_ylim()[1], 
                        where=is_door_open, color='red', alpha=0.1, transform=axs[3].get_xaxis_transform())

    plt.tight_layout()
    logger.info("Displaying plot...")
    plt.show()


def main():
    args = parse_arguments()
    data = read_data(args.csv_file)
    plot_data(data, args.csv_file)


if __name__ == "__main__":
    main()
