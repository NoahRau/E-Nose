# main_logger.py
import csv
import logging
import sys
import time
from datetime import datetime, timedelta

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from DataAcquisition import config
from DataAcquisition.door_detector import AdaptiveDoorDetector
from DataAcquisition.sensors import SensorManager

# Configure module logger
logger = logging.getLogger(__name__)


def setup_logging(log_file: str | None = None, level: int = logging.INFO) -> None:
    """Configure logging for the data acquisition module.

    Args:
        log_file: Optional path to log file. If None, logs only to console.
        level: Logging level (default: INFO).
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler with concise format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler with detailed format
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)


def get_user_input() -> tuple[str, datetime | None]:
    """Get experiment label and duration from user input."""
    print("\n" + "=" * 45)
    print("   E-NOSE DATA RECORDER (Full Parallel Logging)")
    print("=" * 45)

    while True:
        label = input(">> LABEL eingeben (z.B. Testlauf_Beide_Sensoren): ").strip()
        if label:
            label = label.replace(" ", "_").replace("/", "-")
            break
        logger.warning("Label darf nicht leer sein.")

    duration_input = input("\n>> LAUFZEIT (Stunden) oder ENTER für endlos: ").strip()

    end_time = None
    if duration_input:
        try:
            hours = float(duration_input)
            end_time = datetime.now() + timedelta(hours=hours)
            logger.info(
                "Aufnahme stoppt automatisch am: %s",
                end_time.strftime("%Y-%m-%d %H:%M:%S"),
            )
        except ValueError:
            logger.warning("Ungültige Eingabe. Starte Endlos-Modus.")
    else:
        logger.info("Endlos-Modus aktiviert.")

    return label, end_time


def _fmt(v, ndigits: int = 2) -> str:
    """Format a value for display."""
    if v is None:
        return "-"
    try:
        if isinstance(v, (int, float)):
            return str(round(float(v), ndigits))
        return str(v)
    except KeyboardInterrupt:
        raise
    except Exception:
        return "-"


def _format_sensor_line(
    ts: str,
    scd_c,
    scd_t,
    scd_h,
    bme_t,
    bme_h,
    bme_p,
    bme_g,
    door_open: int,
    sigma_co2: float,
    sigma_temp: float,
) -> str:
    """Format sensor readings into a single line for display."""
    line1 = f"[{ts}] door={door_open} | sigma_co2={_fmt(sigma_co2)} sigma_temp={_fmt(sigma_temp)}"
    line2 = f"SCD | CO2={_fmt(scd_c, 0)} ppm | T={_fmt(scd_t)} °C | H={_fmt(scd_h)} %"
    line3 = f"BME | T={_fmt(bme_t)} °C | H={_fmt(bme_h)} % | P={_fmt(bme_p)} hPa | G={_fmt(bme_g, 0)} Ω"
    return f"{line1} || {line2} || {line3}"


def main() -> None:
    """Main entry point for the E-Nose data logger."""
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get user input before setting up logging (so prompts aren't logged)
    experiment_label, auto_stop_time = get_user_input()

    # Setup logging with file output
    log_filename = f"log_{experiment_label}_{timestamp_str}.log"
    setup_logging(log_file=log_filename, level=logging.DEBUG)

    logger.info("-" * 40)
    logger.info("SYSTEM START")
    logger.info("-" * 40)

    csv_filename = f"data_{experiment_label}_{timestamp_str}.csv"

    csv_header = [
        "timestamp",
        "datetime",
        "label",
        "door_open",
        "sigma_co2",
        "sigma_temp",
        "scd_co2",
        "scd_temp",
        "scd_hum",
        "bme_temp",
        "bme_hum",
        "bme_pres",
        "bme_gas",
    ]

    try:
        with open(csv_filename, mode="w", newline="") as f:
            csv.writer(f).writerow(csv_header)
        logger.info("CSV-Aufzeichnung gestartet: %s", csv_filename)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.exception("Konnte CSV Datei nicht erstellen: %s", e)
        sys.exit(1)

    # InfluxDB connection
    client = None
    write_api = None
    try:
        client = InfluxDBClient(
            url=config.INFLUX_URL, token=config.INFLUX_TOKEN, org=config.INFLUX_ORG
        )
        write_api = client.write_api(write_options=SYNCHRONOUS)
        logger.info("InfluxDB Verbindung erfolgreich.")
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.warning("InfluxDB nicht erreichbar: %s", e)
        client = None
        write_api = None

    # Initialize sensors and door detection logic
    try:
        sensors = SensorManager()
        logger.info("SensorManager initialisiert.")
    except Exception as e:
        logger.exception("Fehler bei Sensor-Initialisierung: %s", e)
        sys.exit(1)

    door_logic = AdaptiveDoorDetector(window_size=60, sensitivity=4.0)
    logger.info("AdaptiveDoorDetector initialisiert (window=60, sensitivity=4.0)")
    logger.info("Sensoren und Logik bereit. Starte Datenaufnahme...")

    sample_count = 0

    try:
        with open(csv_filename, mode="a", newline="") as f:
            writer = csv.writer(f)

            while True:
                loop_start = time.time()

                # Auto-stop check
                if auto_stop_time and datetime.now() > auto_stop_time:
                    logger.info("Automatische Laufzeit beendet.")
                    break

                # A) Read sensor data
                try:
                    readings = sensors.get_formatted_data() or {}
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.debug("Sensor read error: %s", e)
                    readings = {}

                scd_c = readings.get("scd_c")
                scd_t = readings.get("scd_t")
                scd_h = readings.get("scd_h")

                bme_t = readings.get("bme_t")
                bme_h = readings.get("bme_h")
                bme_p = readings.get("bme_p")
                bme_g = readings.get("bme_g")

                # B) Door detection logic
                door_open, sigma_co2, sigma_temp = 0, 0.0, 0.0
                try:
                    if scd_c is not None:
                        temp_for_logic = bme_t if bme_t is not None else scd_t
                        if temp_for_logic is not None:
                            door_open, sigma_co2, sigma_temp = door_logic.update(
                                scd_c, temp_for_logic
                            )
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.debug("Door detection error: %s", e)
                    door_open, sigma_co2, sigma_temp = 0, 0.0, 0.0

                # Normalize door_open to 0/1
                try:
                    door_open = 1 if int(door_open) == 1 else 0
                except KeyboardInterrupt:
                    raise
                except Exception:
                    door_open = 0

                # Log door state changes
                if door_open == 1:
                    logger.warning(
                        "Door OPEN detected! sigma_co2=%.2f, sigma_temp=%.2f",
                        sigma_co2,
                        sigma_temp,
                    )

                # C) Write to CSV
                now_iso = datetime.now().isoformat()
                try:
                    writer.writerow(
                        [
                            loop_start,
                            now_iso,
                            experiment_label,
                            door_open,
                            round(float(sigma_co2), 2) if sigma_co2 is not None else 0.0,
                            round(float(sigma_temp), 2) if sigma_temp is not None else 0.0,
                            scd_c,
                            scd_t,
                            scd_h,
                            bme_t,
                            bme_h,
                            bme_p,
                            bme_g,
                        ]
                    )
                    f.flush()  # Ensure data is written
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error("CSV write error: %s", e)

                # D) Write to InfluxDB
                if client and write_api:
                    try:
                        point = Point("sensor_metrics").tag("experiment", experiment_label)

                        if scd_c is not None:
                            point.field("scd_co2", float(scd_c))
                        if scd_t is not None:
                            point.field("scd_temp", float(scd_t))
                        if scd_h is not None:
                            point.field("scd_hum", float(scd_h))

                        if bme_t is not None:
                            point.field("bme_temp", float(bme_t))
                        if bme_h is not None:
                            point.field("bme_hum", float(bme_h))
                        if bme_p is not None:
                            point.field("bme_pres", float(bme_p))
                        if bme_g is not None:
                            point.field("gas_res", float(bme_g))

                        point.field("door_open", int(door_open))
                        point.field(
                            "sigma_co2",
                            float(sigma_co2) if sigma_co2 is not None else 0.0,
                        )
                        point.field(
                            "sigma_temp",
                            float(sigma_temp) if sigma_temp is not None else 0.0,
                        )

                        write_api.write(
                            bucket=config.INFLUX_BUCKET,
                            org=config.INFLUX_ORG,
                            record=point,
                        )
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        logger.debug("InfluxDB write error: %s", e)

                # E) Live terminal output (overwriting line)
                sensor_line = _format_sensor_line(
                    datetime.now().strftime("%H:%M:%S"),
                    scd_c,
                    scd_t,
                    scd_h,
                    bme_t,
                    bme_h,
                    bme_p,
                    bme_g,
                    door_open,
                    sigma_co2,
                    sigma_temp,
                )
                print("\r" + sensor_line.ljust(180), end="", flush=True)

                sample_count += 1

                # Log periodic status every 100 samples
                if sample_count % 100 == 0:
                    logger.debug("Collected %d samples", sample_count)

                # Pacing to maintain sampling rate
                elapsed = time.time() - loop_start
                try:
                    time.sleep(max(0, config.SAMPLING_RATE - elapsed))
                except KeyboardInterrupt:
                    raise
                except Exception:
                    pass

    except KeyboardInterrupt:
        print()  # New line after the live output
        logger.info("Aufnahme durch Benutzer beendet.")
        logger.info("Gesammelte Samples: %d", sample_count)
        logger.info("Daten gespeichert in: %s", csv_filename)
        logger.info("Log gespeichert in: %s", log_filename)
    finally:
        try:
            if client:
                client.close()
                logger.debug("InfluxDB connection closed.")
        except KeyboardInterrupt:
            raise
        except Exception:
            pass


if __name__ == "__main__":
    main()
