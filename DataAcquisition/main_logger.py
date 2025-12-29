# main_logger.py
import time
import sys
import csv
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# --- Eigene Module ---
import config
from sensors import SensorManager
from door_detector import AdaptiveDoorDetector


def get_user_input():
    print("\n" + "=" * 45)
    print("   E-NOSE DATA RECORDER (Full Parallel Logging)")
    print("=" * 45)

    while True:
        label = input(">> LABEL eingeben (z.B. Testlauf_Beide_Sensoren): ").strip()
        if label:
            label = label.replace(" ", "_").replace("/", "-")
            break
        print("[!] Label darf nicht leer sein.")

    duration_input = input("\n>> LAUFZEIT (Stunden) oder ENTER für endlos: ").strip()

    end_time = None
    if duration_input:
        try:
            hours = float(duration_input)
            end_time = datetime.now() + timedelta(hours=hours)
            print(f"[*] Aufnahme stoppt automatisch am: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        except ValueError:
            print("[!] Ungültige Eingabe. Starte Endlos-Modus.")
    else:
        print("[*] Endlos-Modus aktiviert.")

    return label, end_time


def _fmt(v, ndigits=2):
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


def _print_debug_table(ts, scd_c, scd_t, scd_h, bme_t, bme_h, bme_p, bme_g, door_open, sigma_co2, sigma_temp):
    line1 = f"[{ts}] door={door_open} | sigma_co2={_fmt(sigma_co2)} sigma_temp={_fmt(sigma_temp)}"
    line2 = f"SCD | CO2={_fmt(scd_c,0)} ppm | T={_fmt(scd_t)} °C | H={_fmt(scd_h)} %"
    line3 = f"BME | T={_fmt(bme_t)} °C | H={_fmt(bme_h)} % | P={_fmt(bme_p)} hPa | G={_fmt(bme_g,0)} Ω"
    msg = f"{line1} || {line2} || {line3}"
    print("\r" + msg.ljust(180), end="")


def main():
    experiment_label, auto_stop_time = get_user_input()

    print("\n" + "-" * 40)
    print("   SYSTEM START")
    print("-" * 40)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"data_{experiment_label}_{timestamp_str}.csv"

    csv_header = [
        "timestamp", "datetime", "label",
        "door_open", "sigma_co2", "sigma_temp",
        "scd_co2", "scd_temp", "scd_hum",
        "bme_temp", "bme_hum", "bme_pres", "bme_gas"
    ]

    try:
        with open(csv_filename, mode="w", newline="") as f:
            csv.writer(f).writerow(csv_header)
        print(f"[OK] CSV-Aufzeichnung gestartet: {csv_filename}")
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"[ERROR] Konnte CSV Datei nicht erstellen: {e}")
        sys.exit(1)

    # Influx
    client = None
    write_api = None
    try:
        client = InfluxDBClient(
            url=config.INFLUX_URL,
            token=config.INFLUX_TOKEN,
            org=config.INFLUX_ORG
        )
        write_api = client.write_api(write_options=SYNCHRONOUS)
        print("[OK] DB Verbindung erfolgreich.")
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"[WARN] InfluxDB nicht erreichbar: {e}")
        client = None
        write_api = None

    sensors = SensorManager()
    door_logic = AdaptiveDoorDetector(window_size=60, sensitivity=4.0)
    print("[*] Sensoren und Logik bereit.")

    try:
        with open(csv_filename, mode="a", newline="") as f:
            writer = csv.writer(f)

            while True:
                loop_start = time.time()

                # Auto-stop
                if auto_stop_time and datetime.now() > auto_stop_time:
                    print("\n[FINISH] Automatische Laufzeit beendet.")
                    break

                # A) read
                try:
                    readings = sensors.get_formatted_data() or {}
                except KeyboardInterrupt:
                    raise
                except Exception:
                    readings = {}

                scd_c = readings.get("scd_c")
                scd_t = readings.get("scd_t")
                scd_h = readings.get("scd_h")

                bme_t = readings.get("bme_t")
                bme_h = readings.get("bme_h")
                bme_p = readings.get("bme_p")
                bme_g = readings.get("bme_g")

                # B) logic
                door_open, sigma_co2, sigma_temp = 0, 0.0, 0.0
                try:
                    if scd_c is not None:
                        temp_for_logic = bme_t if bme_t is not None else scd_t
                        if temp_for_logic is not None:
                            door_open, sigma_co2, sigma_temp = door_logic.update(scd_c, temp_for_logic)
                except KeyboardInterrupt:
                    raise
                except Exception:
                    door_open, sigma_co2, sigma_temp = 0, 0.0, 0.0

                # normalize door_open (0/1)
                try:
                    door_open = 1 if int(door_open) == 1 else 0
                except KeyboardInterrupt:
                    raise
                except Exception:
                    door_open = 0

                # C) CSV
                now_iso = datetime.now().isoformat()
                try:
                    writer.writerow([
                        loop_start,
                        now_iso,
                        experiment_label,
                        door_open,
                        round(float(sigma_co2), 2) if sigma_co2 is not None else 0.0,
                        round(float(sigma_temp), 2) if sigma_temp is not None else 0.0,
                        scd_c, scd_t, scd_h,
                        bme_t, bme_h, bme_p, bme_g
                    ])
                except KeyboardInterrupt:
                    raise
                except Exception:
                    pass

                # D) Influx
                if client and write_api:
                    try:
                        point = Point("sensor_metrics").tag("experiment", experiment_label)

                        if scd_c is not None: point.field("scd_co2", float(scd_c))
                        if scd_t is not None: point.field("scd_temp", float(scd_t))
                        if scd_h is not None: point.field("scd_hum", float(scd_h))

                        if bme_t is not None: point.field("bme_temp", float(bme_t))
                        if bme_h is not None: point.field("bme_hum", float(bme_h))
                        if bme_p is not None: point.field("bme_pres", float(bme_p))  # pressure in influx
                        if bme_g is not None: point.field("gas_res", float(bme_g))

                        point.field("door_open", int(door_open))
                        point.field("sigma_co2", float(sigma_co2) if sigma_co2 is not None else 0.0)
                        point.field("sigma_temp", float(sigma_temp) if sigma_temp is not None else 0.0)

                        write_api.write(bucket=config.INFLUX_BUCKET, org=config.INFLUX_ORG, record=point)
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        pass

                # E) Debug
                _print_debug_table(
                    datetime.now().strftime("%H:%M:%S"),
                    scd_c, scd_t, scd_h,
                    bme_t, bme_h, bme_p, bme_g,
                    door_open, sigma_co2, sigma_temp
                )

                # pacing
                elapsed = time.time() - loop_start
                try:
                    time.sleep(max(0, config.SAMPLING_RATE - elapsed))
                except KeyboardInterrupt:
                    raise
                except Exception:
                    pass

    except KeyboardInterrupt:
        print(f"\n\n[!] Aufnahme beendet. Daten gespeichert in: {csv_filename}")
    finally:
        try:
            if client:
                client.close()
        except KeyboardInterrupt:
            raise
        except Exception:
            pass


if __name__ == "__main__":
    main()
