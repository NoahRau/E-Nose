# main_logger.py
import time
import sys
import csv
import os
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

import config
from sensors import SensorManager
from door_detector import AdaptiveDoorDetector

def get_user_input():
    print("\n" + "="*45)
    print("   E-NOSE DATA RECORDER (Adafruit Driver)")
    print("="*45)
    while True:
        label = input(">> LABEL eingeben (z.B. Testlauf): ").strip()
        if label:
            label = label.replace(" ", "_").replace("/", "-")
            break
        print("[!] Label darf nicht leer sein.")

    duration_input = input("\n>> LAUFZEIT (Stunden) oder ENTER für endlos: ").strip()
    end_time = None
    if duration_input:
        try:
            end_time = datetime.now() + timedelta(hours=float(duration_input))
        except ValueError: pass
    return label, end_time

def main():
    experiment_label, auto_stop_time = get_user_input()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"data_{experiment_label}_{timestamp_str}.csv"

    # Header Definition
    csv_header = [
        "timestamp", "datetime", "label",
        "soft_door_open", "sigma_co2", "sigma_temp",
        "co2", "temp", "humidity", "gas_resistance"
    ]

    try:
        with open(csv_filename, mode='w', newline='') as f:
            csv.writer(f).writerow(csv_header)
        print(f"[OK] CSV-Aufzeichnung gestartet: {csv_filename}")
    except Exception as e:
        print(f"[ERROR] CSV-Erstellung fehlgeschlagen: {e}"); sys.exit(1)

    client = None
    try:
        client = InfluxDBClient(url=config.INFLUX_URL, token=config.INFLUX_TOKEN, org=config.INFLUX_ORG)
        write_api = client.write_api(write_options=SYNCHRONOUS)
        print("[OK] DB Verbindung erfolgreich.")
    except Exception as e:
        print(f"[WARN] InfluxDB nicht erreichbar: {e}")

    sensors = SensorManager()
    door_logic = AdaptiveDoorDetector(window_size=60, sensitivity=4.0)

    try:
        while True:
            loop_start = time.time()
            if auto_stop_time and datetime.now() > auto_stop_time: break

            # --- A) Messen mit der neuen get_formatted_data() ---
            readings = sensors.get_formatted_data()

            # Falls der SCD30 noch keine Daten hat (WARTE...), überspringen wir kurz
            if readings['scd_c'] is None:
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Warte auf SCD30...", end="")
                time.sleep(1)
                continue

            # Zuordnung der Werte aus der neuen sensors.py Struktur
            curr_co2 = readings['scd_c']
            curr_temp = readings['scd_t']
            curr_hum = readings['scd_h']
            gas_res = readings['bme_g']

            # --- B) Adaptive Logic ---
            is_door, sigma_co2, sigma_temp = door_logic.update(curr_co2, curr_temp)

            # --- C) CSV Speichern ---
            with open(csv_filename, mode='a', newline='') as f:
                csv.writer(f).writerow([
                    loop_start, datetime.now().isoformat(), experiment_label,
                    is_door, round(sigma_co2, 2), round(sigma_temp, 2),
                    curr_co2, curr_temp, curr_hum, gas_res
                ])

            # --- D) InfluxDB Speichern ---
            if client:
                try:
                    p = Point("sensor_metrics").tag("experiment", experiment_label)
                    p.field("co2", float(curr_co2)).field("temp", float(curr_temp))
                    p.field("humidity", float(curr_hum)).field("gas_res", float(gas_res or 0))
                    p.field("door_open", int(is_door))
                    write_api.write(bucket=config.INFLUX_BUCKET, org=config.INFLUX_ORG, record=p)
                except: pass

            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] CO2: {int(curr_co2)}ppm | Temp: {curr_temp}°C | Tür: {is_door}", end="")

            elapsed = time.time() - loop_start
            time.sleep(max(0, config.SAMPLING_RATE - elapsed))

    except KeyboardInterrupt:
        print(f"\n[!] Beendet. Daten in {csv_filename}")
        if client: client.close()

if __name__ == "__main__":
    main()