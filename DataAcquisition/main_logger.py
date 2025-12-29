# main_logger.py
import time
import sys
import csv
import os
import numpy as np
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# --- Eigene Module ---
import config
from sensors import SensorManager
from door_detector import AdaptiveDoorDetector

def get_user_input():
    """Fragt den Nutzer interaktiv nach Experiment-Parametern"""
    print("\n" + "="*45)
    print("   E-NOSE ULTIMATE LOGGER (Multi-Gas & Resilience)")
    print("="*45)
    while True:
        label = input(">> EXPERIMENT-LABEL (z.B. Apfel_Tag1): ").strip()
        if label:
            return label.replace(" ", "_").replace("/", "-"), datetime.now()
        print("[!] Label darf nicht leer sein.")

def main():
    # --- Interaktiver Start ---
    experiment_label, start_time = get_user_input()
    csv_filename = f"data_{experiment_label}_{start_time.strftime('%Y%m%d_%H%M%S')}.csv"

    # 1. Header definieren
    csv_header = [
        "timestamp", "datetime", "label",
        "soft_door_open", "is_recovery",       # Status-Flags
        "sigma_co2", "sigma_temp",             # Anomalie-Metriken
        "scd_co2", "scd_temp", "scd_hum",      # SCD30 Rohdaten
        "bme_temp", "bme_hum", "bme_press"     # BME688 Rohdaten
    ]
    # Alle 10 Gaskanäle (gas_0 bis gas_9) hinzufügen
    csv_header.extend([f"gas_{i}" for i in range(10)])

    # Datei anlegen
    try:
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
        print(f"[OK] CSV-Aufzeichnung gestartet: {csv_filename}")
    except Exception as e:
        print(f"[ERROR] Konnte CSV Datei nicht erstellen: {e}")
        sys.exit(1)

    # 2. Verbindung zur InfluxDB
    client = None
    try:
        client = InfluxDBClient(
            url=config.INFLUX_URL,
            token=config.INFLUX_TOKEN,
            org=config.INFLUX_ORG
        )
        write_api = client.write_api(write_options=SYNCHRONOUS)
        print("[OK] InfluxDB Verbindung erfolgreich.")
    except Exception as e:
        print(f"[WARN] InfluxDB nicht erreichbar: {e}")

    # 3. Initialisierung der Hardware & Logik
    sensors = SensorManager()
    door_logic = AdaptiveDoorDetector(window_size=60, sensitivity=4.0)

    # Recovery-Logik: Sperrt KI-Training für X Sekunden nach Tür-Schluss
    recovery_timer = 0
    RECOVERY_DURATION = 300  # 5 Minuten Erholungszeit (in Sekunden)

    print("\n[*] Warte auf stabile Sensordaten...")

    try:
        while True:
            loop_start = time.time()
            readings = sensors.get_formatted_data()

            # --- Sicherheitscheck: Warte bis beide Sensoren bereit sind (verhindert NoneType Error) ---
            if readings['scd_c'] is None or readings['bme_t'] is None:
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Initialisiere Sensoren...", end="")
                time.sleep(1)
                continue

            # --- A) Tür & Recovery Status ---
            # Nutzt SCD CO2 und BME Temperatur für die Detektion
            is_door, sigma_co2, sigma_temp = door_logic.update(readings['scd_c'], readings['bme_t'])

            if is_door:
                recovery_timer = RECOVERY_DURATION
                is_recovery = 0
            elif recovery_timer > 0:
                # Exakte Zeit seit dem letzten Durchlauf abziehen
                recovery_timer -= (time.time() - loop_start + config.SAMPLING_RATE)
                is_recovery = 1
            else:
                recovery_timer = 0
                is_recovery = 0

            # --- B) Speichern (CSV) ---
            try:
                row = [
                    loop_start,
                    datetime.now().isoformat(),
                    experiment_label,
                    is_door,
                    is_recovery,
                    round(sigma_co2, 2),
                    round(sigma_temp, 2),
                    readings['scd_c'],
                    readings['scd_t'],
                    readings['scd_h'],
                    readings['bme_t'],
                    readings['bme_h'],
                    readings.get('bme_p', 0)
                ]
                # Alle 10 Gas-Werte aus dem Sensor-Buffer anhängen
                for i in range(10):
                    row.append(readings.get(f'gas_{i}', 0))

                with open(csv_filename, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
            except Exception as e:
                print(f"\n[ERROR] CSV Write: {e}")

            # --- C) Speichern (InfluxDB) ---
            if client:
                try:
                    point = Point("sensor_metrics").tag("experiment", experiment_label)
                    point.field("scd_co2", float(readings['scd_c']))
                    point.field("scd_temp", float(readings['scd_t']))
                    point.field("bme_temp", float(readings['bme_t']))
                    point.field("door_open", int(is_door))
                    point.field("is_recovery", int(is_recovery))
                    point.field("gas_res_0", float(readings.get('gas_0', 0)))
                    write_api.write(bucket=config.INFLUX_BUCKET, org=config.INFLUX_ORG, record=point)
                except:
                    pass

            # --- D) Feedback ---
            status = "OPEN " if is_door else ("RECOV" if is_recovery else "STABLE")
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] {status} | CO2: {int(readings['scd_c']):4}ppm | Gas_0: {int(readings.get('gas_0',0)):6}Ω", end="")

            # Taktung (Standard 2s) einhalten
            elapsed = time.time() - loop_start
            time.sleep(max(0, config.SAMPLING_RATE - elapsed))

    except KeyboardInterrupt:
        print(f"\n\n[!] Aufnahme beendet. Daten in: {csv_filename}")
        if client: client.close()

if __name__ == "__main__":
    main()