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
    print("\n" + "="*45)
    print("   E-NOSE ULTIMATE LOGGER (Resilience Mode)")
    print("="*45)
    while True:
        label = input(">> EXPERIMENT-LABEL (z.B. Apfel_Tag1): ").strip()
        if label:
            return label.replace(" ", "_"), datetime.now()
        print("[!] Label erforderlich.")

def main():
    experiment_label, start_time = get_user_input()
    csv_filename = f"data_{experiment_label}_{start_time.strftime('%Y%m%d_%H%M%S')}.csv"

    # 1. Header mit allen Features (Physik, Chemie, Status)
    csv_header = [
        "timestamp", "datetime", "label",
        "soft_door_open", "is_recovery",       # Status-Flags
        "sigma_co2", "sigma_temp",             # Anomalie-Metriken
        "scd_co2", "scd_temp", "scd_hum",      # SCD30
        "bme_temp", "bme_hum", "bme_press"     # BME688 Physik
    ]
    # Alle 10 Gaskanäle (Spektrum) hinzufügen
    csv_header.extend([f"gas_{i}" for i in range(10)])

    try:
        with open(csv_filename, mode='w', newline='') as f:
            csv.writer(f).writerow(csv_header)
        print(f"[OK] CSV gestartet: {csv_filename}")
    except Exception as e:
        print(f"[ERROR] CSV: {e}"); sys.exit(1)

    # 2. InfluxDB Verbindung
    client = None
    try:
        client = InfluxDBClient(url=config.INFLUX_URL, token=config.INFLUX_TOKEN, org=config.INFLUX_ORG)
        write_api = client.write_api(write_options=SYNCHRONOUS)
        print("[OK] InfluxDB Verbindung aktiv.")
    except Exception as e:
        print(f"[WARN] InfluxDB nicht erreichbar: {e}")

    # 3. Hardware & Logik Initialisierung
    sensors = SensorManager()
    door_logic = AdaptiveDoorDetector(window_size=60, sensitivity=4.0)

    # Recovery-Logik
    recovery_timer = 0
    RECOVERY_DURATION = 300  # 5 Minuten Erholungszeit

    print("\n[*] Warte auf stabile Sensordaten...")

    try:
        while True:
            loop_start = time.time()
            readings = sensors.get_formatted_data()

            # --- FIX: Warte bis BEIDE Sensoren Daten liefern ---
            if readings['scd_c'] is None or readings['bme_t'] is None:
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Initialisiere Sensoren...", end="")
                time.sleep(1)
                continue

            # --- A) Tür & Recovery Status ---
            # Wir nutzen SCD30 CO2 und BME Temperatur (da der BME schneller auf Luftzüge reagiert)
            is_door, sigma_co2, sigma_temp = door_logic.update(readings['scd_c'], readings['bme_t'])

            if is_door:
                recovery_timer = RECOVERY_DURATION
                is_recovery = 0
            elif recovery_timer > 0:
                recovery_timer -= (time.time() - loop_start) # Exakte Zeit abziehen
                is_recovery = 1
            else:
                recovery_timer = 0
                is_recovery = 0

            # --- B) Speichern (CSV) ---
            row = [
                loop_start, datetime.now().isoformat(), experiment_label,
                is_door, is_recovery,
                round(sigma_co2, 2), round(sigma_temp, 2),
                readings['scd_c'], readings['scd_t'], readings['scd_h'],
                readings['bme_t'], readings['bme_h'], readings.get('bme_p', 0)
            ]
            for i in range(10):
                row.append(readings.get(f'gas_{i}', 0))

            try:
                with open(csv_filename, mode='a', newline='') as f:
                    csv.writer(f).writerow(row)
            except Exception as e:
                print(f"\n[ERROR] CSV Write: {e}")

            # --- C) Speichern (InfluxDB) ---
            if client:
                try:
                    p = Point("sensor_data").tag("label", experiment_label)
                    p.field("co2", float(readings['scd_c']))
                    p.field("bme_temp", float(readings['bme_t']))
                    p.field("door_open", int(is_door))
                    p.field("recovery", int(is_recovery))
                    # Haupt-Gaskanal (gas_0) für die Live-Ansicht
                    p.field("gas_res", float(readings.get('gas_0', 0)))
                    write_api.write(bucket=config.INFLUX_BUCKET, org=config.INFLUX_ORG, record=p)
                except:
                    pass

            # --- D) Terminal Feedback ---
            status = "OPEN  " if is_door else ("RECOV " if is_recovery else "STABLE")
            msg = f"\r[{datetime.now().strftime('%H:%M:%S')}] {status} | CO2: {int(readings['scd_c']):4}ppm | σ-CO2: {sigma_co2:5.1f} | Gas: {int(readings.get('gas_0',0)):6}Ω"
            print(msg, end="", flush=True)

            # Taktung einhalten (SAMPLING_RATE)
            elapsed = time.time() - loop_start
            time.sleep(max(0, config.SAMPLING_RATE - elapsed))

    except KeyboardInterrupt:
        print(f"\n\n[!] Aufnahme beendet. Datei: {csv_filename}")
        if client: client.close()

if __name__ == "__main__":
    main()