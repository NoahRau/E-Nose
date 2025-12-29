# main_logger.py
import time
import sys
import csv
import os
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
    print("   E-NOSE DATA RECORDER (Full Parallel Logging)")
    print("="*45)

    # 1. Label abfragen
    while True:
        label = input(">> LABEL eingeben (z.B. Testlauf_Beide_Sensoren): ").strip()
        if label:
            label = label.replace(" ", "_").replace("/", "-")
            break
        print("[!] Label darf nicht leer sein.")

    # 2. Laufzeit abfragen
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

def main():
    # --- Interaktiver Start ---
    experiment_label, auto_stop_time = get_user_input()

    print("\n" + "-"*40)
    print("   SYSTEM START")
    print("-"*40)

    # 1. Dateinamen generieren
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"data_{experiment_label}_{timestamp_str}.csv"

    # 2. Header definieren - Jetzt mit Spalten für beide Sensoren
    csv_header = [
        "timestamp", "datetime", "label",
        "soft_door_open", "sigma_co2", "sigma_temp",
        "scd_co2", "scd_temp", "scd_hum",      # SCD30 Rohdaten
        "bme_temp", "bme_hum", "bme_gas"       # BME688 Rohdaten
    ]

    # Datei anlegen
    try:
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
        print(f"[OK] CSV-Aufzeichnung gestartet: {csv_filename}")
    except Exception as e:
        print(f"[ERROR] Konnte CSV Datei nicht erstellen: {e}")
        sys.exit(1)

    # 3. Verbindung zur InfluxDB
    client = None
    try:
        client = InfluxDBClient(
            url=config.INFLUX_URL,
            token=config.INFLUX_TOKEN,
            org=config.INFLUX_ORG
        )
        write_api = client.write_api(write_options=SYNCHRONOUS)
        print("[OK] DB Verbindung erfolgreich.")
    except Exception as e:
        print(f"[WARN] InfluxDB nicht erreichbar: {e}")

    # 4. Initialisierung
    sensors = SensorManager()
    door_logic = AdaptiveDoorDetector(window_size=60, sensitivity=4.0)
    print("[*] Sensoren und Logik bereit.")

    try:
        while True:
            loop_start = time.time()

            # Auto-Stop prüfen
            if auto_stop_time and datetime.now() > auto_stop_time:
                print("\n[FINISH] Automatische Laufzeit beendet.")
                break

            # --- A) Messen ---
            readings = sensors.get_formatted_data()

            # Falls SCD30 noch keine Daten hat, kurz warten
            if readings['scd_c'] is None:
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Warte auf SCD30...", end="")
                time.sleep(1)
                continue

            # --- B) Denken (Adaptive Logic) ---
            # Wir nutzen SCD30 für CO2 und BME688 für die Temperatur-Anomalie (reagiert oft schneller)
            is_door, sigma_co2, sigma_temp = door_logic.update(readings['scd_c'], readings['bme_t'])

            # --- C) Speichern (CSV) ---
            try:
                with open(csv_filename, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        loop_start,
                        datetime.now().isoformat(),
                        experiment_label,
                        is_door,
                        round(sigma_co2, 2),
                        round(sigma_temp, 2),
                        # SCD30 Block
                        readings['scd_c'],
                        readings['scd_t'],
                        readings['scd_h'],
                        # BME688 Block
                        readings['bme_t'],
                        readings['bme_h'],
                        readings['bme_g']
                    ])
            except Exception as e:
                print(f"[ERROR] CSV Write: {e}")

            # --- D) Speichern (InfluxDB) ---
            if client:
                try:
                    point = Point("sensor_metrics").tag("experiment", experiment_label)
                    point.field("scd_co2", float(readings['scd_c']))
                    point.field("scd_temp", float(readings['scd_t']))
                    point.field("bme_temp", float(readings['bme_t']))
                    point.field("gas_res", float(readings['bme_g'] or 0))
                    point.field("door_open", int(is_door))
                    write_api.write(bucket=config.INFLUX_BUCKET, org=config.INFLUX_ORG, record=point)
                except:
                    pass

            # --- E) Feedback ---
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] CO2: {int(readings['scd_c'])} | T-SCD: {readings['scd_t']}° | T-BME: {readings['bme_t']}° | Tür: {is_door}", end="")

            # Taktung einhalten (Standard 2s)
            elapsed = time.time() - loop_start
            time.sleep(max(0, config.SAMPLING_RATE - elapsed))

    except KeyboardInterrupt:
        print(f"\n\n[!] Aufnahme beendet. Daten gespeichert in: {csv_filename}")
        if client: client.close()

if __name__ == "__main__":
    main()