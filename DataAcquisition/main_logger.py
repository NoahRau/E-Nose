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
from door_detector import AdaptiveDoorDetector  # <--- Importiert deine neue Klasse

def get_user_input():
    """Fragt den Nutzer interaktiv nach Experiment-Parametern"""
    print("\n" + "="*45)
    print("   E-NOSE DATA RECORDER (Adaptive Logic)")
    print("="*45)

    # 1. Label abfragen
    while True:
        label = input(">> LABEL eingeben (z.B. erdbeere_tag1): ").strip()
        if label:
            label = label.replace(" ", "_").replace("/", "-")
            break
        print("[!] Label darf nicht leer sein.")

    # 2. Laufzeit abfragen
    print("\n>> LAUFZEIT eingeben (in Stunden)")
    print("   (Drücke ENTER für Endlos-Aufnahme bis STRG+C)")
    duration_input = input("   Stunden: ").strip()

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

    # 2. Header definieren
    # Wir speichern jetzt auch die "Sigmas" (Abweichungsstärken), damit du das Tuning prüfen kannst
    csv_header = [
        "timestamp", "datetime", "label",
        "soft_door_open", "sigma_co2", "sigma_temp",  # <--- Unsere KI-Features
        "co2", "temp", "humidity", "pressure"
    ]
    # Gas Kanäle hinzufügen (gas_0 bis gas_9) für das BME Heater Profile
    gas_cols = [f"gas_{i}" for i in range(10)]
    csv_header.extend(gas_cols)

    # Datei anlegen
    try:
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
        print(f"[OK] CSV-Aufzeichnung gestartet: {csv_filename}")
    except Exception as e:
        print(f"[ERROR] Konnte CSV Datei nicht erstellen: {e}")
        sys.exit(1)

    # 3. Verbindung zur InfluxDB (Optional, läuft parallel)
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
        print(f"[WARN] InfluxDB nicht erreichbar (CSV läuft trotzdem): {e}")

    # 4. Initialisierung der Logik
    sensors = SensorManager()

    # Hier wird die adaptive Logik gestartet (Lernfenster: 60 Samples = ca 2 Min)
    door_logic = AdaptiveDoorDetector(window_size=60, sensitivity=4.0)
    print("[*] Adaptive Logik gestartet (Kalibriert sich auf Umgebung...)")

    print("\n" + "="*40)
    print(f"   RECORDER LÄUFT: {experiment_label}")
    print("   (Drücke STRG+C zum Beenden)")
    print("="*40)

    # 5. Hauptschleife
    try:
        while True:
            loop_start = time.time()

            # Auto-Stop prüfen
            if auto_stop_time and datetime.now() > auto_stop_time:
                print("\n[FINISH] Automatische Laufzeit beendet.")
                break

            # --- A) Messen ---
            # Wir nutzen read_all() für den Scan, falls verfügbar
            if hasattr(sensors, 'read_all'):
                readings = sensors.read_all()
            else:
                readings = sensors.read_data()

            if not readings:
                print("[WARN] Keine Sensordaten.")
                time.sleep(1)
                continue

            # Werte extrahieren (mit Fallbacks für Stabilität)
            curr_co2 = readings.get('co2', 0)
            # Nimm BME Temp wenn verfügbar, sonst SCD Temp
            curr_temp = readings.get('temp', readings.get('bme_temp', 0))

            # --- B) Denken (Adaptive Logic) ---
            is_door, sigma_co2, sigma_temp = door_logic.update(curr_co2, curr_temp)

            if is_door == 1:
                # Kleines visuelles Feedback im Terminal
                print(f"   >>> TÜR ERKANNT! (Sigma CO2: {sigma_co2:.1f})")

            # --- C) Speichern (CSV) ---
            try:
                with open(csv_filename, mode='a', newline='') as f:
                    writer = csv.writer(f)

                    row = [
                        loop_start,
                        datetime.now().isoformat(),
                        experiment_label,
                        is_door,            # 0 oder 1
                        round(sigma_co2, 2), # Debugging-Wert
                        round(sigma_temp, 2),# Debugging-Wert
                        curr_co2,
                        curr_temp,
                        readings.get('hum', readings.get('scd_humidity', 0)),
                        readings.get('pressure', 0)
                    ]
                    # Gas Array füllen (10 Werte oder 0)
                    for i in range(10):
                        row.append(readings.get(f'gas_{i}', 0))

                    writer.writerow(row)
            except Exception as e:
                print(f"[ERROR] CSV Write: {e}")

            # --- D) Speichern (InfluxDB) ---
            if client:
                try:
                    point = Point("environment_metrics")
                    point.tag("location", "fridge_box")
                    point.tag("experiment_label", experiment_label)

                    # Alles als Fields speichern
                    for k, v in readings.items():
                        if isinstance(v, (int, float)):
                            point.field(k, float(v))

                    # Logik-Ergebnisse auch speichern!
                    point.field("ai_door_detected", int(is_door))
                    point.field("ai_sigma_co2", float(sigma_co2))

                    write_api.write(bucket=config.INFLUX_BUCKET, org=config.INFLUX_ORG, record=point)
                except:
                    pass # DB Fehler ignorieren wir hier, CSV ist wichtiger

            # --- E) Feedback ---
            # Zeige aktuelle Sigma-Werte an, damit man sieht, wie "nervös" der Sensor ist
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] CO2: {int(curr_co2)} | σ-CO2: {sigma_co2:5.1f} | Tür: {is_door}", end="")

            # Taktung einhalten
            elapsed = time.time() - loop_start
            sleep_time = max(0, config.SAMPLING_RATE - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n[!] Aufnahme beendet.")
        if client: client.close()
        print(f"Daten gespeichert in: {csv_filename}")

if __name__ == "__main__":
    main()