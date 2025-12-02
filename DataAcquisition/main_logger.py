# main_logger.py
import time
import sys
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# Eigene Module importieren
import config
from sensors import SensorManager

def main():
    print("--- E-Nose Data Logger Start ---")

    # 1. Verbindung zur Datenbank aufbauen
    print(f"[*] Verbinde zu InfluxDB ({config.INFLUX_URL})...")
    try:
        client = InfluxDBClient(
            url=config.INFLUX_URL,
            token=config.INFLUX_TOKEN,
            org=config.INFLUX_ORG
        )
        write_api = client.write_api(write_options=SYNCHRONOUS)
        print("[OK] DB Verbindung erfolgreich.")
    except Exception as e:
        print(f"[ERROR] Konnte nicht zur DB verbinden: {e}")
        sys.exit(1)

    # 2. Sensoren initialisieren
    sensors = SensorManager()

    print("[*] Starte Messschleife... (Drücke STRG+C zum Beenden)")

    # 3. Endlosschleife
    try:
        while True:
            start_time = time.time()

            # Daten lesen
            readings = sensors.read_data()

            if not readings:
                print("[WARN] Keine Daten von Sensoren erhalten.")
            else:
                # Datenpunkt für InfluxDB erstellen
                point = Point("environment_metrics")

                # Tags (Metadaten)
                point.tag("location", "fridge_box")

                # Fields (Messwerte) hinzufügen
                for key, value in readings.items():
                    if value is not None:
                        point.field(key, float(value))

                # In die DB schreiben
                try:
                    write_api.write(bucket=config.INFLUX_BUCKET, org=config.INFLUX_ORG, record=point)

                    # Console Output (optional, zur Kontrolle)
                    print(f"Saved: CO2={readings.get('co2', 'N/A')} | Gas={readings.get('gas_resistance', 'N/A')} | Temp={readings.get('bme_temp', 'N/A')}")

                except Exception as db_err:
                    print(f"[ERROR] DB Write Fehler: {db_err}")

            # Smart Sleep: Warte genau so lange, um den Takt zu halten
            elapsed = time.time() - start_time
            sleep_time = max(0, config.SAMPLING_RATE - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[!] Programm vom Benutzer beendet.")
        client.close()
        print("Bye!")

if __name__ == "__main__":
    main()