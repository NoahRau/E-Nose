import time
from DataAcquisition.sensors import SensorManager

manager = SensorManager()

print("\n--- DEBUG MESSUNG ---")
print("Zeigt rohe Daten an, auch wenn sie falsch sind.")

try:
    while True:
        d = manager.get_data()

        print(f"[BME] {d['bme_raw']}   ||   [SCD] {d['scd_status']} -> {d['scd_co2'] or '---'}")

        time.sleep(2.0)

except KeyboardInterrupt:
    print("\nEnde.")