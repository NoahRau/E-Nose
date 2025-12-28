import time
from DataAcquisition.sensors import SensorManager

print("\n--- INITIALISIERUNG ---")
manager = SensorManager()

print("\n--- REGISTER & BUS DIAGNOSE ---")
# Wir rufen die Diagnose-Funktion auf, die wir eben gebaut haben
diag_report = manager.diagnose_register()
print(diag_report)
print("-" * 40)

print("\n--- STARTE LIVE MESSUNG ---")
print("Hinweis: BME zeigt hier RAW-Werte. Wenn die sich ändern, funktioniert er.")
print("Zombie-Wert (Schlafend) ist oft um die 327680 (0x80000).")

try:
    while True:
        d = manager.get_data()

        print(f"[BME] {d['bme_raw']}   ||   [SCD] {d['scd_co2']}")

        time.sleep(2.0)
except KeyboardInterrupt:
    print("Ende")