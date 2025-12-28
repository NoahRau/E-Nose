import time
from DataAcquisition.sensors import SensorManager

manager = SensorManager()

print("\n--- AGGRESSIVE MESSUNG ---")
try:
    while True:
        d = manager.get_data()
        print(f"[BME] {d['bme_raw']}   ||   [SCD] {d['scd_co2']}")
        time.sleep(2.0)
except KeyboardInterrupt:
    print("Ende")