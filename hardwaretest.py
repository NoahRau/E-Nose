import time
import os
from DataAcquisition.sensors import SensorManager

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

print("--- Initialisiere E-Nose Hardware ---")
manager = SensorManager()

# Diagnose Bericht
print(f"BME688 (Bus 1): {'✅ ONLINE' if manager.status['bme'] else '❌ OFFLINE'}")
print(f"SCD30  (Bus 3): {'✅ ONLINE' if manager.status['scd'] else '❌ OFFLINE'}")
print("-" * 60)
time.sleep(2)

print(f"{'ZEIT':<10} | {'BME TEMP':<10} {'BME HUM':<10} {'GAS (Ω)':<10} || {'SCD CO2':<10} {'SCD TEMP':<10}")
print("-" * 75)

try:
    while True:
        data = manager.get_formatted_data()

        # BME String bauen
        if data['bme_t'] is not None:
            bme_str = f"{data['bme_t']}°C".ljust(10) + f"{data['bme_h']}%".ljust(10) + f"{data['bme_g'] or '---'}".ljust(10)
        else:
            bme_str = "WARTE...".ljust(30)

        # SCD String bauen
        if data['scd_c'] is not None:
            scd_str = f"|| {data['scd_c']}ppm".ljust(11) + f"{data['scd_t']}°C".ljust(10)
        else:
            scd_str = "|| WARTE...".ljust(21)

        timestamp = time.strftime("%H:%M:%S")
        print(f"{timestamp:<10} | {bme_str} {scd_str}")

        time.sleep(1.0)

except KeyboardInterrupt:
    print("\nTest beendet.")