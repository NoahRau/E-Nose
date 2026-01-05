import os
import time

from DataAcquisition.sensors import SensorManager


# Screen clear helper
def clear():
    os.system("cls" if os.name == "nt" else "clear")


print("--- System Start ---")
manager = SensorManager()
print("Sensoren initialisiert. Warte auf stabile Werte (ca. 10s)...")
time.sleep(2)

# Header
print(
    f"\n{'ZEIT':<10} | {'BME T':<8} {'BME H':<8} {'GAS':<8} || {'SCD CO2':<10} {'SCD T':<8} {'SCD H':<8}"
)
print("-" * 80)

try:
    while True:
        data = manager.get_formatted_data()

        t_str = time.strftime("%H:%M:%S")

        # BME Formatierung
        if data["bme_t"] is not None:
            bme_part = (
                f"{data['bme_t']}°C".ljust(9)
                + f"{data['bme_h']}%".ljust(9)
                + f"{data['bme_g'] or '-'}".ljust(9)
            )
        else:
            bme_part = "WARTE... ".ljust(27)

        # SCD Formatierung
        if data["scd_c"] is not None:
            # Hier haben wir jetzt echte Werte!
            scd_part = (
                f"|| {data['scd_c']} ppm".ljust(13)
                + f"{data['scd_t']}°C".ljust(9)
                + f"{data['scd_h']}%".ljust(8)
            )
        else:
            # Wenn der Sensor noch spinnt oder kalibriert
            scd_part = "|| WARTE... (Kalibrierung)"

        print(f"{t_str:<10} | {bme_part} {scd_part}")

        time.sleep(1.0)

except KeyboardInterrupt:
    print("\nEnde.")
