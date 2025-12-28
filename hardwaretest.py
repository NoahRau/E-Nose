import time
from sensors import SensorManager

# Hier passiert der magische Reset im Hintergrund
manager = SensorManager()

print("\n--- Start Messung (Reset durchgeführt) ---")
print("HINWEIS: Es kann 10-20 Sekunden dauern, bis sich die Werte stabilisieren.")
time.sleep(2)

try:
    while True:
        # 1. Daten holen
        d = manager.get_data()

        # 2. BME Ausgabe
        bme_str = "[BME] --"
        if d["bme_temp"]:
            bme_str = f"[BME] T:{d['bme_temp']}°C H:{d['bme_hum']}% P:{d['bme_press']}hPa G:{d['bme_gas']}Ω"

        # 3. SCD Ausgabe
        scd_str = "[SCD] --"
        if d["scd_co2"]:
            scd_str = f"[SCD] CO2:{d['scd_co2']}ppm T:{d['scd_temp']}°C"
        elif manager.scd and manager.scd.is_connected:
            scd_str = "[SCD] Warte..."

        print(f"{bme_str}   |   {scd_str}")

        time.sleep(2.0) # SCD30 misst eh nur alle 2 Sekunden

except KeyboardInterrupt:
    print("\nEnde.")