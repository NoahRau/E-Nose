import time
import board
import busio
from adafruit_bme680 import Adafruit_BME680_I2C
import adafruit_scd30

# I2C Initialisierung
i2c = board.I2C()

print("--- Hardware Test ---")

# 1. BME688 Test
try:
    # Adresse 0x77 probieren (laut deinem i2cdetect)
    bme = Adafruit_BME680_I2C(i2c, address=0x77)
    print(f"[SUCCESS] BME688 gefunden! Temp: {bme.temperature:.1f}C Pressure: {bme.pressure:.1f}hPa")
except Exception as e:
    print(f"[FAIL] BME688 Fehler: {e}")

# 2. SCD30 Test
try:
    scd = adafruit_scd30.SCD30(i2c)
    print("[WAIT] Warte auf SCD30 Daten (ca. 2s)...")
    while not scd.data_available:
        time.sleep(0.5)
    print(f"[SUCCESS] SCD30 gefunden! CO2: {scd.CO2:.0f}ppm")
except Exception as e:
    print(f"[FAIL] SCD30 Fehler: {e}")

print("--- Test Ende ---")