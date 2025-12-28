import time
import board
import busio
import adafruit_scd30
from adafruit_bme680 import Adafruit_BME680_I2C

print("--- Start Test ---")

# Wir nutzen board.SCL und board.SDA.
# Dank dtoverlay=i2c-gpio in der config.txt ist das jetzt unser stabiler Software-Bus.
try:
    i2c = board.I2C()
except Exception as e:
    print(f"Standard board.I2C() Fehler: {e}")
    print("Versuche Fallback...")
    # Fallback, falls Blinka den Software-Bus nicht mag
    from adafruit_blinka.microcontroller.generic_linux.i2c import I2C as _I2C
    i2c = _I2C(1, mode=_I2C.MASTER)

# 1. BME688 Test
try:
    # Wichtig: Adresse explizit angeben (0x77)
    bme = Adafruit_BME680_I2C(i2c, address=0x77)
    # Einmal lesen zum "Aufwecken"
    _ = bme.temperature
    time.sleep(0.5)
    print(f"[OK] BME688: {bme.temperature:.1f}°C, {bme.pressure:.1f} hPa")
except Exception as e:
    print(f"[ERR] BME688: {e}")

# 2. SCD30 Test
try:
    scd = adafruit_scd30.SCD30(i2c)
    print("Warte auf SCD30 (kann 2-3s dauern)...")
    # Timeout Zähler, damit wir nicht ewig hängen
    timeout = 10
    while not scd.data_available and timeout > 0:
        time.sleep(0.5)
        timeout -= 1
        print(".", end="", flush=True)

    if scd.data_available:
        print(f"\n[OK] SCD30: CO2 {scd.CO2:.0f} ppm")
    else:
        print("\n[ERR] SCD30: Keine Daten (Timeout)")
except Exception as e:
    print(f"\n[ERR] SCD30: {e}")

print("--- Ende ---")