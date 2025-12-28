import time
# Wir brauchen board hier gar nicht mehr zwingend für den Bus!
import adafruit_scd30
from adafruit_bme680 import Adafruit_BME680_I2C

# Das ist der Fix: Wir holen uns den Bus direkt über die ID
# Wenn dieser Import fehlschlägt, hast du "pip install adafruit-circuitpython-extended-bus" vergessen
try:
    from adafruit_extended_bus import ExtendedI2C as I2C
except ImportError:
    print("Bitte installiere erst die Bibliothek:")
    print("pip install adafruit-circuitpython-extended-bus")
    exit()

print("--- Start Test ---")

# Wir erzwingen Bus 1 (weil ls /dev/i2c* bei dir i2c-1 gezeigt hat)
try:
    i2c = I2C(1)
    print("I2C Bus 1 erfolgreich verbunden (Software/GPIO Modus).")
except Exception as e:
    print(f"Kritischer Fehler beim Laden des I2C Bus: {e}")
    exit()

# 1. BME688 Test
try:
    # Adresse explizit lassen (0x77 ist Standard bei vielen Modulen, sonst 0x76 testen)
    bme = Adafruit_BME680_I2C(i2c, address=0x77)

    # Der erste Zugriff wirft oft Fehler wenn der Bus schläft, daher kurz warten
    time.sleep(0.1)
    temp = bme.temperature
    press = bme.pressure

    print(f"[OK] BME688: {temp:.1f}°C, {press:.1f} hPa")
except Exception as e:
    print(f"[ERR] BME688: {e}")
    print("     -> Tipp: Prüfe, ob die Adresse evtl. 0x76 ist.")

# 2. SCD30 Test
try:
    scd = adafruit_scd30.SCD30(i2c)
    # SCD30 mag langsame Taktraten, die haben wir jetzt durch den Overlay!
    print("Warte auf SCD30 Daten (Aufwärmen)...")

    timeout = 10
    while not scd.data_available and timeout > 0:
        time.sleep(0.5)
        timeout -= 1
        print(".", end="", flush=True)

    if scd.data_available:
        print(f"\n[OK] SCD30: CO2 {scd.CO2:.0f} ppm, T: {scd.temperature:.1f}°C, RH: {scd.relative_humidity:.0f}%")
    else:
        print("\n[ERR] SCD30: Timeout - Sensor antwortet, hat aber keine neuen Daten.")
except Exception as e:
    print(f"\n[ERR] SCD30: {e}")

print("--- Ende ---")