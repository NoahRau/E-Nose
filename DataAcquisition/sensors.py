# sensors.py
import board
import busio
import adafruit_scd30
from adafruit_bme680 import Adafruit_BME680_I2C

class SensorManager:
    def __init__(self):
        # I2C Bus initialisieren (SDA=GPIO2, SCL=GPIO3)
        self.i2c = board.I2C()

        # --- SCD30 Setup ---
        try:
            self.scd30 = adafruit_scd30.SCD30(self.i2c)
            # Messintervall des Sensors einstellen (nicht zu schnell!)
            self.scd30.measurement_interval = 2
            print("[Sensors] SCD30 gefunden.")
        except Exception as e:
            print(f"[ERROR] SCD30 nicht gefunden: {e}")
            self.scd30 = None

        # --- BME688 Setup ---
        try:
            # Adresse ist meist 0x77 oder 0x76
            try:
                self.bme680 = Adafruit_BME680_I2C(self.i2c, address=0x77)
            except:
                self.bme680 = Adafruit_BME680_I2C(self.i2c, address=0x76)

            # Basis-Konfiguration für Gas-Messung
            self.bme680.sea_level_pressure = 1013.25
            print("[Sensors] BME688 gefunden.")
        except Exception as e:
            print(f"[ERROR] BME688 nicht gefunden: {e}")
            self.bme680 = None

    def read_data(self):
        """Liest alle Sensoren aus und gibt ein Dictionary zurück"""
        data = {}

        # 1. SCD30 lesen
        if self.scd30 and self.scd30.data_available:
            data['co2'] = self.scd30.CO2
            data['scd_temp'] = self.scd30.temperature
            data['scd_humidity'] = self.scd30.relative_humidity

        # 2. BME688 lesen
        if self.bme680:
            data['bme_temp'] = self.bme680.temperature
            data['gas_resistance'] = self.bme680.gas
            data['bme_humidity'] = self.bme680.relative_humidity
            data['pressure'] = self.bme680.pressure

        return data