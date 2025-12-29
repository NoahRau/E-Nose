import time
import board
import busio
import adafruit_scd30
import adafruit_bme680

class SensorManager:
    def __init__(self):
        self.scd = None
        self.bme = None
        self.i2c = None

        try:
            # 1. I2C Bus initialisieren (Nutzt automatisch deinen Config-Hack auf Bus 1)
            # board.SCL und board.SDA sind Pin 3 und 5
            self.i2c = busio.I2C(board.SCL, board.SDA, frequency=20000) # 20 kHz für Stabilität
            print("   [I2C] Bus gestartet.")
        except Exception as e:
            print(f"   [I2C] Kritischer Fehler: {e}")
            return

        # 2. SCD30 Setup (Offizielle Library)
        try:
            self.scd = adafruit_scd30.SCD30(self.i2c)
            self.scd.measurement_interval = 2
            print("   [SCD30] Verbunden (Adafruit Driver).")
        except Exception as e:
            print(f"   [SCD30] Nicht gefunden: {e}")

        # 3. BME680 Setup (Offizielle Library)
        try:
            self.bme = adafruit_bme680.Adafruit_BME680_I2C(self.i2c, address=0x77)
            # Settings
            self.bme.sea_level_pressure = 1013.25
            print("   [BME688] Verbunden (Adafruit Driver).")
        except Exception:
            try:
                # Fallback auf Adresse 0x76
                self.bme = adafruit_bme680.Adafruit_BME680_I2C(self.i2c, address=0x76)
                print("   [BME688] Verbunden (Addr 0x76).")
            except Exception as e:
                print(f"   [BME688] Fehler: {e}")

    def get_formatted_data(self):
        result = { "bme_t": None, "bme_h": None, "bme_g": None,
                   "scd_c": None, "scd_t": None, "scd_h": None }

        # BME Lesen
        if self.bme:
            try:
                result["bme_t"] = round(self.bme.temperature, 2)
                result["bme_h"] = round(self.bme.relative_humidity, 2)
                result["bme_g"] = int(self.bme.gas)
            except:
                pass

        # SCD Lesen
        if self.scd:
            try:
                # Die Library prüft intern "data_ready".
                # Wir müssen nur prüfen, ob Daten da sind.
                if self.scd.data_available:
                    result["scd_c"] = int(self.scd.CO2)
                    result["scd_t"] = round(self.scd.temperature, 2)
                    result["scd_h"] = round(self.scd.relative_humidity, 2)
            except Exception:
                pass # Fehler werden von der Lib abgefangen

        return result