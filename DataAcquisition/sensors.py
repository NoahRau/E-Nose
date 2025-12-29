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
            self.i2c = busio.I2C(board.SCL, board.SDA, frequency=20000)
            print("   [I2C] Bus gestartet.")
        except Exception as e:
            print(f"   [I2C] Kritischer Fehler: {e}")
            return

        # SCD30 Setup
        try:
            self.scd = adafruit_scd30.SCD30(self.i2c)
            self.scd.measurement_interval = 2
            print("   [SCD30] Verbunden (Adafruit Driver).")
        except Exception as e:
            print(f"   [SCD30] Nicht gefunden: {e}")
            self.scd = None

        # BME680/688 Setup
        try:
            self.bme = adafruit_bme680.Adafruit_BME680_I2C(self.i2c, address=0x77)
            self.bme.sea_level_pressure = 1013.25
            print("   [BME688] Verbunden (Adafruit Driver).")
        except Exception:
            try:
                self.bme = adafruit_bme680.Adafruit_BME680_I2C(self.i2c, address=0x76)
                self.bme.sea_level_pressure = 1013.25
                print("   [BME688] Verbunden (Addr 0x76).")
            except Exception as e:
                print(f"   [BME688] Fehler: {e}")
                self.bme = None

    def get_formatted_data(self):
        result = {
            "bme_t": None, "bme_h": None, "bme_p": None, "bme_g": None,
            "scd_c": None, "scd_t": None, "scd_h": None
        }

        # BME
        if self.bme:
            try:
                result["bme_t"] = round(self.bme.temperature, 2)
                result["bme_h"] = round(self.bme.relative_humidity, 2)
                result["bme_p"] = round(self.bme.pressure, 2)

                # gas property name can vary
                gas_val = None
                try:
                    gas_val = self.bme.gas
                except Exception:
                    gas_val = None
                if gas_val is None:
                    try:
                        gas_val = self.bme.gas_resistance
                    except Exception:
                        gas_val = None

                if gas_val is not None:
                    result["bme_g"] = int(gas_val)
            except Exception:
                pass

        # SCD
        if self.scd:
            try:
                if self.scd.data_available:
                    result["scd_c"] = int(self.scd.CO2)
                    result["scd_t"] = round(self.scd.temperature, 2)
                    result["scd_h"] = round(self.scd.relative_humidity, 2)
            except Exception:
                pass

        return result
