import time
import board
import busio
import adafruit_scd30
import adafruit_bme680

class SensorManager:
    def __init__(self):
        # I2C Bus mit niedriger Frequenz (20kHz) für SCD30 Stabilität
        self.i2c = busio.I2C(board.SCL, board.SDA, frequency=20000)

        # SCD30 Initialisierung
        try:
            self.scd = adafruit_scd30.SCD30(self.i2c)
            self.scd.measurement_interval = 2 # Mindestintervall laut Datenblatt
        except:
            self.scd = None

        # BME688 Initialisierung
        try:
            self.bme = adafruit_bme680.Adafruit_BME680_I2C(self.i2c)

            # Standard-Heizprofil aktivieren
            self.bme.gas_heat_temperature = 320
            self.bme.gas_heat_duration = 150

            # Initialer Buffer für 10 Kanäle (Masken)
            self.gas_index = 0
            self.gas_buffer = [0] * 10
        except:
            self.bme = None

    def get_formatted_data(self):
        # Resultat-Dictionary mit allen Feldern für main_logger.py vorinitialisieren
        res = {f"gas_{i}": 0 for i in range(10)}
        res.update({
            "bme_t": None, "bme_h": None, "bme_p": None, "bme_g": None,
            "scd_c": None, "scd_t": None, "scd_h": None
        })

        # BME688 Messung
        if self.bme:
            try:
                # Aktuellen Gaswiderstand der aktuellen Heizstufe lesen
                gas_res = int(self.bme.gas)
                res["bme_g"] = gas_res
                res["bme_t"] = round(self.bme.temperature, 2)
                res["bme_h"] = round(self.bme.relative_humidity, 2)
                res["bme_p"] = round(self.bme.pressure, 2)

                # Aktuellen Wert im Buffer an der richtigen Position speichern
                self.gas_buffer[self.gas_index] = gas_res

                # Den gesamten Buffer (alle 10 Masken) in das Resultat-Dict mappen
                for i in range(10):
                    res[f"gas_{i}"] = self.gas_buffer[i]

                # Index für die nächste Heizstufe (Maske) im nächsten Zyklus erhöhen
                self.gas_index = (self.gas_index + 1) % 10
            except:
                pass

        # SCD30 Messung (nur wenn neue Daten bereitstehen)
        if self.scd and self.scd.data_available:
            try:
                res["scd_c"] = int(self.scd.CO2)
                res["scd_t"] = round(self.scd.temperature, 2)
                res["scd_h"] = round(self.scd.relative_humidity, 2)
            except:
                pass

        return res