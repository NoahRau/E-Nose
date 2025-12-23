import time
import board
import busio
import adafruit_scd30
from adafruit_bme680 import Adafruit_BME680_I2C

class SensorManager:
    def __init__(self):
        self.i2c = board.I2C()

        # --- SCD30 (Bleibt gleich) ---
        try:
            self.scd30 = adafruit_scd30.SCD30(self.i2c)
            self.scd30.measurement_interval = 2
            print("[Sensors] SCD30 ready.")
        except:
            self.scd30 = None

        # --- BME688 Setup ---
        try:
            self.bme680 = Adafruit_BME680_I2C(self.i2c, address=0x77)
        except:
            try:
                self.bme680 = Adafruit_BME680_I2C(self.i2c, address=0x76)
            except Exception as e:
                print(f"[ERROR] BME688 Error: {e}")
                self.bme680 = None

        if self.bme680:
            self.bme680.sea_level_pressure = 1013.25

            # --- BOSCH EMPFEHLUNG: Standard Heater Profile ---
            # Wir definieren 10 Stufen (Temp in °C, Dauer in ms)
            # Dies deckt ein breites Spektrum ab, um Fäule vs. Normal zu unterscheiden.
            self.heater_profile = [
                (320, 150), (320, 150), (320, 150), # High Heat (Cleaning)
                (250, 150), (250, 150),             # Mid Range
                (150, 150), (150, 150),             # Low Range (Sensible VOCs)
                (200, 150), (200, 150),
                (320, 150)                          # Final Burn
            ]

    def read_gas_scan(self):
        """
        Führt einen kompletten Gas-Scan durch (dauert ca. 2-3 Sekunden!).
        Gibt eine Liste mit 10 Gas-Widerständen zurück.
        """
        if not self.bme680:
            return [0] * 10

        gas_fingerprint = []

        # WICHTIG: Loop durch das Profil
        for temp, duration in self.heater_profile:
            # 1. Heizung für diesen Schritt konfigurieren
            self.bme680.gas_heater_temperature = temp
            self.bme680.gas_heater_duration = duration

            # 2. Messung erzwingen (Forced Mode)
            # Der Sensor misst jetzt mit DEN OBEN eingestellten Werten
            # Zugriff auf .gas property triggert bei Adafruit oft die Messung oder Abfrage
            # Sicherer Weg: Property lesen, die den I2C Transfer auslöst
            try:
                # Ein Dummy-Read, um den Sensor zu aktualisieren
                _ = self.bme680.temperature
                # Jetzt den Gaswert holen
                gas_val = self.bme680.gas
                gas_fingerprint.append(gas_val)
            except OSError:
                gas_fingerprint.append(-1)

            # Kurze Pause, um dem Sensor Zeit zu geben (optional, da duration im Sensor)
            # time.sleep(duration / 1000.0)

        return gas_fingerprint

    def read_all(self):
        """Holt SCD30 Daten + den BME Scan"""
        data = {}

        # SCD30
        if self.scd30 and self.scd30.data_available:
            data['co2'] = self.scd30.CO2
            data['temp'] = self.scd30.temperature
            data['hum'] = self.scd30.relative_humidity

        # BME Scan
        if self.bme680:
            scan_results = self.read_gas_scan()
            # Wir speichern das als gas_0 bis gas_9
            for i, val in enumerate(scan_results):
                data[f'gas_{i}'] = val

            # Basiswerte vom BME (vom letzten Schritt)
            data['pressure'] = self.bme680.pressure

        return data