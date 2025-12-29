import time
import struct
import sys
import smbus
import math

# BME680 Library Import
try:
    import bme680
except ImportError:
    pass

class SCD30_Native:
    def __init__(self, bus_id=1, address=0x61):
        self.bus_id = bus_id
        self.addr = address
        self.connected = False
        self.bus = None

        # Speicher für die letzten guten Werte (Memory-Effekt)
        self.last_co2 = None
        self.last_temp = None
        self.last_hum = None

        try:
            self.bus = smbus.SMBus(bus_id)
            # 1. Ping (Test ob Sensor da ist)
            try:
                self.bus.write_i2c_block_data(self.addr, 0xD1, [0x00])
            except:
                pass
            self.connected = True

            # --- INITIALISIERUNG ---
            # Reset (0xD304)
            self._write(0xD304)
            time.sleep(2.0) # Sensor braucht 2s zum Booten

            # Intervall setzen (0x4600) auf 2 Sekunden
            # WICHTIG: Hier berechnet _write jetzt die korrekte CRC (0xE3)
            self._write(0x4600, [0x00, 0x02])
            time.sleep(0.1)

            # Start Messung (0x0010)
            self._write(0x0010, [0x00, 0x00])
            print(f"   [SCD30] Initialisiert auf Bus {bus_id}.")

        except Exception as e:
            print(f"   [SCD30] Fehler Init: {e}")
            self.connected = False

    def _calc_crc(self, data):
        """Berechnet die CRC8 Checksumme nach Sensirion Vorgabe"""
        crc = 0xFF
        for b in data:
            crc ^= b
            for _ in range(8):
                if crc & 0x80:
                    crc = (crc << 1) ^ 0x31
                else:
                    crc = (crc << 1)
        return crc & 0xFF

    def _write(self, cmd, args=None):
        """Sendet Befehle MIT korrekter CRC Berechnung"""
        if not self.connected: return

        # Befehl zerlegen (MSB, LSB)
        data = [(cmd >> 8) & 0xFF, cmd & 0xFF]

        if args:
            data.extend(args)
            # CRC für die Argumente berechnen und anhängen
            crc = self._calc_crc(args)
            data.append(crc)

        try:
            self.bus.write_i2c_block_data(self.addr, data[0], data[1:])
            # Wichtig: Dem Sensor Zeit geben, den Befehl zu verarbeiten
            time.sleep(0.05)
        except OSError:
            pass

    def _check_crc_received(self, data):
        """Prüft empfangene Daten auf Bit-Fehler"""
        # Datenformat: [Byte1, Byte2, CRC]
        return self._calc_crc(data[:2]) == data[2]

    def read_measurement(self):
        # Wenn nicht verbunden, gib alte Werte oder None zurück
        if not self.connected: return self.last_co2, self.last_temp, self.last_hum

        try:
            # 1. Daten anfordern (Blindflug, wir sparen uns das Polling)
            self._write(0x0300)

            # WICHTIG: Wartezeit. Der Sensor muss die Daten bereitstellen.
            time.sleep(0.05)

            # 2. 18 Bytes lesen
            raw = self.bus.read_i2c_block_data(self.addr, 0, 18)

            # Check: Ist der Bus leer (0xFF)?
            if raw[0] == 0xFF:
                return self.last_co2, self.last_temp, self.last_hum

            # Check: Stimmen die Prüfsummen? (Schutz vor wilden Werten)
            if not self._check_crc_received(raw[0:3]) or \
                    not self._check_crc_received(raw[6:9]) or \
                    not self._check_crc_received(raw[12:15]):
                # CRC falsch -> Paket verwerfen -> Alte Werte nutzen
                return self.last_co2, self.last_temp, self.last_hum

            # 3. Umrechnung Bytes -> Float
            def parse(b):
                return struct.unpack('>f', bytes([b[0], b[1], b[3], b[4]]))[0]

            co2 = parse(raw[0:6])
            temp = parse(raw[6:12])
            hum = parse(raw[12:18])

            # Check: Plausibilität (Filtert 0ppm oder 40000+ppm)
            if co2 < 1.0 or co2 > 40000.0:
                return self.last_co2, self.last_temp, self.last_hum

            # Alles OK -> Werte speichern und zurückgeben
            self.last_co2 = co2
            self.last_temp = temp
            self.last_hum = hum
            return co2, temp, hum

        except Exception:
            # Bei jeglichem I2C Fehler: Nicht abstürzen, alte Werte nutzen
            return self.last_co2, self.last_temp, self.last_hum

class SensorManager:
    def __init__(self):
        self.bme = None
        self.scd = None

        # --- BME688 Setup (Bus 1) ---
        try:
            # Pimoroni BME680 Initialisierung
            try:
                self.bme = bme680.BME680(bme680.I2C_ADDR_SECONDARY)
            except IOError:
                self.bme = bme680.BME680(bme680.I2C_ADDR_PRIMARY)

            # Settings für stabile Heizung
            self.bme.set_humidity_oversample(bme680.OS_2X)
            self.bme.set_pressure_oversample(bme680.OS_4X)
            self.bme.set_temperature_oversample(bme680.OS_8X)
            self.bme.set_filter(bme680.FILTER_SIZE_3)
            self.bme.set_gas_status(bme680.ENABLE_GAS_MEAS)
            self.bme.set_gas_heater_temperature(320)
            self.bme.set_gas_heater_duration(150)
            self.bme.select_gas_heater_profile(0)
            print("   [BME688] OK.")
        except Exception:
            print("   [BME688] Nicht gefunden.")

        # --- SCD30 Setup (Bus 1) ---
        self.scd = SCD30_Native(bus_id=1)

    def get_formatted_data(self):
        result = { "bme_t": None, "bme_h": None, "bme_g": None,
                   "scd_c": None, "scd_t": None, "scd_h": None }

        # 1. BME auslesen (Schnell)
        if self.bme and self.bme.get_sensor_data():
            result["bme_t"] = round(self.bme.data.temperature, 2)
            result["bme_h"] = round(self.bme.data.humidity, 2)
            if self.bme.data.heat_stable:
                result["bme_g"] = int(self.bme.data.gas_resistance)

        # 2. BUS-PAUSE (Die wichtigste Regel!)
        # Gib dem Bus Zeit, sich zu erholen, bevor der SCD30 angesprochen wird
        time.sleep(0.2)

        # 3. SCD auslesen (Langsam, mit Checks)
        c, t, h = self.scd.read_measurement()
        if c is not None:
            result["scd_c"] = int(c)
            result["scd_t"] = round(t, 2)
            result["scd_h"] = round(h, 2)

        return result