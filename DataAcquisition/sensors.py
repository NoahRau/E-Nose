import time
import struct
import sys
import smbus
import math

# BME680 Library
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

        self.last_co2 = None
        self.last_temp = None
        self.last_hum = None

        try:
            self.bus = smbus.SMBus(bus_id)
            # --- SETUP NACH DATENBLATT ---
            # 1. Reset
            self._write_cmd(0xD304)
            time.sleep(2.0)

            # 2. Intervall 2s (Cmd 0x4600, Arg 0x0002, CRC 0xE3)
            self._write_cmd(0x4600, [0x00, 0x02])
            time.sleep(0.1)

            # 3. Start (Cmd 0x0010, Arg 0x0000, CRC 0x81)
            self._write_cmd(0x0010, [0x00, 0x00])

            self.connected = True
            print(f"   [SCD30] Verbunden (Auto-Align Mode).")

        except Exception as e:
            print(f"   [SCD30] Init Fehler: {e}")
            self.connected = False

    def _calc_crc(self, data):
        """Berechnet CRC8 nach Sensirion Formel (Polynom 0x31)"""
        crc = 0xFF
        for b in data:
            crc ^= b
            for _ in range(8):
                if crc & 0x80: crc = (crc << 1) ^ 0x31
                else: crc = (crc << 1)
        return crc & 0xFF

    def _write_cmd(self, cmd, args=None):
        if not self.connected: return
        data = [(cmd >> 8) & 0xFF, cmd & 0xFF]
        if args:
            data.extend(args)
            data.append(self._calc_crc(args)) # CRC berechnen!
        try:
            self.bus.write_i2c_block_data(self.addr, data[0], data[1:])
            time.sleep(0.05)
        except OSError:
            pass

    def _check_crc_block(self, data):
        """Prüft einen 3-Byte Block [Byte1, Byte2, CRC]"""
        return self._calc_crc(data[:2]) == data[2]

    def read_measurement(self):
        if not self.connected: return self.last_co2, self.last_temp, self.last_hum

        try:
            # 1. READ REQUEST (0x0300)
            self._write_cmd(0x0300)
            time.sleep(0.05) # Warten > 3ms laut Datenblatt

            # 2. ROHDATEN LESEN (Mehr lesen, um Verschiebung zu fangen)
            # Wir lesen 24 Bytes statt 18, falls der Anfang fehlt
            raw = self.bus.read_i2c_block_data(self.addr, 0, 24)

            # 3. INTELLIGENTE SUCHE (ALIGNMENT)
            # Wir suchen im Datenstrom nach gültigen Blöcken.
            # Ein valider Frame besteht aus 3 Blöcken à 3 Bytes (CO2, Temp, Hum)

            # Wir testen jeden möglichen Startpunkt (Offset 0 bis 6)
            for i in range(7):
                # Wir brauchen 18 Bytes ab Position i
                if len(raw) < i + 18: break

                # Kandidat für den Daten-Frame
                frame = raw[i : i+18]

                # Check: Stimmen die CRCs für CO2(0-2), Temp(6-8) und Hum(12-14)?
                if self._check_crc_block(frame[0:3]) and \
                        self._check_crc_block(frame[6:9]) and \
                        self._check_crc_block(frame[12:15]):

                    # TREFFER! Wir haben die Verschiebung gefunden.
                    # Jetzt strikt nach Datenblatt umwandeln (Big Endian Float)
                    def parse_float(idx):
                        b = [frame[idx], frame[idx+1], frame[idx+3], frame[idx+4]]
                        return struct.unpack('>f', bytes(b))[0]

                    co2 = parse_float(0)
                    temp = parse_float(6)
                    hum = parse_float(12)

                    # Plausibilitäts-Check (Filtert 0ppm Bugs)
                    if co2 < 1.0 or co2 > 40000.0: continue

                    self.last_co2 = co2
                    self.last_temp = temp
                    self.last_hum = hum
                    return co2, temp, hum

            # Wenn kein gültiges Muster gefunden wurde
            return self.last_co2, self.last_temp, self.last_hum

        except Exception:
            return self.last_co2, self.last_temp, self.last_hum

class SensorManager:
    def __init__(self):
        self.bme = None
        self.scd = None

        # BME Setup
        try:
            try:
                self.bme = bme680.BME680(bme680.I2C_ADDR_SECONDARY)
            except IOError:
                self.bme = bme680.BME680(bme680.I2C_ADDR_PRIMARY)
            # BME Settings
            self.bme.set_humidity_oversample(bme680.OS_2X)
            self.bme.set_pressure_oversample(bme680.OS_4X)
            self.bme.set_temperature_oversample(bme680.OS_8X)
            self.bme.set_filter(bme680.FILTER_SIZE_3)
            self.bme.set_gas_status(bme680.ENABLE_GAS_MEAS)
            self.bme.set_gas_heater_temperature(320)
            self.bme.set_gas_heater_duration(150)
            self.bme.select_gas_heater_profile(0)
        except Exception:
            pass

        self.scd = SCD30_Native(bus_id=1)

    def get_formatted_data(self):
        result = { "bme_t": None, "bme_h": None, "bme_g": None,
                   "scd_c": None, "scd_t": None, "scd_h": None }

        # BME
        if self.bme and self.bme.get_sensor_data():
            result["bme_t"] = round(self.bme.data.temperature, 2)
            result["bme_h"] = round(self.bme.data.humidity, 2)
            if self.bme.data.heat_stable:
                result["bme_g"] = int(self.bme.data.gas_resistance)

        time.sleep(0.1) # Bus-Pause

        # SCD
        c, t, h = self.scd.read_measurement()
        if c is not None:
            result["scd_c"] = int(c)
            result["scd_t"] = round(t, 2)
            result["scd_h"] = round(h, 2)

        return result