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
            # Ping
            try:
                self.bus.write_i2c_block_data(self.addr, 0xD1, [0x00])
            except:
                pass
            self.connected = True

            # --- TURBO START ---
            # Wir machen KEINEN Reset mehr (das dauert 2s).
            # Wir setzen nur das Intervall, falls es falsch ist.
            try:
                # Intervall 2s, CRC ist 0xE3
                self.bus.write_i2c_block_data(self.addr, 0x46, [0x00, 0x00, 0x02, 0xE3])
                time.sleep(0.05)
                # Start Messung (falls noch nicht läuft), CRC ist 0x81
                self.bus.write_i2c_block_data(self.addr, 0x00, [0x10, 0x00, 0x00, 0x81])
            except:
                pass # Wenn er schon läuft, ignorieren wir Fehler

            print(f"   [SCD30] Verbunden (Turbo Mode).")

        except Exception as e:
            print(f"   [SCD30] Fehler: {e}")
            self.connected = False

    def _calc_crc(self, data):
        """Berechnet CRC8"""
        crc = 0xFF
        for b in data:
            crc ^= b
            for _ in range(8):
                if crc & 0x80: crc = (crc << 1) ^ 0x31
                else: crc = (crc << 1)
        return crc & 0xFF

    def _check_crc_received(self, data):
        """Prüft Datenpaket"""
        return self._calc_crc(data[:2]) == data[2]

    def read_measurement(self):
        if not self.connected: return self.last_co2, self.last_temp, self.last_hum

        # --- DIE SCHLEIFE GEGEN DAS WARTEN ---
        # Wir versuchen es 5 mal blitzschnell hintereinander.
        # Einer davon WIRD funktionieren.

        for attempt in range(5):
            try:
                # Daten anfordern
                self.bus.write_i2c_block_data(self.addr, 0x03, [0x00])
                time.sleep(0.03) # Kurz warten

                # Lesen
                raw = self.bus.read_i2c_block_data(self.addr, 0, 18)

                # Schnell-Check: Ist das Byte 0 FF? -> Müll
                if raw[0] == 0xFF: continue

                # CRC Check
                if not self._check_crc_received(raw[0:3]) or \
                        not self._check_crc_received(raw[6:9]):
                    continue # Kaputt? Sofort nächster Versuch!

                # Daten Parsen
                def parse(b):
                    return struct.unpack('>f', bytes([b[0], b[1], b[3], b[4]]))[0]

                co2 = parse(raw[0:6])
                temp = parse(raw[6:12])
                hum = parse(raw[12:18])

                # Plausibilität
                if co2 < 1.0 or co2 > 40000.0: continue

                # TREFFER!
                self.last_co2 = co2
                self.last_temp = temp
                self.last_hum = hum
                return co2, temp, hum

            except Exception:
                time.sleep(0.02)
                continue # Bus Fehler? Sofort nochmal!

        # Wenn es 5x nicht geklappt hat, geben wir die alten Werte
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
            pass

        # SCD Setup
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

        # Kurze Pause für den Bus
        time.sleep(0.1)

        # SCD
        c, t, h = self.scd.read_measurement()
        if c is not None:
            result["scd_c"] = int(c)
            result["scd_t"] = round(t, 2)
            result["scd_h"] = round(h, 2)

        return result