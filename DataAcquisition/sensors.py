import time
import struct
import sys
import smbus
import math

# BME680 Library Check
try:
    import bme680
except ImportError:
    pass

class SCD30_Native:
    def __init__(self, bus_id=1, address=0x61): # WIEDER AUF BUS 1
        self.bus_id = bus_id
        self.addr = address
        self.connected = False
        self.bus = None

        try:
            self.bus = smbus.SMBus(bus_id)
            # Test-Schreiben
            try:
                self.bus.write_i2c_block_data(self.addr, 0xD1, [0x00])
            except:
                pass
            self.connected = True

            # Setup
            self._write(0xD304) # Reset
            time.sleep(2.0)
            self._write(0x4600, [0x00, 0x02]) # 2s Intervall
            time.sleep(0.1)
            self._write(0x0010, [0x00, 0x00]) # Start
            print(f"   [SCD30] Verbunden auf Bus {bus_id}.")

        except Exception as e:
            print(f"   [SCD30] Fehler: {e}")
            self.connected = False

    def _write(self, cmd, args=None):
        if not self.connected: return
        data = [(cmd >> 8) & 0xFF, cmd & 0xFF]
        if args:
            data.extend(args)
            data.append(0x81)
        try:
            self.bus.write_i2c_block_data(self.addr, data[0], data[1:])
            time.sleep(0.02)
        except OSError:
            pass

    def _check_crc(self, data):
        """Prüfsummen-Check gegen Lesefehler"""
        crc = 0xFF
        for b in data[:2]:
            crc ^= b
            for _ in range(8):
                if crc & 0x80: crc = (crc << 1) ^ 0x31
                else: crc = (crc << 1)
        return (crc & 0xFF) == data[2]

    def read_measurement(self):
        if not self.connected: return None, None, None

        # Wir versuchen es 5x, falls der BME gerade den Bus blockiert hat
        for _ in range(5):
            try:
                self._write(0x0300)
                time.sleep(0.05) # Warten

                raw = self.bus.read_i2c_block_data(self.addr, 0, 18)

                # Prüfen
                if raw[0] == 0xFF: continue
                if not self._check_crc(raw[0:3]): continue
                if not self._check_crc(raw[6:9]): continue

                def parse(b):
                    return struct.unpack('>f', bytes([b[0], b[1], b[3], b[4]]))[0]

                co2 = parse(raw[0:6])
                temp = parse(raw[6:12])
                hum = parse(raw[12:18])

                if co2 < 1.0 or co2 > 40000.0: continue

                return co2, temp, hum

            except OSError:
                time.sleep(0.1)
                continue

        return None, None, None

class SensorManager:
    def __init__(self):
        self.bme = None
        self.scd = None

        # --- BME688 (Bus 1) ---
        try:
            try:
                # 0x77 ist Standard Secondary
                self.bme = bme680.BME680(bme680.I2C_ADDR_SECONDARY)
            except IOError:
                # 0x76 ist Primary
                self.bme = bme680.BME680(bme680.I2C_ADDR_PRIMARY)

            self.bme.set_humidity_oversample(bme680.OS_2X)
            self.bme.set_pressure_oversample(bme680.OS_4X)
            self.bme.set_temperature_oversample(bme680.OS_8X)
            self.bme.set_filter(bme680.FILTER_SIZE_3)
            self.bme.set_gas_status(bme680.ENABLE_GAS_MEAS)
            self.bme.set_gas_heater_temperature(320)
            self.bme.set_gas_heater_duration(150)
            self.bme.select_gas_heater_profile(0)
            print("   [BME688] Verbunden auf Bus 1.")
        except Exception as e:
            print(f"   [BME688] Fehler: {e}")

        # --- SCD30 (AUCH Bus 1) ---
        # Wichtig: Bus ID ist hier jetzt 1!
        self.scd = SCD30_Native(bus_id=1)

    def get_formatted_data(self):
        result = { "bme_t": None, "bme_h": None, "bme_g": None,
                   "scd_c": None, "scd_t": None, "scd_h": None }

        # 1. BME auslesen
        if self.bme:
            try:
                if self.bme.get_sensor_data():
                    result["bme_t"] = round(self.bme.data.temperature, 2)
                    result["bme_h"] = round(self.bme.data.humidity, 2)
                    if self.bme.data.heat_stable:
                        result["bme_g"] = int(self.bme.data.gas_resistance)
            except:
                pass # I2C Kollision ignorieren

        # 2. Kurze Pause für den Bus
        time.sleep(0.05)

        # 3. SCD auslesen
        c, t, h = self.scd.read_measurement()
        if c is not None:
            result["scd_c"] = int(c)
            result["scd_t"] = round(t, 2)
            result["scd_h"] = round(h, 2)

        return result