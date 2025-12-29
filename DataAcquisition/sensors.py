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

            # --- SETUP ---
            # 1. Reset
            self._write(0xD304)
            time.sleep(2.0) # Datasheet: Boot-up time < 2s

            # 2. Set Interval to 2s
            # Datasheet: 0x4600 + 0x0002 + CRC(0xE3)
            self._write(0x4600, [0x00, 0x02])
            time.sleep(0.1)

            # 3. Start Measurement (0 mBar compensation)
            # Datasheet: 0x0010 + 0x0000 + CRC(0x81)
            self._write(0x0010, [0x00, 0x00])
            print(f"   [SCD30] Verbunden auf Bus {bus_id} (CRC Fixed).")

        except Exception as e:
            print(f"   [SCD30] Fehler: {e}")
            self.connected = False

    def _calc_crc(self, data):
        """Calculates CRC8 according to Sensirion datasheet (Poly 0x31)"""
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
        if not self.connected: return
        data = [(cmd >> 8) & 0xFF, cmd & 0xFF]

        if args:
            data.extend(args)
            # --- FIX: Calculate CRC dynamically instead of hardcoded 0x81 ---
            crc = self._calc_crc(args)
            data.append(crc)

        try:
            self.bus.write_i2c_block_data(self.addr, data[0], data[1:])
            # Datasheet 1.4.4: "header should be sent with a delay of > 3ms"
            time.sleep(0.02)
        except OSError:
            pass

    def _check_crc(self, data):
        """Validates received data CRC"""
        # data format: [Byte1, Byte2, CRC]
        return self._calc_crc(data[:2]) == data[2]

    def read_measurement(self):
        if not self.connected: return self.last_co2, self.last_temp, self.last_hum

        try:
            # Direct Read (Blind Mode)
            self._write(0x0300)
            time.sleep(0.05)

            raw = self.bus.read_i2c_block_data(self.addr, 0, 18)

            # 1. Empty Check
            if raw[0] == 0xFF:
                return self.last_co2, self.last_temp, self.last_hum

            # 2. CRC Check (Critical for valid data)
            if not self._check_crc(raw[0:3]) or \
                    not self._check_crc(raw[6:9]) or \
                    not self._check_crc(raw[12:15]):
                return self.last_co2, self.last_temp, self.last_hum

            def parse(b):
                return struct.unpack('>f', bytes([b[0], b[1], b[3], b[4]]))[0]

            co2 = parse(raw[0:6])
            temp = parse(raw[6:12])
            hum = parse(raw[12:18])

            if co2 < 1.0 or co2 > 40000.0:
                return self.last_co2, self.last_temp, self.last_hum

            self.last_co2 = co2
            self.last_temp = temp
            self.last_hum = hum
            return co2, temp, hum

        except Exception:
            return self.last_co2, self.last_temp, self.last_hum

class SensorManager:
    def __init__(self):
        self.bme = None
        self.scd = None

        # BME Setup
        try:
            self.bme = bme680.BME680(bme680.I2C_ADDR_SECONDARY)
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
            print("   [BME688] Fehler.")

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

        time.sleep(0.1)

        # SCD
        c, t, h = self.scd.read_measurement()
        if c is not None:
            result["scd_c"] = int(c)
            result["scd_t"] = round(t, 2)
            result["scd_h"] = round(h, 2)

        return result