import time
import struct
import sys
import smbus

# Pimoroni BME680 Library
try:
    import bme680
except ImportError:
    pass

class SCD30_Native:
    def __init__(self, bus_id=3, address=0x61):
        self.bus_id = bus_id
        self.addr = address
        self.connected = False
        self.bus = None

        try:
            self.bus = smbus.SMBus(bus_id)
            # Test: Firmware Version lesen (als Ping)
            self.bus.write_i2c_block_data(self.addr, 0xD1, [0x00])
            self.connected = True
            print(f"   [SCD30] Erkannt auf Bus {bus_id}.")

            # --- RESET & START ---
            self._write(0xD304) # Soft Reset
            time.sleep(2.0)     # Warten auf Reboot

            self._write(0x4600, [0x00, 0x02]) # 2s Intervall
            time.sleep(0.1)
            self._write(0x0010, [0x00, 0x00]) # Start Messung
            print("   [SCD30] Messung gestartet.")

        except Exception as e:
            print(f"   [SCD30] Fehler Init: {e}")
            self.connected = False

    def _write(self, cmd, args=None):
        if not self.connected: return
        data = [(cmd >> 8) & 0xFF, cmd & 0xFF]
        if args:
            data.extend(args)
            data.append(0x81) # CRC Dummy
        try:
            self.bus.write_i2c_block_data(self.addr, data[0], data[1:])
            time.sleep(0.05)
        except OSError:
            pass

    def read_measurement(self):
        """
        Liest Daten sofort (Brute Force), ohne auf Ready-Status zu warten.
        Da der Sensor blinkt, wissen wir, dass Daten da sind.
        """
        if not self.connected: return None, None, None
        try:
            # 1. Befehl: Lies Messwerte (0x0300)
            self._write(0x0300)
            time.sleep(0.02) # Kurz warten

            # 2. Lies 18 Bytes
            raw = self.bus.read_i2c_block_data(self.addr, 0, 18)

            def parse(b):
                # Konvertierung Bytes -> Float
                return struct.unpack('>f', bytes([b[0], b[1], b[3], b[4]]))[0]

            co2 = parse(raw[0:6])
            temp = parse(raw[6:12])
            hum = parse(raw[12:18])

            # 3. Sanity Check: Filtere Nullen oder Datenmüll raus
            if co2 < 10.0 or co2 > 40000.0:
                return None, None, None

            return co2, temp, hum

        except Exception:
            # Falls I2C mal kurz hakt, einfach ignorieren
            return None, None, None

class SensorManager:
    def __init__(self):
        self.bme = None
        self.scd = None
        self.status = {"bme": False, "scd": False}

        # --- BME688 (Bus 1) ---
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
            self.status["bme"] = True
        except Exception:
            self.status["bme"] = False

        # --- SCD30 (Bus 3) ---
        self.scd = SCD30_Native(bus_id=3)
        self.status["scd"] = self.scd.connected

    def get_formatted_data(self):
        result = { "bme_t": None, "bme_h": None, "bme_g": None,
                   "scd_c": None, "scd_t": None }

        # BME
        if self.bme and self.bme.get_sensor_data():
            result["bme_t"] = round(self.bme.data.temperature, 2)
            result["bme_h"] = round(self.bme.data.humidity, 2)
            if self.bme.data.heat_stable:
                result["bme_g"] = int(self.bme.data.gas_resistance)

        # SCD
        c, t, h = self.scd.read_measurement()
        if c is not None:
            result["scd_c"] = int(c)
            result["scd_t"] = round(t, 2)

        return result