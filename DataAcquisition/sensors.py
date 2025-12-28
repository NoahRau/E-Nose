import time
import struct
import sys
import smbus

# Wir nutzen die offizielle Pimoroni Lib für BME, weil sie jetzt sicher läuft
try:
    import bme680
except ImportError:
    print("Fehler: Bitte 'pip install bme680' ausführen!")
    sys.exit(1)

class SCD30_Native:
    """
    Treiber für SCD30 auf einem dedizierten Bus (Standard Bus 3).
    """
    def __init__(self, bus_id=3, address=0x61):
        self.bus_id = bus_id
        self.addr = address
        self.connected = False
        self.bus = None

        try:
            self.bus = smbus.SMBus(bus_id)
            # Test-Lesen (Firmware Version)
            self.bus.write_i2c_block_data(self.addr, 0xD1, [0x00])
            self.connected = True

            # Reset und Startsequenz
            self._write(0xD304) # Soft Reset
            time.sleep(0.1)
            self._write(0x4600, [0x00, 0x02]) # 2s Intervall
            self._write(0x0010, [0x00, 0x00]) # Start Messung

        except Exception as e:
            self.connected = False

    def _write(self, cmd, args=None):
        if not self.connected: return
        data = [(cmd >> 8) & 0xFF, cmd & 0xFF]
        if args:
            data.extend(args)
            data.append(0x81) # CRC Dummy
        try:
            self.bus.write_i2c_block_data(self.addr, data[0], data[1:])
            time.sleep(0.02)
        except OSError:
            pass

    def read_measurement(self):
        if not self.connected: return None, None, None
        try:
            # Check Ready Status
            self._write(0x0202)
            ready = self.bus.read_i2c_block_data(self.addr, 0, 3)
            if ready[1] != 1: return None, None, None

            # Read Data
            self._write(0x0300)
            time.sleep(0.01)
            raw = self.bus.read_i2c_block_data(self.addr, 0, 18)

            def parse(b):
                return struct.unpack('>f', bytes([b[0], b[1], b[3], b[4]]))[0]

            return parse(raw[0:6]), parse(raw[6:12]), parse(raw[12:18])
        except:
            return None, None, None

class SensorManager:
    def __init__(self):
        self.bme = None
        self.scd = None
        self.status = {"bme": False, "scd": False}

        # --- INIT BME688 (Bus 1) ---
        # Die Library nutzt automatisch Bus 1
        try:
            try:
                self.bme = bme680.BME680(bme680.I2C_ADDR_SECONDARY) # 0x77
            except IOError:
                self.bme = bme680.BME680(bme680.I2C_ADDR_PRIMARY)   # 0x76

            # Setup für schöne, geglättete Werte
            self.bme.set_humidity_oversample(bme680.OS_2X)
            self.bme.set_pressure_oversample(bme680.OS_4X)
            self.bme.set_temperature_oversample(bme680.OS_8X)
            self.bme.set_filter(bme680.FILTER_SIZE_3)
            # Gas-Messung aktivieren
            self.bme.set_gas_status(bme680.ENABLE_GAS_MEAS)
            self.bme.set_gas_heater_temperature(320)
            self.bme.set_gas_heater_duration(150)
            self.bme.select_gas_heater_profile(0)

            self.status["bme"] = True
        except Exception:
            self.status["bme"] = False

        # --- INIT SCD30 (Bus 3) ---
        self.scd = SCD30_Native(bus_id=3)
        self.status["scd"] = self.scd.connected

    def get_formatted_data(self):
        """Holt Daten und gibt ein schönes Dictionary zurück"""
        result = {
            "bme_t": None, "bme_h": None, "bme_p": None, "bme_g": None,
            "scd_c": None, "scd_t": None, "scd_h": None
        }

        # BME Lesen
        if self.bme and self.bme.get_sensor_data():
            result["bme_t"] = round(self.bme.data.temperature, 2)
            result["bme_h"] = round(self.bme.data.humidity, 2)
            result["bme_p"] = round(self.bme.data.pressure, 1)
            if self.bme.data.heat_stable:
                result["bme_g"] = int(self.bme.data.gas_resistance)

        # SCD Lesen
        c, t, h = self.scd.read_measurement()
        if c is not None:
            result["scd_c"] = int(c)
            result["scd_t"] = round(t, 2)
            result["scd_h"] = round(h, 2)

        return result