import time
import struct
import sys
import smbus

# BME Import
try:
    import bme680
except ImportError:
    pass

class SCD30_Native:
    def __init__(self, bus_id=1, address=0x61):
        self.bus = smbus.SMBus(bus_id)
        self.addr = address
        self.is_connected = False

        try:
            # Verbindungstest
            self._write_command(0xD100)
            self.is_connected = True
            print("   ✅ SCD30 erkannt.")

            # Intervall 2s
            self._write_command(0x4600, [0x00, 0x02])
            time.sleep(0.1)
            # Start Messung
            self._write_command(0x0010, [0x00, 0x00])

        except OSError:
            print("   ❌ SCD30 nicht gefunden.")
            self.is_connected = False

    def _write_command(self, cmd, args=None):
        data = [(cmd >> 8) & 0xFF, cmd & 0xFF]
        if args:
            data.extend(args)
            data.append(0x81)
        try:
            self.bus.write_i2c_block_data(self.addr, data[0], data[1:])
            time.sleep(0.05)
        except OSError:
            pass

    def check_status(self):
        """Debug Funktion: Was sagt das Ready-Register?"""
        if not self.is_connected: return "Disconn"
        try:
            self._write_command(0x0202)
            block = self.bus.read_i2c_block_data(self.addr, 0, 3)
            # Gib den rohen Wert zurück (1 = Ready, 0 = Waiting)
            return block[1]
        except OSError:
            return "Err"

    def read_measurement(self):
        if not self.is_connected: return None, None, None
        try:
            self._write_command(0x0300)
            time.sleep(0.02)
            raw = self.bus.read_i2c_block_data(self.addr, 0, 18)

            def parse_float(b):
                return struct.unpack('>f', bytes([b[0], b[1], b[3], b[4]]))[0]

            co2 = parse_float(raw[0:6])
            temp = parse_float(raw[6:12])
            hum = parse_float(raw[12:18])
            return co2, temp, hum
        except OSError:
            return None, None, None

class SensorManager:
    def __init__(self):
        self.bme = None
        self.scd = None
        print("\n[System] Init Sensoren (Debug Mode)...")
        self._init_scd30()
        self._init_bme688()

    def _init_bme688(self):
        try:
            try:
                self.bme = bme680.BME680(bme680.I2C_ADDR_SECONDARY)
            except IOError:
                self.bme = bme680.BME680(bme680.I2C_ADDR_PRIMARY)

            # Standard Config
            self.bme.set_humidity_oversample(bme680.OS_2X)
            self.bme.set_pressure_oversample(bme680.OS_4X)
            self.bme.set_temperature_oversample(bme680.OS_8X)
            self.bme.set_filter(bme680.FILTER_SIZE_3)
            # Gas AUS, um Fehlerquellen zu minimieren
            self.bme.set_gas_status(bme680.DISABLE_GAS_MEAS)

            print(f"   ✅ BME688 verbunden.")
        except Exception as e:
            print(f"   ⚠️ BME Fehler: {e}")

    def _init_scd30(self):
        self.scd = SCD30_Native()

    def get_data(self):
        data = { "bme_raw": "N/A", "scd_status": "N/A", "scd_co2": None }

        # --- BME Debug ---
        if self.bme:
            if self.bme.get_sensor_data():
                # Wir geben den rohen Text zurück, egal wie falsch er aussieht
                t = self.bme.data.temperature
                h = self.bme.data.humidity
                p = self.bme.data.pressure
                data["bme_raw"] = f"T:{t:.1f} H:{h:.1f} P:{p:.0f}"
            else:
                data["bme_raw"] = "Keine neuen Daten"

        # --- SCD Debug ---
        if self.scd:
            # Wir lesen den Status aus
            status = self.scd.check_status()
            data["scd_status"] = f"ReadyBit: {status}"

            # Wenn ReadyBit 1 ist, lesen wir
            if status == 1:
                c, t, h = self.scd.read_measurement()
                if c is not None:
                    data["scd_co2"] = f"{int(c)}ppm T:{t:.1f}"

        return data