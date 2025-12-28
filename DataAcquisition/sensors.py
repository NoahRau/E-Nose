import time
import struct
import sys
import smbus

# --- 1. Import BME680 ---
try:
    import bme680
except ImportError:
    print("FEHLER: 'bme680' fehlt. (pip install bme680)")
    sys.exit(1)

# --- 2. SCD30 Native Driver ---
class SCD30_Native:
    def __init__(self, bus_id=1, address=0x61):
        self.bus = smbus.SMBus(bus_id)
        self.addr = address
        self.is_connected = False

        try:
            # Check: Firmware Version lesen
            self._write_command(0xD100)
            self.is_connected = True
            print("   ✅ SCD30 erkannt.")

            # 1. Reset
            print("      -> Sende Soft-Reset...")
            self._write_command(0xD304)
            time.sleep(3.0) # WICHTIG: SCD30 braucht lange zum Neustart!

            # 2. Messintervall setzen (2 Sekunden)
            self._write_command(0x4600, [0x00, 0x02])
            time.sleep(0.2)

            # 3. Start Messung (mit Druckkompensation 0 mBar)
            print("      -> Starte Messung...")
            self._write_command(0x0010, [0x00, 0x00])

        except OSError:
            print("   ❌ SCD30 nicht gefunden (Adresse 0x61).")
            self.is_connected = False

    def _write_command(self, cmd, args=None):
        data = [(cmd >> 8) & 0xFF, cmd & 0xFF]
        if args:
            data.extend(args)
            data.append(0x81) # Dummy CRC, SCD30 ignoriert das oft bei Write
        try:
            self.bus.write_i2c_block_data(self.addr, data[0], data[1:])
            time.sleep(0.1)
        except OSError:
            pass

    def data_ready(self):
        if not self.is_connected: return False
        try:
            self._write_command(0x0202)
            # Lese 3 Bytes: MSB, LSB, CRC
            block = self.bus.read_i2c_block_data(self.addr, 0, 3)
            # Check if Data Ready bit (Byte 1, letztes Bit) is 1
            return block[1] == 1
        except OSError:
            return False

    def read_measurement(self):
        if not self.is_connected: return None, None, None
        try:
            self._write_command(0x0300)
            time.sleep(0.05)
            # Lese 18 Bytes
            raw = self.bus.read_i2c_block_data(self.addr, 0, 18)

            def parse_float(b):
                # Bytes in Big-Endian Float umwandeln (CRC überspringen)
                return struct.unpack('>f', bytes([b[0], b[1], b[3], b[4]]))[0]

            co2 = parse_float(raw[0:6])
            temp = parse_float(raw[6:12])
            hum = parse_float(raw[12:18])
            return co2, temp, hum
        except OSError:
            return None, None, None

# --- 3. Sensor Manager ---
class SensorManager:
    def __init__(self):
        self.bme = None
        self.scd = None

        print("\n[System] Initialisiere Sensoren...")
        self._init_bme688_robust()
        self._init_scd30()

    def _init_bme688_robust(self):
        """Versucht mehrfach, den BME zu initialisieren"""
        attempts = 0
        max_attempts = 5

        while attempts < max_attempts:
            try:
                # 1. Adresse finden
                try:
                    self.bme = bme680.BME680(bme680.I2C_ADDR_SECONDARY)
                except IOError:
                    self.bme = bme680.BME680(bme680.I2C_ADDR_PRIMARY)

                # 2. Einstellungen schreiben
                self.bme.set_humidity_oversample(bme680.OS_2X)
                self.bme.set_pressure_oversample(bme680.OS_4X)
                self.bme.set_temperature_oversample(bme680.OS_8X)
                self.bme.set_filter(bme680.FILTER_SIZE_3)

                # 3. Gas-Heizung setzen (Hier crashte es vorher!)
                # Wir machen das in einem try-block, um Kalibrierungsfehler abzufangen
                self.bme.set_gas_status(bme680.ENABLE_GAS_MEAS)
                self.bme.set_gas_heater_temperature(320)
                self.bme.set_gas_heater_duration(150)
                self.bme.select_gas_heater_profile(0)

                print(f"   ✅ BME688 verbunden (Versuch {attempts+1}).")
                return # Erfolg!

            except Exception as e:
                attempts += 1
                print(f"   ⚠️ BME Init Fehler (Versuch {attempts}): {e}")
                time.sleep(1.0)

        print("   ❌ BME688 konnte nicht initialisiert werden.")
        self.bme = None

    def _init_scd30(self):
        self.scd = SCD30_Native()

    def get_data(self):
        data = {
            "bme_temp": None, "bme_hum": None, "bme_press": None, "bme_gas": None,
            "scd_co2": None, "scd_temp": None, "scd_hum": None
        }

        # BME Lesen
        if self.bme:
            # get_sensor_data() triggered auch die Messung im Hintergrund
            if self.bme.get_sensor_data():
                data["bme_temp"] = round(self.bme.data.temperature, 2)
                data["bme_hum"] = round(self.bme.data.humidity, 2)
                data["bme_press"] = round(self.bme.data.pressure, 2)

                if self.bme.data.heat_stable:
                    data["bme_gas"] = int(self.bme.data.gas_resistance)
            else:
                # Manchmal hilft ein Trigger, wenn er schläft
                pass

                # SCD Lesen
        if self.scd:
            if self.scd.data_ready():
                c, t, h = self.scd.read_measurement()
                if c is not None:
                    data["scd_co2"] = int(c)
                    data["scd_temp"] = round(t, 2)
                    data["scd_hum"] = round(h, 2)

        return data