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
            # Init ohne Reset, nur Start
            self._write_command(0x4600, [0x00, 0x02]) # 2s Intervall
            time.sleep(0.1)
            self._write_command(0x0010, [0x00, 0x00]) # Start Messung
            self.is_connected = True
            print("   ✅ SCD30 initiiert.")
        except OSError:
            print("   ❌ SCD30 Fehler.")

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

    def read_measurement_blind(self):
        """Liest Daten OHNE Ready-Check (Brute Force)"""
        if not self.is_connected: return None, None, None
        try:
            self._write_command(0x0300)
            time.sleep(0.02)
            # Wir lesen 18 Bytes
            raw = self.bus.read_i2c_block_data(self.addr, 0, 18)

            def parse_float(b):
                # Checksummen-Prüfung ignorieren wir erstmal, wir wollen IRGENDWAS sehen
                return struct.unpack('>f', bytes([b[0], b[1], b[3], b[4]]))[0]

            co2 = parse_float(raw[0:6])
            temp = parse_float(raw[6:12])
            hum = parse_float(raw[12:18])

            # Filter: Wenn CO2 0 ist oder unrealistisch hoch, war es Lesefehler
            if co2 < 100 or co2 > 40000: return None, None, None

            return co2, temp, hum
        except Exception:
            return None, None, None

class SensorManager:
    def __init__(self):
        self.bme = None
        self.scd = None
        print("\n[System] Init Sensoren (Aggressive Mode)...")
        self._init_scd30()
        self._init_bme688()

    def _init_bme688(self):
        try:
            try:
                self.bme = bme680.BME680(bme680.I2C_ADDR_SECONDARY)
            except IOError:
                self.bme = bme680.BME680(bme680.I2C_ADDR_PRIMARY)

            # WICHTIG: Setze Oversampling
            self.bme.set_humidity_oversample(bme680.OS_2X)
            self.bme.set_pressure_oversample(bme680.OS_4X)
            self.bme.set_temperature_oversample(bme680.OS_8X)
            self.bme.set_filter(bme680.FILTER_SIZE_3)
            self.bme.set_gas_status(bme680.DISABLE_GAS_MEAS)
            print(f"   ✅ BME688 verbunden.")
        except Exception:
            pass

    def _init_scd30(self):
        self.scd = SCD30_Native()

    def get_data(self):
        data = { "bme_raw": "Wait", "scd_co2": "Wait" }

        # --- BME FORCE ---
        if self.bme:
            try:
                # TRITT IN DEN HINTERN: Forced Mode aktivieren
                # Das zwingt den Sensor aufzuwachen
                self.bme.set_power_mode(bme680.FORCED_MODE)
                time.sleep(0.2) # Warten bis Messung fertig

                if self.bme.get_sensor_data():
                    t = self.bme.data.temperature
                    h = self.bme.data.humidity
                    # Check ob immer noch Zombie-Werte
                    if h == 100.00 and t > 30:
                        data["bme_raw"] = f"ZOMBIE (T:{t:.1f} H:{h:.0f})"
                    else:
                        data["bme_raw"] = f"T:{t:.2f} H:{h:.2f}"
            except Exception as e:
                data["bme_raw"] = "Err"

        # --- SCD BLIND ---
        if self.scd:
            # Wir fragen nicht ob er bereit ist, wir nehmen was da ist
            c, t, h = self.scd.read_measurement_blind()
            if c is not None:
                data["scd_co2"] = f"{int(c)}ppm"
            else:
                data["scd_co2"] = "Lesefehler"

        return data