import time
import struct
import sys
import smbus

# --- SCD30 TREIBER (Jetzt auf Bus 3!) ---
class SCD30_Native:
    def __init__(self, bus_id=3, address=0x61): # ACHTUNG: Default ist jetzt Bus 3
        try:
            self.bus = smbus.SMBus(bus_id)
        except Exception as e:
            print(f"   ❌ Fehler: Konnte Bus {bus_id} nicht öffnen: {e}")
            self.connected = False
            return

        self.addr = address
        self.connected = False

        try:
            # 1. Stop
            self._write(0xD304)
            time.sleep(0.1)
            # 2. Intervall 2s
            self._write(0x4600, [0x00, 0x02])
            # 3. Start
            self._write(0x0010, [0x00, 0x00])
            self.connected = True
            print(f"   ✅ SCD30 auf Bus {bus_id} verbunden.")
        except:
            print(f"   ❌ SCD30 auf Bus {bus_id} nicht gefunden.")
            self.connected = False

    def _write(self, cmd, args=None):
        data = [(cmd >> 8) & 0xFF, cmd & 0xFF]
        if args:
            data.extend(args)
            data.append(0x81)
        try:
            self.bus.write_i2c_block_data(self.addr, data[0], data[1:])
            time.sleep(0.05)
        except OSError:
            pass

    def read_measurement(self):
        if not self.connected: return None, None, None
        try:
            # Ready Check
            self._write(0x0202)
            ready = self.bus.read_i2c_block_data(self.addr, 0, 3)
            if ready[1] != 1: return None, None, None

            # Read
            self._write(0x0300)
            time.sleep(0.02)
            raw = self.bus.read_i2c_block_data(self.addr, 0, 18)

            def parse(b):
                return struct.unpack('>f', bytes([b[0], b[1], b[3], b[4]]))[0]

            return parse(raw[0:6]), parse(raw[6:12]), parse(raw[12:18])
        except:
            return None, None, None

# --- SENSOR MANAGER ---
class SensorManager:
    def __init__(self):
        print("\n[System] Starte Dual-Bus System...")

        # BME auf Bus 1
        try:
            self.bus1 = smbus.SMBus(1)
            self.bme_addr = 0x77
            # Kurzer Check ob er da ist
            self.bus1.read_byte_data(0x77, 0xD0)
            print("   ✅ BME688 auf Bus 1 gefunden.")
        except:
            try:
                self.bme_addr = 0x76
                self.bus1.read_byte_data(0x76, 0xD0)
                print("   ✅ BME688 auf Bus 1 gefunden (Alt).")
            except:
                print("   ❌ BME688 Fehler.")
                self.bus1 = None

        # SCD auf Bus 3
        self.scd = SCD30_Native(bus_id=3)

    def get_data(self):
        data = { "bme_temp": None, "bme_hum": None, "scd_co2": None }

        # --- BME LESEN (Bus 1) ---
        if self.bus1:
            try:
                # 1. Config (Hum x1)
                self.bus1.write_byte_data(self.bme_addr, 0x72, 0x01)
                # 2. Trigger Messung (Forced)
                self.bus1.write_byte_data(self.bme_addr, 0x74, 0x25)

                time.sleep(0.1)

                # 3. Lesen (Temp & Hum)
                # Temp MSB (0x22)
                d = self.bus1.read_i2c_block_data(self.bme_addr, 0x22, 3)
                raw_t = (d[0] << 12) | (d[1] << 4) | (d[2] >> 4)

                # Hum MSB (0x25)
                d_h = self.bus1.read_i2c_block_data(self.bme_addr, 0x25, 2)
                raw_h = (d_h[0] << 8) | d_h[1]

                # Grobe Umrechnung für Anzeige (nicht wissenschaftlich exakt ohne Kalibrierung)
                # Aber wir sehen, ob es lebt!
                if raw_t != 0x80000:
                    # Calibration ignorieren wir hier für den Roh-Test, wir wollen sehen ob es schwankt
                    data["bme_temp"] = raw_t
                    data["bme_hum"] = raw_h
            except:
                pass

        # --- SCD LESEN (Bus 3) ---
        c, t, h = self.scd.read_measurement()
        if c is not None:
            data["scd_co2"] = int(c)

        return data