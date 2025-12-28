import time
import struct
import sys
import smbus

class SCD30_Native:
    def __init__(self, bus_id=1, address=0x61):
        self.bus = smbus.SMBus(bus_id)
        self.addr = address
        self.connected = False

        # Init Versuch
        try:
            # 1. Stop (falls er hängt)
            self._write(0x0104)
            time.sleep(0.1)
            # 2. Intervall 2s
            self._write(0x4600, [0x00, 0x02])
            # 3. Start
            self._write(0x0010, [0x00, 0x00])
            self.connected = True
        except:
            self.connected = False

    def _write(self, cmd, args=None):
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
        if not self.connected: return None, None, None
        try:
            # Data Ready Check
            self._write(0x0202)
            ready = self.bus.read_i2c_block_data(self.addr, 0, 3)
            if ready[1] != 1: return None, None, None # Noch nicht bereit

            # Daten lesen
            self._write(0x0300)
            time.sleep(0.02)
            raw = self.bus.read_i2c_block_data(self.addr, 0, 18)

            def parse(b):
                return struct.unpack('>f', bytes([b[0], b[1], b[3], b[4]]))[0]

            return parse(raw[0:6]), parse(raw[6:12]), parse(raw[12:18])
        except:
            return None, None, None

class SensorManager:
    def __init__(self):
        self.bus = smbus.SMBus(1)
        self.bme_addr = 0x77 # Standard Sekundär
        self.scd = SCD30_Native()

        # BME Adresse suchen
        try:
            self.bus.read_byte_data(0x77, 0xD0)
            self.bme_addr = 0x77
        except:
            self.bme_addr = 0x76

    def diagnose_register(self):
        """Prüft, ob wir Register schreiben können (Anti-Zombie Check)"""
        results = []

        # --- BME CHECK ---
        results.append(f"BME Addr: {hex(self.bme_addr)}")
        try:
            # Chip ID lesen (0xD0) sollte 0x61 sein
            cid = self.bus.read_byte_data(self.bme_addr, 0xD0)
            results.append(f"BME ID: {hex(cid)} ({'OK' if cid==0x61 else 'FAIL'})")

            # Schreib-Test: Wir versuchen, den Sleep-Mode zu lesen
            ctrl_meas = self.bus.read_byte_data(self.bme_addr, 0x74)
            results.append(f"BME Ctrl_Meas (vorher): {bin(ctrl_meas)}")

            # Wir schreiben FORCED MODE (0x25: Temp x1, Press x1, Forced)
            self.bus.write_byte_data(self.bme_addr, 0x74, 0x25)
            time.sleep(0.1)

            # Wir lesen zurück
            ctrl_meas_new = self.bus.read_byte_data(self.bme_addr, 0x74)
            results.append(f"BME Ctrl_Meas (nachher): {bin(ctrl_meas_new)}")

            if ctrl_meas_new != ctrl_meas or (ctrl_meas_new & 0b11) == 0:
                results.append("✅ BME Schreib-Test: ERFOLG (Wert geändert oder zurückgesetzt)")
            else:
                results.append("❌ BME Schreib-Test: FEHLGESCHLAGEN (Wert klemmt)")

        except Exception as e:
            results.append(f"❌ BME Bus Fehler: {e}")

        # --- SCD CHECK ---
        results.append(f"SCD Connected: {self.scd.connected}")

        return "\n".join(results)

    def get_data(self):
        data = { "bme_raw": "Init", "scd_co2": "Wait" }

        # --- BME LOGIK (MANUELL) ---
        try:
            # 1. Wake Up Call (Schreiben)
            # Register 0x74 auf 0x25 setzen (Forced Mode, x1 Oversampling)
            self.bus.write_byte_data(self.bme_addr, 0x74, 0x25)

            # Warten bis Messung fertig (Status Reg 0x1D Bit 5 prüfen, oder einfach warten)
            time.sleep(0.1)

            # 2. Daten lesen (Burst Read ab 0x1F)
            # Wir lesen Temp (0x22..0x24) und Hum (0x25..0x26)
            # Um es simpel zu halten, lesen wir byte-weise

            # Temp
            msb = self.bus.read_byte_data(self.bme_addr, 0x22)
            lsb = self.bus.read_byte_data(self.bme_addr, 0x23)
            xlsb = self.bus.read_byte_data(self.bme_addr, 0x24)
            temp_raw = ((msb << 12) | (lsb << 4) | (xlsb >> 4))

            # Hum
            msb_h = self.bus.read_byte_data(self.bme_addr, 0x25)
            lsb_h = self.bus.read_byte_data(self.bme_addr, 0x26)
            hum_raw = (msb_h << 8) | lsb_h

            # Rohdaten Analyse
            if temp_raw == 0x80000:
                data["bme_raw"] = "ZOMBIE (Keine Messung)"
            else:
                # Sehr grobe Umrechnung nur zur Kontrolle ob sich was bewegt
                # (Echte Umrechnung braucht Kalibrierungsdaten, das ist hier zu viel Code)
                # Wir zeigen einfach den RAW wert, wenn der sich ändert, lebt er!
                data["bme_raw"] = f"RAW: T={temp_raw} H={hum_raw}"

        except Exception as e:
            data["bme_raw"] = "I/O Error"

        # --- SCD LOGIK ---
        c, t, h = self.scd.read_measurement()
        if c is not None:
            data["scd_co2"] = f"{int(c)}ppm"

        return data