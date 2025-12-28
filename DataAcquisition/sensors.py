import time
import struct
import sys
import smbus

class SCD30_Native:
    def __init__(self, bus_id=1, address=0x61):
        self.bus = smbus.SMBus(bus_id)
        self.addr = address
        self.connected = False

        try:
            # 1. Reset/Stop zur Sicherheit
            self._write(0xD304)
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
            data.append(0x81)
        try:
            self.bus.write_i2c_block_data(self.addr, data[0], data[1:])
            time.sleep(0.05)
        except OSError:
            pass

    def read_measurement(self):
        if not self.connected: return None, None, None
        try:
            # Check Ready Status
            self._write(0x0202)
            ready = self.bus.read_i2c_block_data(self.addr, 0, 3)

            # Wenn nicht ready, geben wir None zurück (kein Blockieren)
            if ready[1] != 1:
                return None, None, None

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
        self.bme_addr = 0x77
        self.scd = SCD30_Native()

        # BME Adresse prüfen
        try:
            self.bus.read_byte_data(0x77, 0xD0)
            self.bme_addr = 0x77
        except:
            self.bme_addr = 0x76

    def diagnose_register(self):
        results = []
        # Register Diagnose BME
        results.append(f"BME Addr: {hex(self.bme_addr)}")
        try:
            # Schreiben nach Vorschrift: Erst Hum, dann Meas
            # 1. Humidity Control (0x72) -> x1 Oversampling (0x01)
            self.bus.write_byte_data(self.bme_addr, 0x72, 0x01)

            # 2. Meas Control (0x74) -> Forced Mode (0x25)
            self.bus.write_byte_data(self.bme_addr, 0x74, 0x25)

            time.sleep(0.2)

            # Status lesen (0x1D) - Bit 7 muss 1 sein wenn neue Daten da sind
            status = self.bus.read_byte_data(self.bme_addr, 0x1D)
            new_data = (status & 0x80) != 0
            measuring = (status & 0x20) != 0

            results.append(f"BME Status Reg (0x1D): {bin(status)}")
            if new_data:
                results.append("✅ BME: Neue Daten verfügbar!")
            else:
                results.append(f"❌ BME: Keine Daten (Measuring: {measuring})")

        except Exception as e:
            results.append(f"❌ Bus Fehler: {e}")

        return "\n".join(results)

    def get_data(self):
        data = { "bme_raw": "Init", "scd_co2": "Wait" }

        # --- BME SEQUENZ ---
        try:
            # WICHTIG: Die Reihenfolge ist entscheidend!
            # 1. ctrl_hum (0x72) setzen (x1 Oversampling = 0x01)
            self.bus.write_byte_data(self.bme_addr, 0x72, 0x01)

            # 2. ctrl_meas (0x74) setzen (x1 Temp, x1 Press, FORCED Mode = 0x25)
            self.bus.write_byte_data(self.bme_addr, 0x74, 0x25)

            # Warten (Messung dauert ca 10ms, wir geben ihm 100ms)
            time.sleep(0.1)

            # 3. Status prüfen (0x1D)
            status = self.bus.read_byte_data(self.bme_addr, 0x1D)

            # Bit 7 (0x80) ist "New Data"
            if status & 0x80:
                # Daten lesen (Temp MSB 0x22)
                msb = self.bus.read_byte_data(self.bme_addr, 0x22)
                lsb = self.bus.read_byte_data(self.bme_addr, 0x23)
                xlsb = self.bus.read_byte_data(self.bme_addr, 0x24)

                temp_raw = ((msb << 12) | (lsb << 4) | (xlsb >> 4))

                # Hum (0x25, 0x26)
                msb_h = self.bus.read_byte_data(self.bme_addr, 0x25)
                lsb_h = self.bus.read_byte_data(self.bme_addr, 0x26)
                hum_raw = (msb_h << 8) | lsb_h

                if temp_raw == 0x80000:
                    data["bme_raw"] = "ZOMBIE (ADC Inaktiv)"
                else:
                    # Rohe Werte anzeigen -> Wenn die schwanken, ist alles gut!
                    data["bme_raw"] = f"OK (T_Raw:{temp_raw} H_Raw:{hum_raw})"
            else:
                data["bme_raw"] = "Keine neuen Daten (Bit 7 low)"

        except:
            data["bme_raw"] = "Bus Fehler"

        # --- SCD ---
        c, t, h = self.scd.read_measurement()
        if c is not None:
            data["scd_co2"] = f"{int(c)}ppm"

        return data