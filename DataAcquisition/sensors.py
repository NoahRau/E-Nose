import time
import struct
import sys
import smbus
import math

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

        # Cache
        self.last_co2 = None
        self.last_temp = None
        self.last_hum = None

        try:
            self.bus = smbus.SMBus(bus_id)
            # Setup: Reset & Start
            try:
                # Intervall 2s (CRC 0xE3)
                self.bus.write_i2c_block_data(self.addr, 0x46, [0x00, 0x02, 0xE3])
                time.sleep(0.1)
                # Start (CRC 0x81)
                self.bus.write_i2c_block_data(self.addr, 0x00, [0x10, 0x00, 0x00, 0x81])
            except:
                pass
            self.connected = True
            print(f"   [SCD30] Verbunden (Strict Filter Mode).")

        except Exception as e:
            print(f"   [SCD30] Fehler: {e}")
            self.connected = False

    def _check_crc(self, data):
        """Berechnet CRC8"""
        crc = 0xFF
        for b in data:
            crc ^= b
            for _ in range(8):
                if crc & 0x80: crc = (crc << 1) ^ 0x31
                else: crc = (crc << 1)
        return crc & 0xFF

    def read_measurement(self):
        if not self.connected: return self.last_co2, self.last_temp, self.last_hum

        try:
            # 1. Daten anfordern
            self.bus.write_i2c_block_data(self.addr, 0x03, [0x00])
            time.sleep(0.05) # Wichtig für Software-I2C

            # 2. Wir lesen GENUG Bytes, um Verschiebungen zu finden
            raw = self.bus.read_i2c_block_data(self.addr, 0, 24)

            # 3. SUCHE NACH GÜLTIGEN DATEN (Sliding Window)
            # Wir schieben ein Fenster über die Daten und suchen etwas Plausibles.

            for i in range(7): # Teste Offset 0 bis 6
                if len(raw) < i+18: break
                frame = raw[i : i+18]

                # A. CRC Prüfen (Muss stimmen)
                if self._check_crc(frame[0:3]) != frame[2]: continue
                if self._check_crc(frame[6:9]) != frame[8]: continue
                if self._check_crc(frame[12:15]) != frame[14]: continue

                # B. Werte decodieren
                def parse(b):
                    val = struct.unpack('>f', bytes([b[0], b[1], b[3], b[4]]))[0]
                    return val

                co2 = parse(frame[0:6])
                temp = parse(frame[6:12])
                hum = parse(frame[12:18])

                # C. ABSTURZ-SCHUTZ (NaN Check)
                if math.isnan(co2) or math.isnan(temp) or math.isnan(hum):
                    continue

                # D. REALITÄTS-CHECK (Der Türsteher)
                # CO2 muss > 200 sein (26 ppm ist unmöglich -> das ist verschobene Temp)
                # Temp muss < 60 sein (46 Grad ist meistens verschobene Feuchte)
                if co2 < 200.0 or co2 > 40000.0: continue
                if temp < -10.0 or temp > 60.0: continue

                # Wenn wir hier sind, ist der Wert GOLD.
                self.last_co2 = co2
                self.last_temp = temp
                self.last_hum = hum
                return co2, temp, hum

            # Nichts gefunden? Alte Werte zurückgeben (verhindert Flackern)
            return self.last_co2, self.last_temp, self.last_hum

        except Exception:
            return self.last_co2, self.last_temp, self.last_hum

class SensorManager:
    def __init__(self):
        self.bme = None
        self.scd = None

        # BME Setup
        try:
            try:
                self.bme = bme680.BME680(bme680.I2C_ADDR_SECONDARY)
            except IOError:
                self.bme = bme680.BME680(bme680.I2C_ADDR_PRIMARY)

            # Standard BME Settings
            self.bme.set_humidity_oversample(bme680.OS_2X)
            self.bme.set_pressure_oversample(bme680.OS_4X)
            self.bme.set_temperature_oversample(bme680.OS_8X)
            self.bme.set_filter(bme680.FILTER_SIZE_3)
            self.bme.set_gas_status(bme680.ENABLE_GAS_MEAS)
            self.bme.set_gas_heater_temperature(320)
            self.bme.set_gas_heater_duration(150)
            self.bme.select_gas_heater_profile(0)
        except Exception:
            pass

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

        # Pause für den Bus
        time.sleep(0.1)

        # SCD
        c, t, h = self.scd.read_measurement()

        # WICHTIG: Hier prüfen wir nochmal auf NaN, damit hardwaretest.py nicht crasht
        if c is not None and not math.isnan(c):
            result["scd_c"] = int(c)
            result["scd_t"] = round(t, 2)
            result["scd_h"] = round(h, 2)

        return result