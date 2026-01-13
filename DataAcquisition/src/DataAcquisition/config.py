# config.py
import os

# --- InfluxDB Einstellungen ---
# Token wird aus Umgebungsvariable gelesen (gesetzt via .bashrc nach Ansible-Setup)
INFLUX_URL = os.environ.get("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN = os.environ.get("INFLUX_TOKEN", "")
INFLUX_ORG = os.environ.get("INFLUX_ORG", "enose_org")
INFLUX_BUCKET = os.environ.get("INFLUX_BUCKET", "sensor_data")

# --- Sensor Einstellungen ---
# Wie oft soll gemessen werden? (in Sekunden)
# Achtung: SCD30 braucht mind. 2 Sekunden Intervall
SAMPLING_RATE = 2.0

# Höhe über Meeresspiegel (für korrekte Druck-Berechnung) - Optional
SEALEVEL_PRESSURE = 1013.25

# --- Hardware Einstellungen ---
# I2C Pins (als Attribut-Namen von 'board', z.B. "D13" oder "SCL")
I2C_SCL_PIN = "SCL"
I2C_SDA_PIN = "SDA"
