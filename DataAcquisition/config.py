# config.py

# --- InfluxDB Einstellungen ---
# Den TOKEN findest du in der Datei: INFLUX_CREDENTIALS.md
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "gxCDdqi1MZfJGk_X5FJH5K5qCd1AFIjpQfAD-rQ2dr0u-c_R4zQ1-l5lZUL8UX9_BHuPLrbhCaXmWJZJr88jiQ=="
INFLUX_ORG = "enose_org"
INFLUX_BUCKET = "sensor_data"

# --- Sensor Einstellungen ---
# Wie oft soll gemessen werden? (in Sekunden)
# Achtung: SCD30 braucht mind. 2 Sekunden Intervall
SAMPLING_RATE = 2.0

# Höhe über Meeresspiegel (für korrekte Druck-Berechnung) - Optional
SEALEVEL_PRESSURE = 1013.25