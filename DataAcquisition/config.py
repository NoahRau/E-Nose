# config.py

# --- InfluxDB Einstellungen ---
# Den TOKEN findest du in der Datei: INFLUX_CREDENTIALS.md
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "HIER_DEN_LANGEN_TOKEN_EINFUEGEN"
INFLUX_ORG = "enose_org"
INFLUX_BUCKET = "sensor_data"

# --- Sensor Einstellungen ---
# Wie oft soll gemessen werden? (in Sekunden)
# Achtung: SCD30 braucht mind. 2 Sekunden Intervall
SAMPLING_RATE = 2.0

# Höhe über Meeresspiegel (für korrekte Druck-Berechnung) - Optional
SEALEVEL_PRESSURE = 1013.25