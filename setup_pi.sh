#!/bin/bash

# ==========================================
# E-Nose Setup Script für Raspberry Pi 5
# (Inkl. InfluxDB Setup & Auto-Konfiguration)
# ==========================================

# Farben für Output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# --- KONFIGURATION FÜR INFLUXDB STANDARD-USER ---
INFLUX_USER="admin"
INFLUX_PASSWORD="enose_secret_password"  # Sollte nach Setup geändert werden!
INFLUX_ORG="enose_org"
INFLUX_BUCKET="sensor_data"
INFLUX_RETENTION="300d" # Daten werden nach 30 Tagen gelöscht

echo -e "${GREEN}[*] Starte Setup für E-Nose Projekt...${NC}"

# 1. System aktualisieren
echo -e "${GREEN}[*] Aktualisiere System-Pakete...${NC}"
sudo apt-get update && sudo apt-get upgrade -y

# 2. Notwendige System-Pakete installieren
# - influxdb2: Die Zeitreihen-Datenbank
# - influxdb2-cli: Command Line Interface zum Einrichten der DB
echo -e "${GREEN}[*] Installiere System-Abhängigkeiten (inkl. InfluxDB Repo)...${NC}"
sudo apt-get install -y i2c-tools python3-venv python3-pip python3-dev build-essential libatlas-base-dev wget gpg

# InfluxDB Repository Key hinzufügen (für Debian Bookworm/Pi OS)
wget -q https://repos.influxdata.com/influxdata-archive_compat.key
echo "393e8779c89ac8d958f81f942f9ad7fb82a25e133faddaf92e15b16e6ac9ce4c influxdata-archive_compat.key" | sha256sum -c && cat influxdata-archive_compat.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg > /dev/null
echo 'deb [signed-by=/etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg] https://repos.influxdata.com/debian stable main' | sudo tee /etc/apt/sources.list.d/influxdata.list

# Paketquellen neu laden und InfluxDB installieren
sudo apt-get update
sudo apt-get install -y influxdb2 influxdb2-cli

# InfluxDB starten und aktivieren
sudo systemctl unmask influxdb
sudo systemctl enable influxdb
sudo systemctl start influxdb

echo -e "${GREEN}[*] Warte auf InfluxDB Start...${NC}"
sleep 10

# 3. InfluxDB Setup (Automatisch User anlegen)
echo -e "${GREEN}[*] Konfiguriere InfluxDB...${NC}"

# Prüfen, ob InfluxDB schon eingerichtet ist
if influx ping > /dev/null 2>&1; then
    if [ ! -f "$HOME/e-nose-project/INFLUX_CREDENTIALS.md" ]; then
         echo -e "${YELLOW}[!] InfluxDB läuft, aber Setup scheint noch nicht ausgeführt zu sein (oder Credentials fehlen).${NC}"
         echo -e "${YELLOW}[!] Führe initiales Setup aus...${NC}"

         # Setup ausführen und Token extrahieren
         SETUP_OUTPUT=$(influx setup --username "$INFLUX_USER" --password "$INFLUX_PASSWORD" --org "$INFLUX_ORG" --bucket "$INFLUX_BUCKET" --retention "$INFLUX_RETENTION" --force 2>&1)

         # Token aus der Config lesen (Da influx setup den Token nur in die lokale Config schreibt)
         # Wir holen uns den Token direkt über auth list, da wir jetzt admin sind
         ADMIN_TOKEN=$(influx auth list --user "$INFLUX_USER" --hide-headers | cut -f 4)
    else
         echo -e "${GREEN}[OK] InfluxDB ist bereits konfiguriert.${NC}"
    fi
fi

# 4. Projekt-Ordner & Credential File erstellen
PROJECT_DIR="$HOME/e-nose-project"
mkdir -p "$PROJECT_DIR"
CRED_FILE="$PROJECT_DIR/INFLUX_CREDENTIALS.md"

# Wenn wir gerade ein Setup gemacht haben (Token Variable ist gesetzt), speichern wir die Daten
if [ ! -z "$ADMIN_TOKEN" ]; then
    echo -e "${GREEN}[*] Speichere Zugangsdaten in $CRED_FILE...${NC}"
    cat <<EOF > "$CRED_FILE"
# InfluxDB Zugangsdaten
Generiert am: $(date)

## Login Details
- **URL:** http://localhost:8086
- **Username:** $INFLUX_USER
- **Password:** $INFLUX_PASSWORD
- **Organization:** $INFLUX_ORG
- **Bucket:** $INFLUX_BUCKET

## API Token (WICHTIG FÜR PYTHON SKRIPT!)
\`\`\`
$ADMIN_TOKEN
\`\`\`

> Bitte kopiere diesen Token in dein Python-Skript (logic.py / main.py).
EOF
fi


# 5. I2C Schnittstelle aktivieren
echo -e "${GREEN}[*] Aktiviere I2C Interface...${NC}"
sudo raspi-config nonint do_i2c 0
sudo modprobe i2c-dev
sudo modprobe i2c-bcm2835

# Fallback für config.txt
if ! grep -q "dtparam=i2c_arm=on" /boot/firmware/config.txt; then
    echo "dtparam=i2c_arm=on" | sudo tee -a /boot/firmware/config.txt
fi

# 6. Python Environment Setup
cd "$PROJECT_DIR"
if [ ! -d "venv" ]; then
    echo -e "${GREEN}[*] Erstelle Python venv...${NC}"
    python3 -m venv venv
fi

echo -e "${GREEN}[*] Installiere Python-Bibliotheken (inkl. InfluxDB Client)...${NC}"
./venv/bin/pip install --upgrade pip
./venv/bin/pip install \
    adafruit-circuitpython-bme680 \
    pandas \
    numpy \
    scikit-learn \
    joblib \
    RPi.GPIO \
    influxdb-client

# 7. Berechtigungen
sudo usermod -aG i2c,gpio $USER

# 8. Abschluss
echo -e "${GREEN}==============================================${NC}"
echo -e "${GREEN}   SETUP ABGESCHLOSSEN!${NC}"
echo -e "${GREEN}==============================================${NC}"
echo -e "1. Deine Datenbank-Zugangsdaten liegen hier: ${YELLOW}$CRED_FILE${NC}"
echo -e "2. Starte dein Projekt mit: source $PROJECT_DIR/venv/bin/activate"
echo -e ""
read -p "Soll jetzt neu gestartet werden? (j/n) " response
if [[ "$response" =~ ^([jJ][aA]|[jJ])+$ ]]; then
    sudo reboot
fi