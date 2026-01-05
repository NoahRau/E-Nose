#!/bin/bash

# ==========================================
# E-Nose Setup Script for Raspberry Pi 5
# Hardware: BME688 & SCD30
# Fix: Enforced Python 3.11 to avoid Build Errors
# ==========================================

# Colors for Output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# --- CONFIGURATION ---
INFLUX_USER="admin"
INFLUX_PASSWORD="enose_secret_password"  # Change this after setup!
INFLUX_ORG="enose_org"
INFLUX_BUCKET="sensor_data"
INFLUX_RETENTION="300d"

echo -e "${GREEN}[*] Starting Setup for E-Nose Project (Pi 5 Safe Mode)...${NC}"

# 1. Update System
echo -e "${GREEN}[*] Updating system packages...${NC}"
sudo apt-get update && sudo apt-get upgrade -y

# 2. Install System Dependencies (Explicitly requesting Python 3.11)
echo -e "${GREEN}[*] Installing dependencies (Python 3.11 specific)...${NC}"
# We specifically install python3.11-dev and python3.11-venv to fix the "Python.h" missing error
sudo apt-get install -y \
    i2c-tools \
    build-essential \
    libatlas-base-dev \
    wget \
    gpg \
    libgpiod2 \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils

# Add InfluxDB Repository Key
wget -q https://repos.influxdata.com/influxdata-archive_compat.key
echo "393e8779c89ac8d958f81f942f9ad7fb82a25e133faddaf92e15b16e6ac9ce4c influxdata-archive_compat.key" | sha256sum -c && cat influxdata-archive_compat.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg > /dev/null
echo 'deb [signed-by=/etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg] https://repos.influxdata.com/debian stable main' | sudo tee /etc/apt/sources.list.d/influxdata.list

# Install InfluxDB
sudo apt-get update
sudo apt-get install -y influxdb2 influxdb2-cli

# Start InfluxDB
sudo systemctl unmask influxdb
sudo systemctl enable influxdb
sudo systemctl start influxdb

echo -e "${GREEN}[*] Waiting for InfluxDB start...${NC}"
sleep 10

# 3. InfluxDB Setup
echo -e "${GREEN}[*] Configuring InfluxDB...${NC}"

if influx ping > /dev/null 2>&1; then
    if [ ! -f "$HOME/e-nose-project/INFLUX_CREDENTIALS.md" ]; then
         echo -e "${YELLOW}[!] InfluxDB running, executing initial setup...${NC}"
         SETUP_OUTPUT=$(influx setup --username "$INFLUX_USER" --password "$INFLUX_PASSWORD" --org "$INFLUX_ORG" --bucket "$INFLUX_BUCKET" --retention "$INFLUX_RETENTION" --force 2>&1)
         ADMIN_TOKEN=$(influx auth list --user "$INFLUX_USER" --hide-headers | cut -f 4)
    else
         echo -e "${GREEN}[OK] InfluxDB already configured.${NC}"
    fi
fi

# 4. Create Project Folder & Credentials
PROJECT_DIR="$HOME/e-nose-project"
mkdir -p "$PROJECT_DIR"
CRED_FILE="$PROJECT_DIR/INFLUX_CREDENTIALS.md"

if [ ! -z "$ADMIN_TOKEN" ]; then
    echo -e "${GREEN}[*] Saving credentials to $CRED_FILE...${NC}"
    cat <<EOF > "$CRED_FILE"
# InfluxDB Credentials
Generated: $(date)

## Login Details
- **URL:** http://localhost:8086
- **Username:** $INFLUX_USER
- **Password:** $INFLUX_PASSWORD
- **Organization:** $INFLUX_ORG
- **Bucket:** $INFLUX_BUCKET

## API Token
\`\`\`
$ADMIN_TOKEN
\`\`\`
EOF
fi

# 5. I2C Interface Setup
echo -e "${GREEN}[*] Enabling I2C and optimizing for SCD30...${NC}"
sudo raspi-config nonint do_i2c 0
sudo modprobe i2c-dev
sudo modprobe i2c-bcm2835

CONFIG_TXT="/boot/firmware/config.txt"

if ! grep -q "dtparam=i2c_arm=on" "$CONFIG_TXT"; then
    echo "dtparam=i2c_arm=on" | sudo tee -a "$CONFIG_TXT"
fi

# SCD30 Clock Stretching Fix
if ! grep -q "dtparam=i2c_arm_baudrate" "$CONFIG_TXT"; then
    echo -e "${YELLOW}[!] Setting I2C baudrate to 50000 for SCD30 stability...${NC}"
    echo "dtparam=i2c_arm_baudrate=50000" | sudo tee -a "$CONFIG_TXT"
else
    echo -e "${GREEN}[OK] I2C baudrate already configured.${NC}"
fi

# 6. Python Environment Setup (STRICT PYTHON 3.11)
cd "$PROJECT_DIR"

# DELETE OLD VENV if it exists (to clear out the broken 3.13 version)
if [ -d "venv" ]; then
    echo -e "${YELLOW}[!] Removing old venv to ensure clean Python 3.11 install...${NC}"
    rm -rf venv
fi

echo -e "${GREEN}[*] Creating Python 3.11 venv...${NC}"
python3.11 -m venv venv

echo -e "${GREEN}[*] Installing Python libraries...${NC}"
./venv/bin/pip install --upgrade pip setuptools wheel

# Install libraries
# Note for Pi 5: We use rpi-lgpio instead of RPi.GPIO if available,
# but installing RPi.GPIO inside a Py3.11 environment often works via compatibility layers.
./venv/bin/pip install \
    adafruit-circuitpython-bme680 \
    adafruit-circuitpython-scd30 \
    adafruit-blinka \
    pandas \
    numpy \
    scikit-learn \
    joblib \
    rpi-lgpio \
    influxdb-client

# Note: replaced 'RPi.GPIO' with 'rpi-lgpio' above.
# rpi-lgpio is the modern drop-in replacement specifically for Pi 5.

# 7. Permissions
sudo usermod -aG i2c,gpio $USER

# 8. Finish
echo -e "${GREEN}==============================================${NC}"
echo -e "${GREEN}    SETUP COMPLETE (Python 3.11)${NC}"
echo -e "${GREEN}==============================================${NC}"
echo -e "1. Credentials: ${YELLOW}$CRED_FILE${NC}"
echo -e "2. Start project: source $PROJECT_DIR/venv/bin/activate"
echo -e "3. Verify sensors: ${YELLOW}i2cdetect -y 1${NC}"
echo -e ""
echo -e "${YELLOW}NOTE: Reboot required for I2C baudrate changes.${NC}"
read -p "Reboot now? (y/n) " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
    sudo reboot
fi