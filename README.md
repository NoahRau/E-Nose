# 👃 Smart E-Nose: Hybride Frische-Erkennung

Ein Raspberry Pi 5 Projekt zur Erkennung von Lebensmittelverfall mithilfe von **Hybrider KI**. Das System kombiniert eine klassische State Machine (für physikalische Events wie "Tür offen") mit Machine Learning (Isolation Forest für chemische Anomalien).

![Status](https://img.shields.io/badge/Status-Development-orange) ![Python](https://img.shields.io/badge/Python-3.11-blue) ![Hardware](https://img.shields.io/badge/Hardware-RPi5%20%7C%20BME688%20%7C%20SCD30-green)

---

## 📑 Inhaltsverzeichnis

1. [Projekt-Überblick & Funktionsweise](#-projekt-überblick--funktionsweise)
2. [Hardware-Setup & Verkabelung](#-hardware-setup--verkabelung)
3. [Installation](#-installation)
4. [Nutzungs-Leitfaden (Der Workflow)](#-nutzungs-leitfaden-der-workflow)
    * [Schritt 1: Config & Kalibrierung](#schritt-1-config--kalibrierung-state-machine)
    * [Schritt 2: Das "Goldene Wochenende" (Training)](#schritt-2-das-goldene-wochenende-datenerfassung)
    * [Schritt 3: Live-Betrieb](#schritt-3-live-betrieb)
5. [Troubleshooting](#-troubleshooting)
6. [Python Software](#-python-software-einrichten)

---

## 🧠 Projekt-Überblick & Funktionsweise

Herkömmliche Sensoren schlagen oft Fehlalarm, wenn man nur den Kühlschrank öffnet (Temperatursturz) oder wenn Obst "atmet" (CO2-Anstieg). Dieses Projekt löst das Problem durch Arbeitsteilung:

1. **Die State Machine (Hardcoded Logik):**
    * Überwacht **Physik** (Rapide Änderungen von Temp/CO2).
    * Erkennt Zustände wie `OPEN` (Deckel offen) oder `RECOVERY` (Erholung).
    * *Aufgabe:* Blockiert Datenaufzeichnung, wenn der Deckel offen ist, um das KI-Modell nicht zu verwirren.

2. **Der Isolation Forest (Machine Learning):**
    * Überwacht **Chemie** (Gase/VOCs im Verhältnis zur Zeit).
    * Wird nur mit "gesunden" Daten trainiert (Unsupervised Learning).
    * *Aufgabe:* Meldet alles als Anomalie, was nicht wie "frisches Obst" oder "leere Box" aussieht.

---

## 🔌 Hardware-Setup & Verkabelung

Wir nutzen den **I2C-Bus**. Das bedeutet, beide Sensoren werden parallel an dieselben Pins des Raspberry Pi angeschlossen.

**Benötigte Hardware:**

* Raspberry Pi 5
* Sensirion **SCD30** (CO2, Temp, Rh)
* Bosch **BME688** (Gas/VOC, Temp, Press, Rh)
* Jumper-Kabel (Female-Female oder Breadboard)

[Image of Raspberry Pi 40 pin GPIO header pinout]

### Verkabelungsplan

| Pin am Sensor (SCD30 & BME688) | Pin am Raspberry Pi 5 | GPIO Nummer | Funktion |
| :--- | :--- | :--- | :--- |
| **VIN / VCC** | Pin 1 (oder 17) | 3V3 Power | 3.3V Stromversorgung |
| **GND** | Pin 6 (oder 9, 14, etc.) | GND | Masse |
| **SDA** | Pin 3 | GPIO 2 | I2C Data |
| **SCL** | Pin 5 | GPIO 3 | I2C Clock |

[Image of I2C parallel connection diagram multiple sensors raspberry pi]

> **⚠️ Wichtiger Hinweis:**
>
> * Der **SCD30** ist empfindlich bei der Spannung. Wenn du ein Breakout-Board nutzt, prüfe, ob es 3.3V oder 5V benötigt. Die meisten modernen Module (Adafruit/Sparkfun) vertragen 3.3V am VIN.
> * Der **BME688** läuft strikt auf 3.3V. Schließe niemals 5V an die Datenleitungen (SDA/SCL) an!

---

## 🚀 Installation

1. **Repository klonen (oder Ordner erstellen):**

    ```bash
    mkdir ~/e-nose-project
    cd ~/e-nose-project
    ```

2. **Setup-Skript ausführen:**
    Das Skript installiert InfluxDB, Python-Umgebungen und aktiviert I2C (inkl. Clock-Stretching Fix für den SCD30).

    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```

    *Folge den Anweisungen im Terminal und starte den Pi neu, wenn gefragt.*

3. **Zugangsdaten prüfen:**
    Nach der Installation findest du deine DB-Passwörter hier:
    `~/e-nose-project/INFLUX_CREDENTIALS.md`

---

## 🎮 Nutzungs-Leitfaden (Der Workflow)

Um Fehlalarme zu vermeiden, muss das System erst deine Umgebung kennenlernen.

### Schritt 1: Config & Kalibrierung (State Machine)

*Ziel: Dem System beibringen, wann der Deckel offen ist.*

1. Lege Sensoren in deine Box. Deckel zu. 10 Min warten.
2. Reiß den Deckel auf.
3. Beobachte die Werte. Fällt CO2 um 100ppm in 10 Sekunden?
4. Trage diesen Wert in deine `config.py` (oder Hauptskript) ein als Threshold für den `OPEN` State.

### Schritt 2: Das "Goldene Wochenende" (Datenerfassung)

*Ziel: Trainingsdaten sammeln. **WICHTIG:** Nur frische Lebensmittel nutzen!*

Wir zeichnen drei Szenarien auf. Das Skript sollte in einem Modus laufen, der Daten speichert (`mode=training`).

* **Szenario A (Leerlauf - 2h):** Leere Box. Das Modell lernt das Sensor-Rauschen.
* **Szenario B (Atmung - 6h):** Frischer Apfel. CO2 steigt linear an. Das Modell lernt: "Steigendes CO2 ist normal, solange VOC niedrig bleibt."
* **Szenario C (Basislast - 4h):** Frischer Käse/Wurst. VOC springt hoch und bleibt stabil. Das Modell lernt: "Hoher VOC ist okay, wenn er nicht explodiert."

### Schritt 3: Training des Modells

Führe das Trainings-Skript aus. Es nimmt die Daten aus Schritt 2 und erstellt eine `.pkl` Datei (dein Gehirn).

```python
# Beispielhafter Ablauf im Code
# Es werden Features berechnet: VOC_Slope (Steigung) und CO2_Slope
model.fit(training_data)
joblib.dump(model, 'frische_modell.pkl')
```

### Schritt 4: Live-Betrieb

Starte das Hauptprogramm.

1. **System Start:** State Machine ist aktiv.
2. **Box zu:** Daten werden gesammelt, "Tara" (Baseline) wird berechnet.
3. **Analyse:** Die aktuellen Steigungen (Slopes) werden an das `frische_modell.pkl` gesendet.
    * *Ausgabe 1:* "Normal" (Inlier) -> Alles gut.
    * *Ausgabe -1:* "Anomalie" (Outlier) -> **ALARM! Verfall erkannt!**
4. **Box wird geöffnet:**
    * State Machine erkennt Temperatursturz.
    * Status wechselt auf `OPEN`.
    * **KI wird pausiert** (Keine Fehlalarme durch Frischluft).

---

## 🛠 Troubleshooting

Hier sind Lösungen für häufige Probleme bei diesem Setup:

### 1. Fehler `[Errno 121] Remote I/O error`

* **Ursache:** Der SCD30 beherrscht "Clock Stretching", aber der Raspberry Pi ist standardmäßig zu schnell dafür.
* **Lösung:** Prüfe `/boot/firmware/config.txt`. Dort muss stehen:

    ```text
    dtparam=i2c_arm=on
    dtparam=i2c_arm_baudrate=50000
    ```

  (Ein Neustart ist nach Änderung erforderlich!)

### 2. Sensoren werden nicht gefunden

* Führe den Befehl aus: `sudo i2cdetect -y 1`
* **Erwartetes Ergebnis:** Eine Tabelle mit Zahlen.
  * `61`: Adresse des SCD30.
  * `76` oder `77`: Adresse des BME688.
* **Wenn leer:** Verkabelung prüfen (SDA und SCL vertauscht?).

### 3. Setup-Script startet nicht

* **Fehler:** `Permission denied`.
* **Lösung:** Mache die Datei ausführbar mit `chmod +x setup.sh`.

### 4. InfluxDB Probleme

* Falls du das Passwort vergisst oder den Token verlierst:
* Führe `influx auth list` aus (wenn du noch eingeloggt bist) oder setze das Setup zurück, indem du InfluxDB neu installierst (Achtung: Datenverlust).

---

## 🐍 Python-Software einrichten

Nach der Installation benötigen wir die eigentliche Logik, um die Sensoren auszulesen und die Daten in die InfluxDB zu speichern. Wir teilen den Code in drei übersichtliche Dateien auf.

Erstelle die Dateien direkt im Ordner `~/e-nose-project/`:

### 1. Die Konfiguration (`config.py`)

Hier werden die Zugangsdaten und Einstellungen gespeichert.
*Öffne die Datei `INFLUX_CREDENTIALS.md`, um deinen Token zu kopieren!*

```python
# config.py
# --- InfluxDB Einstellungen ---
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "HIER_DEINEN_LANGE_TOKEN_EINFÜGEN" # Siehe INFLUX_CREDENTIALS.md
INFLUX_ORG = "enose_org"
INFLUX_BUCKET = "sensor_data"

# --- Sensor Einstellungen ---
SAMPLING_RATE = 2.0  # Sekunden (SCD30 benötigt mind. 2s)
SEALEVEL_PRESSURE = 1013.25 # Optional für Höhenkorrektur
