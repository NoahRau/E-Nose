<div align="center">

<img src="docs/assets/logo.png" width="150" alt="E-Nose Logo">

<h1>Smart E-Nose</h1>

<h3>Hybride Frische-Erkennung</h3>

<p>
  Ein Raspberry Pi 5 Projekt zur Erkennung von Lebensmittelverfall mithilfe von <b>Hybrider KI</b>.
  <br>
  <sub>State Machine + Isolation Forest = Keine Fehlalarme</sub>
</p>

<p>
  <a href="#-installation"><img src="https://img.shields.io/badge/Status-Development-orange" alt="Status"></a>
  <a href="#running"><img src="https://img.shields.io/badge/Python-3.11-blue" alt="Python"></a>
  <a href="#-hardware-setup--verkabelung"><img src="https://img.shields.io/badge/Hardware-RPi5%20%7C%20BME688%20%7C%20SCD30-green" alt="Hardware"></a>
</p>

</div>

---

## 📑 Inhaltsverzeichnis

1. [Projekt-Überblick & Funktionsweise](#-projekt-überblick--funktionsweise)
2. [Hardware-Setup & Verkabelung](#-hardware-setup--verkabelung)
3. [Installation](#-installation)
4. [Skripte laufen lassen](#running)
5. [Nutzungs-Leitfaden (Der Workflow)](#-nutzungs-leitfaden-der-workflow)
    * [Schritt 1: Config & Kalibrierung](#schritt-1-config--kalibrierung-state-machine)
    * [Schritt 2: Das "Goldene Wochenende" (Training)](#schritt-2-das-goldene-wochenende-datenerfassung)
6. [Troubleshooting](#-troubleshooting)

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

<p align="center">
  <img src="docs/assets/circuit.png" width="50%">
</p>

---

## 🚀 Installation

### Option A: Automatisch via Ansible (Empfohlen)

Das Ansible Playbook provisioniert einen Raspberry Pi vollständig: Pakete, I2C, InfluxDB, UV, und das Projekt.

> Getestet mit Ubuntu Server 25.10 64-Bit on pi4/5

```bash
ansible-playbook setup_pi.yml -i "192.168.0.21,"
```

> **Hinweis:** Passe die IP-Adresse in `setup_pi.yml` an deinen Pi an.

Das Playbook:

* Installiert System-Pakete und UV
* Konfiguriert I2C (Software-Bitbang für SCD30 Kompatibilität)
* Erkennt automatisch Pi4 vs Pi5 und installiert die korrekten GPIO-Libraries
* Richtet InfluxDB ein und speichert Credentials

### Option B: Manuell mit UV

Um die Dependencies manuell zu installieren, müssen manuell je nach pi Modell die spezifischen libs installiert werden.

> **Warum?**
>
> Der Pi5 verwendet den neuen RP1-Chip für GPIO. Die klassische `RPi.GPIO`-Library funktioniert dort nicht. `rpi-lgpio` ist ein Drop-in-Replacement, das die gleiche API bereitstellt aber intern `lgpio` verwendet.

1. **Dependencies installieren (Pi-Version beachten!):**

    ```bash
    # Raspberry Pi 4 oder älter
    uv sync --extra pi4

    # Raspberry Pi 5
    uv sync --extra pi5
    ```

---

## Running

Alle Tools können direkt über `uv run` ausgeführt werden. Sie befinden sich im globalen Pfad des Projekts.

### 📊 1. Daten plotten

Visualisiert aufgezeichnete CSV-Dateien und speichert das Bild automatisch als PNG.

```bash
uv run enose-plot Data/data_Raum_20251229_031040.csv
```

* Erzeugt eine PNG-Datei im gleichen Ordner.
* Zeigt ein interaktives Fenster.

### 🧪 2. Simulation

Testet den "Adaptive Door Detector" Algorithmus mit generierten Daten. Ideal zum Entwickeln ohne Hardware.

```bash
uv run enose-simulate
```

### 📡 3. Hardware-Test (Nur Pi)

Prüft, ob Sensoren (SCD30, BME688) korrekt angeschlossen sind und liefert Live-Werte im Terminal.

```bash
uv run enose-hwtest
```

### 📝 4. Daten-Logger (Nur Pi)

Startet die Langzeit-Aufzeichnung. Speichert CSV lokal und sendet Metriken an InfluxDB.

```bash
uv run enose-logger
```

### 5. Training

Trainiert im hintergrung:

```bash
nohup uv run enose-train-advanced --logfile training.log --no-console &
```

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

* **Ursache:** Der SCD30 beherrscht "Clock Stretching", aber der Hardware-I2C des Raspberry Pi hat damit Probleme.
* **Lösung:** Nutze Software-I2C via GPIO Overlay. Prüfe `/boot/firmware/config.txt`:

    ```text
    # Hardware I2C deaktivieren
    #dtparam=i2c_arm=on
    #dtparam=i2c_arm_baudrate=50000

    # Software I2C aktivieren
    dtoverlay=i2c-gpio,bus=1,i2c_gpio_sda=2,i2c_gpio_scl=3,i2c_gpio_delay_us=20
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
