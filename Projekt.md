# 🧠 Leitfaden: Training, State Machine & Resilienz

Diese Dokumentation beschreibt die Schritte, um die E-Nose "intelligent" zu machen. Sie behandelt nicht den Hardware-Aufbau, sondern fokussiert sich rein auf die Algorithmen, die Datenerfassung und wie das System gegen Fehlalarme (Lüften, voller Kühlschrank) robust gemacht wird.

---

## 📑 Inhaltsverzeichnis

1. [Das Konzept: Hybride Intelligenz](#1-das-konzept-hybride-intelligenz)
2. [Schritt 1: Die State Machine konfigurieren (Logik)](#2-schritt-1-die-state-machine-konfigurieren-logik)
3. [Schritt 2: Trainingsdaten erfassen ("Das Goldene Wochenende")](#3-schritt-2-trainingsdaten-erfassen-das-goldene-wochenende)
4. [Schritt 3: Das ML-Modell trainieren (Isolation Forest)](#4-schritt-3-das-ml-modell-trainieren-isolation-forest)
5. [Schritt 4: Resilienz im Live-Betrieb (Tara & Drift)](#5-schritt-4-resilienz-im-live-betrieb-tara--drift)

---

## 1. Das Konzept: Hybride Intelligenz

Wir lösen das Problem der Fehlalarme durch eine Arbeitsteilung:

* **Die State Machine (Hardcoded Logik):** Erkennt *physikalische* Ereignisse (Tür auf/zu) anhand von rapiden Temperatur-/CO2-Abfällen. Sie wird **nicht trainiert**, sondern eingestellt.
* **Der Isolation Forest (Machine Learning):** Erkennt *chemische* Anomalien (Verfall). Er wird **trainiert**, aber nur mit "guten" Daten.
* **Der Baseline-Filter (Mathematik):** Macht das System unabhängig vom Grundgeruch des Kühlschranks (Tara-Funktion).

---

## 2. Schritt 1: Die State Machine konfigurieren (Logik)

Bevor du Daten für die KI sammelst, muss das System wissen, wann der Deckel offen ist. Sonst lernt die KI fälschlicherweise, dass ein rapider Abfall der Werte "normal" ist.

### Das Experiment (Dauer: ~30 Min)
1.  Lege den Sensor in die leere Box. Deckel zu.
2.  Lass ihn 10 Minuten laufen (Werte stabilisieren sich).
3.  **Aktion:** Reiß den Deckel auf und lass ihn offen.
4.  **Beobachtung:** Schau dir im Live-Plot an, wie schnell CO2 und Temperatur fallen.

### Die Konfiguration
Ermittle die Schwellenwerte für deinen Code.

* **Beispiel-Szenario:** Der CO2-Wert fällt innerhalb von 10 Sekunden von 600ppm auf 450ppm.
    * *Delta:* -150ppm.
* **Deine Regel:** `IF (co2_change_10sec < -100) THEN STATE = OPEN`

> **Wichtig:** Dieser Teil schützt dein ML-Modell davor, durch Frischluft verwirrt zu werden. Solange `STATE == OPEN` oder `RECOVERY`, werden keine Daten an das ML-Modell gesendet!

---

## 3. Schritt 2: Trainingsdaten erfassen ("Das Goldene Wochenende")

Wir nutzen **Unsupervised Learning**. Das bedeutet: Wir zeigen dem Modell nur, wie **"Normal/Frisch"** aussieht. Alles, was davon abweicht, ist später automatisch Alarm.

**⚠️ WICHTIG:** Nutze für das Training **KEINE** verdorbenen Lebensmittel!

### Szenario A: Die "Leerkurve" (Rauschen lernen)
* **Ziel:** Lernen, wie stark der Sensor von alleine schwankt.
* **Setup:** Leere, saubere Box.
* **Dauer:** 2 Stunden laufen lassen.
* **Aktion:** Keine. Einfach aufzeichnen.

### Szenario B: Die "Atmungs-Kurve" (Resilienz gegen Obst)
Frisches Obst atmet CO2 aus. Das ist kein Verfall! Das Modell muss lernen, dass ein linearer CO2-Anstieg harmlos ist.
* **Ziel:** CO2-Anstieg akzeptieren, solange VOC niedrig bleibt.
* **Setup:** 1-2 frische Äpfel oder Bananen in die Box.
* **Dauer:** 4-6 Stunden (über Nacht ist gut).
* **Erkenntnis:** Du wirst sehen, dass CO2 stetig steigt. Das ist **gutes** Trainingsmaterial.

### Szenario C: Die "Basislast-Kurve" (Resilienz gegen Eigengeruch)
Frische Wurst oder Käse riecht, gast aber nicht exponentiell aus (Sättigung).
* **Ziel:** Hohen VOC-Startwert akzeptieren, solange die Steigung flach ist.
* **Setup:** Ein Stück frischer Käse oder offene Wurst.
* **Dauer:** 3-4 Stunden.
* **Erkenntnis:** VOC springt hoch, bleibt dann aber auf einem Plateau.

> **Datensatz-Struktur:** Am Ende solltest du eine CSV-Datei haben, die **nur** diese "gesunden" Verläufe enthält.

---

## 4. Schritt 3: Das ML-Modell trainieren (Isolation Forest)

Hier passiert die Magie auf dem Raspberry Pi. Wir füttern das Modell nicht mit den absoluten Werten, sondern mit der **Dynamik**.

### Feature Engineering (Vorbereitung)
Bevor die Daten in den Algorithmus gehen, wandle sie um:
1.  Nimm nicht `VOC_Wert` (z.B. 50.000 Ohm).
2.  Berechne `VOC_Slope` (Steigung der letzten 5-10 Minuten).
3.  Berechne `CO2_Slope` (Steigung der letzten 5-10 Minuten).

### Das Training (Python Logic)
Nutze `sklearn.ensemble.IsolationForest`.

```python
# Konzeptioneller Code für das Training
features = [
    [0.1, 0.5],   # Leer: Kaum Steigung
    [0.2, 10.0],  # Apfel: Wenig VOC Steigung, viel CO2 Steigung (NORMAL!)
    [0.5, 0.2],   # Wurst: Wenig Steigung (Plateau), wenig CO2
]

# contamination=0.01 bedeutet: Wir gehen davon aus, dass unsere Trainingsdaten 
# zu 99% sauber sind. Das Modell toleriert minimale Ausreißer.
model = IsolationForest(contamination=0.01)
model.fit(features)
joblib.dump(model, 'frische_modell.pkl')
```
# zeugs
3:24 tür zu 
digitales interface für i2c! 

sudo nano /boot/firmware/config.txt
#dtparam=i2c_arm=on
#dtparam=i2c_arm_baudrate=50000
dtoverlay=i2c-gpio,bus=1,i2c_gpio_sda=2,i2c_gpio_scl=3,i2c_gpio_delay_us=20

pip freeze 
Schaltplan anpassen sel auf gnd pin korrigieren und 3.3V für scd30 
Setup script korrigieren mit token key für influx und die libs anpassen je nach pi 4/5 