import contextlib
import csv
import logging
import sys
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import board
import busio
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import ASYNCHRONOUS

from DataAcquisition import config
from DataAcquisition.src.DataAcquisition.door_detector import AdaptiveDoorDetector
from DataAcquisition.src.DataAcquisition.sensors import SensorManager

# --- AI Imports ---
from ModelTraining.checkpoints.model_advanced import FridgeMoCA_V3, MoldClassifierHead

# Configure module logger
logger = logging.getLogger(__name__)

# --- AI Configuration ---
SEQ_LEN = 512             # Muss identisch zum Training sein!
CHECKPOINT_DIR = Path("ModelTraining/checkpoints")
FOUNDATION_PATH = CHECKPOINT_DIR / "fridge_moca_v3_foundation.pth"
HEAD_PATH = CHECKPOINT_DIR / "mold_classifier_head.pth"
SCALER_PATH = CHECKPOINT_DIR / "scaler.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logging(log_file: str | None = None, level: int = logging.INFO) -> None:
    """Configure logging for console and file."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    # Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)

def load_ai_models():
    """Lädt Scaler, Foundation Model und Classifier Head."""
    logger.info("--- Lade AI Modelle ---")

    if not SCALER_PATH.exists() or not FOUNDATION_PATH.exists() or not HEAD_PATH.exists():
        logger.error(f"Kritischer Fehler: Modelldateien fehlen in {CHECKPOINT_DIR}!")
        logger.error("Bitte erst 'train_advanced.py' und 'train_classifier.py' ausführen.")
        sys.exit(1)

    # 1. Scaler laden
    try:
        scalers = joblib.load(SCALER_PATH)
        scaler_gas = scalers["gas"]
        scaler_env = scalers["env"]
        logger.info("✔ Scaler geladen.")
    except Exception as e:
        logger.error(f"Fehler beim Laden des Scalers: {e}")
        sys.exit(1)

    # 2. Modelle initialisieren
    # ACHTUNG: Kanalanzahl muss stimmen!
    # Wir nehmen an: Gas=10 (falls BME AI mode) oder 1 (falls nur Resistance).
    # Hier wird dynamisch angepasst, basierend auf dem Scaler (best guess).
    gas_chans = scaler_gas.mean_.shape[0]
    env_chans = scaler_env.mean_.shape[0]
    logger.info(f"Modell Konfiguration: Gas Chans={gas_chans}, Env Chans={env_chans}")

    foundation = FridgeMoCA_V3(gas_chans=gas_chans, env_chans=env_chans, seq_len=SEQ_LEN).to(DEVICE)
    foundation.load_state_dict(torch.load(FOUNDATION_PATH, map_location=DEVICE))
    foundation.eval() # Eval Mode (Batch Norm fixieren etc.)

    classifier = MoldClassifierHead(embed_dim=384, num_classes=2).to(DEVICE)
    classifier.load_state_dict(torch.load(HEAD_PATH, map_location=DEVICE))
    classifier.eval()

    logger.info(f"✔ Modelle auf {DEVICE} geladen.")
    return scaler_gas, scaler_env, foundation, classifier

def get_prediction(buffer, scaler_gas, scaler_env, foundation, classifier):
    """
    Führt die Klassifizierung auf dem aktuellen Buffer durch.
    Gibt die Wahrscheinlichkeit für Schimmel zurück (0.0 - 1.0).
    """
    try:
        # Buffer ist eine Liste von Dictionaries. Wir müssen Arrays daraus machen.
        # Spaltenreihenfolge MUSS identisch zum Training sein!

        # Env: scd_co2, scd_temp, scd_hum, bme_temp, bme_hum, bme_pres
        env_raw = [
            [
                row.get("scd_c", 0), row.get("scd_t", 0), row.get("scd_h", 0),
                row.get("bme_t", 0), row.get("bme_h", 0), row.get("bme_p", 0)
            ]
            for row in buffer
        ]

        # Gas: bme_gas.
        # Falls du nur EINEN Gaswert hast (Widerstand), pack ihn in eine Liste.
        # Falls du 10 hast (BME AI Scan), musst du hier alle 10 holen.
        # Hier: Annahme "bme_g" ist ein einzelner Float-Wert.
        gas_raw = [[row.get("bme_g", 0)] for row in buffer]

        # NumPy Arrays
        gas_arr = np.array(gas_raw, dtype=np.float32)
        env_arr = np.array(env_raw, dtype=np.float32)

        # Skalieren
        gas_arr = scaler_gas.transform(gas_arr)
        env_arr = scaler_env.transform(env_arr)

        # Zu Torch Tensor [Batch, Channels, Time] transformieren
        # Input ist [Time, Channels] -> Transpose zu [Channels, Time]
        gas_t = torch.tensor(gas_arr, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(DEVICE)
        env_t = torch.tensor(env_arr, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # Foundation: Features extrahieren
            features = foundation.forward_features(gas_t, env_t)

            # Head: Klassifizieren (wir nehmen das CLS Token an Index 0)
            cls_token = features[:, 0, :]
            logits = classifier(cls_token)

            # Softmax für Wahrscheinlichkeit
            probs = F.softmax(logits, dim=1)
            mold_prob = probs[0, 1].item() # Index 1 ist "Schimmel"

        return mold_prob

    except Exception as e:
        logger.error(f"Inference Error: {e}")
        return 0.0

def get_user_input():
    print("\n" + "=" * 50)
    print("   E-NOSE LIVE CLASSIFIER & RECORDER")
    print("=" * 50)
    label = input(">> LABEL eingeben (z.B. Test_Kaese): ").strip().replace(" ", "_")
    if not label: label = "Live_Session"
    return label

def main():
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_label = get_user_input()

    # Logging Setup
    log_filename = f"log_LIVE_{experiment_label}_{timestamp_str}.log"
    setup_logging(log_file=log_filename, level=logging.INFO)

    logger.info("-" * 40)
    logger.info("SYSTEM START: RECORDING + AI ANALYSIS")
    logger.info("-" * 40)

    # 1. AI Modelle laden
    scaler_gas, scaler_env, foundation, classifier = load_ai_models()

    # Der Buffer speichert die letzten SEQ_LEN Messungen für das neuronale Netz
    ai_buffer = deque(maxlen=SEQ_LEN)

    # 2. CSV Initialisieren
    csv_filename = f"data_LIVE_{experiment_label}_{timestamp_str}.csv"
    csv_header = [
        "timestamp", "datetime", "label", "door_open",
        "scd_co2", "scd_temp", "scd_hum",
        "bme_temp", "bme_hum", "bme_pres", "bme_gas",
        "ai_mold_prob", "ai_prediction" # <--- Neue Spalten
    ]

    try:
        csv_file = open(csv_filename, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_header)
        logger.info(f"CSV gestartet: {csv_filename}")
    except Exception as e:
        logger.error(f"CSV Fehler: {e}")
        return

    # 3. InfluxDB
    client, write_api = None, None
    if config.INFLUX_TOKEN:
        try:
            client = InfluxDBClient(url=config.INFLUX_URL, token=config.INFLUX_TOKEN, org=config.INFLUX_ORG)
            write_api = client.write_api(write_options=ASYNCHRONOUS)
            logger.info("InfluxDB verbunden.")
        except Exception as e:
            logger.warning(f"InfluxDB Fehler: {e}")

    # 4. Sensoren
    try:
        scl = getattr(board, config.I2C_SCL_PIN)
        sda = getattr(board, config.I2C_SDA_PIN)
        i2c = busio.I2C(scl, sda, frequency=20000)
        sensors = SensorManager(i2c=i2c)
        door_logic = AdaptiveDoorDetector()
        logger.info("Sensoren initialisiert.")
    except Exception as e:
        logger.error(f"Hardware Fehler: {e}")
        sys.exit(1)

    logger.info("Warte auf Buffer-Füllung (ca. 17min bei 2s Takt)...")

    sample_count = 0
    interval = float(config.SAMPLING_RATE)

    try:
        while True:
            loop_start = time.time()

            # A) Lesen
            try:
                readings = sensors.get_formatted_data() or {}
            except Exception:
                readings = {}

            # Werte extrahieren (mit 0 als Fallback)
            scd_c = readings.get("scd_c", 0)
            scd_t = readings.get("scd_t", 0)
            scd_h = readings.get("scd_h", 0)
            bme_t = readings.get("bme_t", 0)
            bme_h = readings.get("bme_h", 0)
            bme_p = readings.get("bme_p", 0)
            bme_g = readings.get("bme_g", 0)

            # B) Tür Logik
            door_open, _, _ = door_logic.update(scd_c, bme_t if bme_t else scd_t)
            door_val = 1 if door_open else 0

            # C) Buffer Update
            # Wir speichern ein Dictionary, um es später einfach abzurufen
            current_row = {
                "scd_c": scd_c, "scd_t": scd_t, "scd_h": scd_h,
                "bme_t": bme_t, "bme_h": bme_h, "bme_p": bme_p, "bme_g": bme_g
            }
            ai_buffer.append(current_row)

            # D) AI Inference (Nur wenn Buffer voll)
            mold_prob = 0.0
            pred_text = "Buffering..."

            if len(ai_buffer) == SEQ_LEN:
                mold_prob = get_prediction(ai_buffer, scaler_gas, scaler_env, foundation, classifier)

                if mold_prob > 0.8:
                    pred_text = "SCHIMMEL!"
                    status_color = "\033[91m" # Rot
                elif mold_prob > 0.5:
                    pred_text = "Verdacht"
                    status_color = "\033[93m" # Gelb
                else:
                    pred_text = "Sauber"
                    status_color = "\033[92m" # Grün
            else:
                status_color = "\033[94m" # Blau für Buffering

            # E) Speichern (CSV)
            csv_writer.writerow([
                loop_start, datetime.now().isoformat(), experiment_label, door_val,
                scd_c, scd_t, scd_h, bme_t, bme_h, bme_p, bme_g,
                f"{mold_prob:.4f}", pred_text
            ])
            csv_file.flush()

            # F) Speichern (InfluxDB)
            if write_api:
                p = Point("sensor_metrics").tag("experiment", experiment_label)
                p.field("scd_co2", float(scd_c))
                p.field("bme_gas", float(bme_g))
                p.field("door", int(door_val))
                p.field("mold_prob", float(mold_prob)) # <--- Der wichtige Wert für Grafana!
                write_api.write(bucket=config.INFLUX_BUCKET, org=config.INFLUX_ORG, record=p)

            # G) Live Ausgabe
            reset = "\033[0m"
            buffer_fill = len(ai_buffer)
            print(f"\r[{buffer_fill}/{SEQ_LEN}] CO2:{scd_c:4.0f} | Gas:{bme_g:6.0f} | {status_color}AI: {pred_text} ({mold_prob*100:5.1f}%){reset}".ljust(100), end="", flush=True)

            # Taktung
            time.sleep(max(0, interval - (time.time() - loop_start)))
            sample_count += 1

    except KeyboardInterrupt:
        print("\n")
        logger.info("Aufnahme gestoppt.")
        logger.info(f"Daten: {csv_filename}")
    finally:
        csv_file.close()
        if write_api: write_api.close()
        if client: client.close()
        sensors.close()

if __name__ == "__main__":
    main()