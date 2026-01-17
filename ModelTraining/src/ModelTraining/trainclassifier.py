import logging
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

# Imports deiner Module
from ModelTraining.model_advanced import FridgeMoCA_V3, MoldClassifierHead

logger = logging.getLogger(__name__)

# --- CONFIG ---
EPOCHS = 30              # Geht schnell, da nur der Head lernt
LR = 1e-3                # Etwas höhere Rate für den Head ok
BATCH_SIZE = 16          # Bei wenigen Labels eher kleine Batch Size
SEQ_LEN = 512

# Pfade
FOUNDATION_MODEL_PATH = Path("checkpoints/fridge_moca_v3_foundation.pth")
SCALER_PATH = Path("checkpoints/scaler.pkl")
LABELED_DATA_DIR = Path("LabeledData") # Struktur: LabeledData/Normal/*.csv und LabeledData/Schimmel/*.csv

# Labels definieren
CLASS_MAPPING = {"Normal": 0, "Schimmel": 1}

# --- 1. Dataset für gelabelte Daten ---
class LabeledFridgeDataset(Dataset):
    def __init__(self, root_dir, seq_len=512, scaler_path="scaler.pkl"):
        self.seq_len = seq_len
        self.samples = []

        # 1. Scaler laden (WICHTIG: Den gleichen wie beim Foundation Training nutzen!)
        if not Path(scaler_path).exists():
            raise FileNotFoundError(f"Scaler nicht gefunden: {scaler_path}. Bitte erst Pre-Training laufen lassen!")

        scalers = joblib.load(scaler_path)
        self.scaler_gas = scalers["gas"]
        self.scaler_env = scalers["env"]

        # 2. Daten laden
        root_dir = Path(root_dir)
        for label_name, label_idx in CLASS_MAPPING.items():
            class_dir = root_dir / label_name
            if not class_dir.exists():
                logger.warning(f"Ordner fehlt: {class_dir}")
                continue

            files = sorted(list(class_dir.glob("*.csv")))
            logger.info(f"Lade {len(files)} Dateien für Klasse '{label_name}'...")

            for f in files:
                try:
                    df = pd.read_csv(f)
                    # Feature Spalten (Müssen identisch zum Pre-Training sein)
                    gas_cols = ["bme_gas"]
                    env_cols = ["scd_co2", "scd_temp", "scd_hum", "bme_temp", "bme_hum", "bme_pres"]

                    # Drop NaN
                    df.dropna(subset=gas_cols + env_cols, inplace=True)

                    if len(df) < seq_len:
                        continue

                    # Sliding Window über die Datei (oder einfach nur den Anfang?)
                    # Hier: Wir schneiden so viele Fenster wie möglich raus
                    vals_gas = df[gas_cols].values.astype(np.float32)
                    vals_env = df[env_cols].values.astype(np.float32)

                    # Skalieren
                    vals_gas = self.scaler_gas.transform(vals_gas)
                    vals_env = self.scaler_env.transform(vals_env)

                    # Fenster erstellen
                    num_windows = len(df) - seq_len
                    # Stride = seq_len // 2 (50% Overlap für mehr Trainingsdaten)
                    stride = seq_len // 2

                    for i in range(0, num_windows, stride):
                        g_win = vals_gas[i : i+seq_len]
                        e_win = vals_env[i : i+seq_len]
                        # Transpose [Time, Channels] -> [Channels, Time]
                        g_tensor = torch.tensor(g_win).transpose(0, 1)
                        e_tensor = torch.tensor(e_win).transpose(0, 1)

                        self.samples.append((g_tensor, e_tensor, label_idx))

                except Exception as e:
                    logger.warning(f"Fehler bei {f}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# --- 2. Setup Logging ---
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

# --- 3. Main Training ---
def main():
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # A. Dataset
    dataset = LabeledFridgeDataset(LABELED_DATA_DIR, seq_len=SEQ_LEN, scaler_path=SCALER_PATH)
    if len(dataset) == 0:
        logger.error("Keine Trainingsdaten gefunden! Ordnerstruktur prüfen.")
        return

    # Train/Val Split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    logger.info(f"Daten: {len(train_set)} Training, {len(val_set)} Validation")

    # B. Modell Setup
    # 1. Foundation Model laden
    # Wir müssen wissen, wie viele Kanäle wir beim Pre-Training hatten (meist 10 Gas, 6 Env)
    # Hier hartcodiert oder aus den Daten lesen:
    sample_g, sample_e, _ = dataset[0]
    foundation = FridgeMoCA_V3(
        gas_chans=sample_g.shape[0],
        env_chans=sample_e.shape[0],
        seq_len=SEQ_LEN
    ).to(device)

    if FOUNDATION_MODEL_PATH.exists():
        logger.info(f"Lade Foundation Weights von {FOUNDATION_MODEL_PATH}")
        foundation.load_state_dict(torch.load(FOUNDATION_MODEL_PATH, map_location=device))
    else:
        logger.error("Foundation Model nicht gefunden! Bitte erst Phase 1 (train_advanced.py) ausführen.")
        return

    # 2. FREEZE BACKBONE (Wichtig!)
    for param in foundation.parameters():
        param.requires_grad = False
    foundation.eval() # BatchNorm fixieren

    # 3. Classifier Head initialisieren (SwiGLU)
    # embed_dim muss zum Foundation Model passen (V3 = 384)
    classifier = MoldClassifierHead(embed_dim=384, num_classes=2).to(device)

    # Optimizer nur für den Classifier!
    optimizer = optim.AdamW(classifier.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    logger.info("Starte Classifier Training (Phase 2)...")

    # C. Training Loop
    for epoch in range(EPOCHS):
        classifier.train()
        total_loss = 0
        correct = 0
        total = 0

        for gas, env, labels in train_loader:
            gas, env, labels = gas.to(device), env.to(device), labels.to(device)

            optimizer.zero_grad()

            # 1. Features extrahieren (Foundation)
            with torch.no_grad():
                # forward_features gibt [Batch, Tokens, Dim] zurück
                features = foundation.forward_features(gas, env)
                # Wir nehmen nur das [CLS] Token (Index 0)
                cls_token = features[:, 0, :]

            # 2. Klassifizieren (Head)
            logits = classifier(cls_token)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Validation
        classifier.eval()
        val_correct = 0
        val_total = 0
        val_preds_all = []
        val_labels_all = []

        with torch.no_grad():
            for gas, env, labels in val_loader:
                gas, env, labels = gas.to(device), env.to(device), labels.to(device)

                features = foundation.forward_features(gas, env)
                cls_token = features[:, 0, :]
                logits = classifier(cls_token)

                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                val_preds_all.extend(preds.cpu().numpy())
                val_labels_all.extend(labels.cpu().numpy())

        val_acc = val_correct / val_total

        logger.info(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")

    # Final Report
    logger.info("Training beendet.")
    cm = confusion_matrix(val_labels_all, val_preds_all)
    logger.info(f"Confusion Matrix:\n{cm}")

    # Speichern des Heads
    torch.save(classifier.state_dict(), "checkpoints/mold_classifier_head.pth")
    logger.info("Classifier gespeichert unter checkpoints/mold_classifier_head.pth")

if __name__ == "__main__":
    main()