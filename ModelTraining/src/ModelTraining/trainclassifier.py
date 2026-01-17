import argparse
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

from ModelTraining.model_advanced import FridgeMoCA_V3, MoldClassifierHead

logger = logging.getLogger(__name__)

# --- DEFAULTS ---
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-3
SEQ_LEN = 512

# Pfade
FOUNDATION_MODEL_PATH = Path("checkpoints/fridge_moca_v3_foundation.pth")
SCALER_PATH = Path("checkpoints/scaler.pkl")
LABELED_DATA_DIR = Path("LabeledData")
CLASS_MAPPING = {"Normal": 0, "Schimmel": 1}

def parse_args():
    parser = argparse.ArgumentParser(description="E-Nose Classifier Training (Phase 2)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

# --- Dataset Class ---
class LabeledFridgeDataset(Dataset):
    def __init__(self, root_dir, seq_len=512, scaler_path="scaler.pkl"):
        self.seq_len = seq_len
        self.samples = []

        if not Path(scaler_path).exists():
            raise FileNotFoundError(f"Scaler {scaler_path} fehlt! Erst Phase 1 trainieren.")

        scalers = joblib.load(scaler_path)
        self.scaler_gas = scalers["gas"]
        self.scaler_env = scalers["env"]

        # Define Columns matching YOUR dataset
        self.gas_cols = ["bme_gas"] # 1 Channel
        self.env_cols = ["scd_co2", "scd_temp", "scd_hum", "bme_temp", "bme_hum", "bme_pres"] # 6 Channels

        root_dir = Path(root_dir)
        for label_name, label_idx in CLASS_MAPPING.items():
            class_dir = root_dir / label_name
            if not class_dir.exists():
                logger.warning(f"Ordner fehlt: {class_dir}")
                continue

            files = sorted(list(class_dir.glob("*.csv")))
            logger.info(f"Lade {len(files)} Dateien für '{label_name}'...")

            for f in files:
                try:
                    df = pd.read_csv(f)
                    df.dropna(subset=self.gas_cols + self.env_cols, inplace=True)

                    if len(df) < seq_len: continue

                    vals_gas = df[self.gas_cols].values.astype(np.float32)
                    vals_env = df[self.env_cols].values.astype(np.float32)

                    # Skalierung
                    vals_gas = self.scaler_gas.transform(vals_gas)
                    vals_env = self.scaler_env.transform(vals_env)

                    # Sliding Window (Stride = halbe Länge)
                    stride = seq_len // 2
                    num_windows = len(df) - seq_len

                    for i in range(0, num_windows, stride):
                        g_win = vals_gas[i : i+seq_len]
                        e_win = vals_env[i : i+seq_len]

                        # [Time, Channels] -> [Channels, Time]
                        g_tensor = torch.tensor(g_win).transpose(0, 1)
                        e_tensor = torch.tensor(e_win).transpose(0, 1)

                        self.samples.append((g_tensor, e_tensor, label_idx))

                except Exception as e:
                    logger.warning(f"Fehler bei {f.name}: {e}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def main():
    args = parse_args()
    setup_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | Epochs: {args.epochs} | Batch: {args.batch_size}")

    # 1. Dataset
    dataset = LabeledFridgeDataset(LABELED_DATA_DIR, seq_len=SEQ_LEN, scaler_path=SCALER_PATH)
    if len(dataset) == 0:
        logger.error("Keine Daten geladen. Überprüfe 'LabeledData' Ordnerstruktur!")
        return

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    logger.info(f"Samples: {len(train_set)} Train, {len(val_set)} Val")

    # 2. Modell Setup
    # Ermitteln der tatsächlichen Kanäle aus dem Dataset
    sample_g, sample_e, _ = dataset[0]
    n_gas = sample_g.shape[0] # Sollte 1 sein
    n_env = sample_e.shape[0] # Sollte 6 sein

    logger.info(f"Modell Init: Gas={n_gas}, Env={n_env}")

    foundation = FridgeMoCA_V3(gas_chans=n_gas, env_chans=n_env, seq_len=SEQ_LEN).to(device)

    if FOUNDATION_MODEL_PATH.exists():
        logger.info(f"Lade Weights: {FOUNDATION_MODEL_PATH}")
        # 'strict=False' erlaubt das Laden auch wenn kleine Unterschiede bestehen,
        # aber hier sollte es exakt passen.
        foundation.load_state_dict(torch.load(FOUNDATION_MODEL_PATH, map_location=device))
    else:
        logger.error("Foundation Model fehlt!")
        return

    # Freeze Foundation
    for p in foundation.parameters(): p.requires_grad = False
    foundation.eval()

    # Head Init
    classifier = MoldClassifierHead(embed_dim=384, num_classes=2).to(device)
    optimizer = optim.AdamW(classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # 3. Training Loop
    logger.info("Starte Training...")

    for epoch in range(args.epochs):
        classifier.train()
        total_loss, correct, total = 0, 0, 0

        for gas, env, labels in train_loader:
            gas, env, labels = gas.to(device), env.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                # Features extrahieren
                features = foundation.forward_features(gas, env)
                cls_token = features[:, 0, :]

            logits = classifier(cls_token)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        # Validation
        classifier.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for gas, env, labels in val_loader:
                gas, env, labels = gas.to(device), env.to(device), labels.to(device)
                features = foundation.forward_features(gas, env)
                val_correct += (classifier(features[:,0]).argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        logger.info(f"Ep {epoch+1}: Loss={total_loss/len(train_loader):.4f} | "
                    f"TrainAcc={correct/total:.1%} | ValAcc={val_correct/val_total:.1%}")

    # Save
    torch.save(classifier.state_dict(), "checkpoints/mold_classifier_head.pth")
    logger.info("Classifier gespeichert.")

if __name__ == "__main__":
    main()