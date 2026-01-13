import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class FridgeDataset(Dataset):
    """
    Lädt ALLE CSV-Dateien aus einem Ordner für das DINO-Training.
    Kombiniert Gas- und Umweltsensoren.
    """

    def __init__(
            self,
            data_dir: str | Path,
            seq_len: int = 512,
            mode: str = "train",
            scaler_path: str = "scaler.pkl",
    ):
        self.seq_len = seq_len
        self.mode = mode
        data_dir = Path(data_dir)

        # 1. Alle CSVs finden und laden
        files = sorted(list(data_dir.glob("*.csv")))
        if not files:
            raise FileNotFoundError(f"Keine .csv Dateien in {data_dir} gefunden.")

        logger.info(f"Lade {len(files)} CSV-Dateien aus {data_dir}...")

        df_list = []
        for f in files:
            try:
                # Nur relevante Zeilen laden, falls Formatierungsprobleme existieren
                tmp_df = pd.read_csv(f)
                # Optional: Hier könnte man NaN-Werte pro Datei füllen
                df_list.append(tmp_df)
            except Exception as e:
                logger.warning(f"Fehler beim Laden von {f}: {e}")

        # Alles zu einem großen DataFrame zusammenfügen
        df = pd.concat(df_list, ignore_index=True)
        logger.info(f"Gesamtdaten: {len(df)} Zeilen.")

        # 2. Features auswählen (Raw Data für DINO!)
        # Wir nutzen MEHR Umweltdaten für besseren Kontext (Translativity)
        self.gas_cols = ["bme_gas"]

        # Erweiterte Umwelt-Features (SCD41 + BME680 Werte)
        # HINWEIS: 'bme_pres' (Luftdruck) ist oft ein guter Indikator für Wetteränderungen
        self.env_cols = [
            "scd_co2", "scd_temp", "scd_hum",
            "bme_temp", "bme_hum", "bme_pres"
        ]

        # Fehlende Spalten mit 0 auffüllen (falls in manchen CSVs nicht vorhanden)
        for c in self.gas_cols + self.env_cols:
            if c not in df.columns:
                logger.warning(f"Spalte {c} fehlt, wird mit 0 gefüllt.")
                df[c] = 0.0

        # Daten extrahieren
        gas_data = df[self.gas_cols].values.astype(np.float32)
        env_data = df[self.env_cols].values.astype(np.float32)

        # 3. Normalisierung (WICHTIG für Transformer)
        if mode == "train":
            self.scaler_gas = StandardScaler()
            self.scaler_env = StandardScaler()

            gas_data = self.scaler_gas.fit_transform(gas_data)
            env_data = self.scaler_env.fit_transform(env_data)

            joblib.dump({"gas": self.scaler_gas, "env": self.scaler_env}, scaler_path)
            logger.info(f"Scaler gespeichert unter {scaler_path}")
        else:
            if Path(scaler_path).exists():
                logger.info(f"Lade Scaler von {scaler_path}")
                scalers = joblib.load(scaler_path)
                self.scaler_gas = scalers["gas"]
                self.scaler_env = scalers["env"]

                gas_data = self.scaler_gas.transform(gas_data)
                env_data = self.scaler_env.transform(env_data)
            else:
                logger.warning("Kein Scaler gefunden für Inference, nutze rohe Daten (nicht empfohlen!)")

        # [Channels, Time] Format für PyTorch (Conv1d/Transformer erwartet oft Channels first im Embedding)
        self.gas_data = torch.tensor(gas_data).transpose(0, 1)
        self.env_data = torch.tensor(env_data).transpose(0, 1)

        self.n_samples = self.gas_data.shape[1] - self.seq_len

    def __len__(self) -> int:
        return max(0, self.n_samples)

    def __getitem__(self, idx: int):
        # Sliding Window über den gesamten Datensatz
        # (Hinweis: An den Übergängen zwischen zwei CSV-Dateien gibt es hier einen kleinen "Sprung",
        #  das ist für das Pre-Training aber meist akzeptabel und glättet sich über die Menge.)
        gas_window = self.gas_data[:, idx : idx + self.seq_len]
        env_window = self.env_data[:, idx : idx + self.seq_len]
        return gas_window, env_window