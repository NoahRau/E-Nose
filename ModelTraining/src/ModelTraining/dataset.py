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
    Lädt ALLE CSV-Dateien aus einem Ordner.
    Entfernt (purged) Zeilen mit fehlenden Sensorwerten (NaN),
    damit das Training nur auf sauberen Daten läuft.
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
                # Wir lesen die CSV, interpretieren leere Felder automatisch als NaN
                tmp_df = pd.read_csv(f)
                df_list.append(tmp_df)
            except Exception as e:
                logger.warning(f"Fehler beim Laden von {f}: {e}")

        # Alles zu einem großen DataFrame zusammenfügen
        df = pd.concat(df_list, ignore_index=True)

        # 2. Features definieren (Das sind die Spalten, die NICHT leer sein dürfen)
        self.gas_cols = ["bme_gas"]
        self.env_cols = [
            "scd_co2", "scd_temp", "scd_hum",
            "bme_temp", "bme_hum", "bme_pres"
        ]
        all_feature_cols = self.gas_cols + self.env_cols

        # Überprüfen, ob die Spalten überhaupt existieren (falls CSVs alt sind)
        available_cols = [c for c in all_feature_cols if c in df.columns]
        if len(available_cols) != len(all_feature_cols):
            missing = set(all_feature_cols) - set(available_cols)
            logger.warning(f"WARNUNG: Folgende Spalten fehlen komplett im CSV: {missing}")

        # 3. PURGE: Zeilen löschen, die NaN enthalten
        original_count = len(df)

        # dropna entfernt die Zeile, sobald AUCH NUR EINE der Feature-Spalten fehlt
        df.dropna(subset=available_cols, inplace=True)

        new_count = len(df)
        dropped_count = original_count - new_count

        if dropped_count > 0:
            logger.info(f"🧹 PURGE: {dropped_count} unvollständige Zeilen gelöscht (NaN Werte).")
        logger.info(f"✅ Saubere Daten: {new_count} Zeilen bereit für Training.")

        if new_count < seq_len:
            raise ValueError(f"Zu wenig Daten übrig ({new_count}) für Sequenzlänge {seq_len}!")

        # 4. Daten extrahieren
        gas_data = df[self.gas_cols].values.astype(np.float32)
        env_data = df[self.env_cols].values.astype(np.float32)

        # 5. Normalisierung (Z-score)
        if mode == "train":
            self.scaler_gas = StandardScaler()
            self.scaler_env = StandardScaler()

            gas_data = self.scaler_gas.fit_transform(gas_data)
            env_data = self.scaler_env.fit_transform(env_data)

            # Pfad korrigieren (String zu Path objekt falls nötig)
            save_path = Path(scaler_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"gas": self.scaler_gas, "env": self.scaler_env}, save_path)
            logger.info(f"Scaler gespeichert unter {save_path}")
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

        # [Channels, Time] Format für PyTorch
        self.gas_data = torch.tensor(gas_data).transpose(0, 1)
        self.env_data = torch.tensor(env_data).transpose(0, 1)

        self.n_samples = self.gas_data.shape[1] - self.seq_len

    def __len__(self) -> int:
        return max(0, self.n_samples)

    def __getitem__(self, idx: int):
        gas_window = self.gas_data[:, idx : idx + self.seq_len]
        env_window = self.env_data[:, idx : idx + self.seq_len]
        return gas_window, env_window