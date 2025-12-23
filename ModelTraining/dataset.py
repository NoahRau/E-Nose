import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib  # Zum Speichern des Scalers (damit wir später im Live-Betrieb gleich normalisieren)

class FridgeDataset(Dataset):
    def __init__(self, csv_file, seq_len=512, mode='train', scaler_path="scaler.pkl"):
        """
        Args:
            csv_file: Pfad zur CSV Datei
            seq_len: Länge des Zeitfensters (Context Window)
            mode: 'train' (fittet Scaler) oder 'val'/'inference' (nutzt existierenden Scaler)
            scaler_path: Wo der Normalizer gespeichert wird
        """
        self.seq_len = seq_len
        self.mode = mode

        # 1. Daten Laden
        print(f"Loading data from {csv_file}...")
        df = pd.read_csv(csv_file)

        # 2. Preprocessing & Cleaning
        # Optional: Zeilen löschen, wo Tür offen war (wenn du nur saubere Phasen lernen willst)
        # df = df[df['soft_door_open'] == 0]

        # Feature Selection: Wir trennen Gas (Chemie) und Env (Physik)
        # Wir suchen Spalten, die mit 'gas_' beginnen
        self.gas_cols = [c for c in df.columns if c.startswith('gas_')]
        self.env_cols = ['co2', 'temp', 'humidity'] # Ggf. anpassen, falls Spaltennamen anders

        # Sicherstellen, dass alles da ist
        missing_gas = [c for c in self.gas_cols if c not in df.columns]
        if missing_gas:
            # Fallback: Falls CSV alt ist und keine 10 Kanäle hat
            print(f"Warnung: Fehlende Gaskanäle {missing_gas}. Fülle mit 0.")
            for c in missing_gas: df[c] = 0.0

        # Daten extrahieren
        gas_data = df[self.gas_cols].values.astype(np.float32)
        env_data = df[self.env_cols].values.astype(np.float32)

        # 3. Normalisierung (Z-Score) -> Extrem wichtig für Transformer!
        if mode == 'train':
            self.scaler_gas = StandardScaler()
            self.scaler_env = StandardScaler()

            gas_data = self.scaler_gas.fit_transform(gas_data)
            env_data = self.scaler_env.fit_transform(env_data)

            # Scaler speichern für später
            joblib.dump({'gas': self.scaler_gas, 'env': self.scaler_env}, scaler_path)
            print(f"Scaler gespeichert in {scaler_path}")

        else:
            # Scaler laden (für Validierung oder Inference nutzen wir den vom Training!)
            scalers = joblib.load(scaler_path)
            self.scaler_gas = scalers['gas']
            self.scaler_env = scalers['env']

            gas_data = self.scaler_gas.transform(gas_data)
            env_data = self.scaler_env.transform(env_data)

        # In Tensoren wandeln und transponieren für [Channels, Time]
        # Dataset liefert später: (Channels, Seq_Len)
        self.gas_data = torch.tensor(gas_data).transpose(0, 1) # [10, Total_Len]
        self.env_data = torch.tensor(env_data).transpose(0, 1) # [3, Total_Len]

        self.n_samples = self.gas_data.shape[1] - self.seq_len

    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        # Sliding Window: Wir schneiden ein Stück der Länge seq_len aus
        # Gas: [10, 512]
        gas_window = self.gas_data[:, idx : idx + self.seq_len]
        # Env: [3, 512]
        env_window = self.env_data[:, idx : idx + self.seq_len]

        return gas_window, env_window

# --- Test Block ---
if __name__ == "__main__":
    # Erstelle dummy CSV zum Testen
    df = pd.DataFrame(np.random.randn(1000, 15), columns=['co2','temp','humidity','pressure','label','soft_door_open'] + [f'gas_{i}' for i in range(10)])
    df.to_csv("test_dummy.csv", index=False)

    ds = FridgeDataset("test_dummy.csv", seq_len=64)
    g, e = ds[0]
    print(f"Gas Shape: {g.shape} (Erwartet: 10, 64)")
    print(f"Env Shape: {e.shape} (Erwartet: 3, 64)")