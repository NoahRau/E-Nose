import logging

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FridgeDataset(Dataset):
    """PyTorch Dataset for E-Nose sensor data.

    Loads CSV data and splits into gas and environment channels,
    applying Z-score normalization for transformer training.

    Attributes:
        seq_len: Length of the time window (context window).
        mode: 'train' (fits scaler) or 'val'/'inference' (uses existing scaler).
        gas_cols: List of gas channel column names.
        env_cols: List of environment column names (CO2, temp, humidity).
    """

    def __init__(
        self,
        csv_file: str,
        seq_len: int = 512,
        mode: str = "train",
        scaler_path: str = "scaler.pkl",
    ):
        """Initialize the dataset.

        Args:
            csv_file: Path to the CSV data file.
            seq_len: Length of the time window (context window).
            mode: 'train' (fits scaler) or 'val'/'inference' (uses existing scaler).
            scaler_path: Path to save/load the normalizer.
        """
        self.seq_len = seq_len
        self.mode = mode

        # Load data
        logger.info("Loading data from %s", csv_file)
        df = pd.read_csv(csv_file)
        logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

        # Feature selection: separate gas (chemistry) and env (physics)
        self.gas_cols = [c for c in df.columns if c.startswith("gas_")]
        self.env_cols = ["co2", "temp", "humidity"]

        logger.debug("Gas columns: %s", self.gas_cols)
        logger.debug("Env columns: %s", self.env_cols)

        # Check for missing columns
        missing_gas = [c for c in self.gas_cols if c not in df.columns]
        if missing_gas:
            logger.warning("Missing gas channels %s, filling with 0", missing_gas)
            for c in missing_gas:
                df[c] = 0.0

        missing_env = [c for c in self.env_cols if c not in df.columns]
        if missing_env:
            logger.warning("Missing env columns %s, filling with 0", missing_env)
            for c in missing_env:
                df[c] = 0.0

        # Extract data
        gas_data = df[self.gas_cols].values.astype(np.float32)
        env_data = df[self.env_cols].values.astype(np.float32)

        logger.info("Gas data shape: %s", gas_data.shape)
        logger.info("Env data shape: %s", env_data.shape)

        # Normalization (Z-score) - critical for transformer performance
        if mode == "train":
            self.scaler_gas = StandardScaler()
            self.scaler_env = StandardScaler()

            gas_data = self.scaler_gas.fit_transform(gas_data)
            env_data = self.scaler_env.fit_transform(env_data)

            # Save scalers for inference
            joblib.dump({"gas": self.scaler_gas, "env": self.scaler_env}, scaler_path)
            logger.info("Scalers saved to %s", scaler_path)

        else:
            # Load scalers from training
            logger.info("Loading scalers from %s", scaler_path)
            scalers = joblib.load(scaler_path)
            self.scaler_gas = scalers["gas"]
            self.scaler_env = scalers["env"]

            gas_data = self.scaler_gas.transform(gas_data)
            env_data = self.scaler_env.transform(env_data)

        # Convert to tensors and transpose for [Channels, Time] format
        self.gas_data = torch.tensor(gas_data).transpose(0, 1)  # [C_gas, Total_Len]
        self.env_data = torch.tensor(env_data).transpose(0, 1)  # [C_env, Total_Len]

        self.n_samples = self.gas_data.shape[1] - self.seq_len
        logger.info(
            "Dataset initialized: %d samples (seq_len=%d)",
            max(0, self.n_samples), self.seq_len
        )

    def __len__(self) -> int:
        return max(0, self.n_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sliding window sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (gas_window, env_window) tensors with shape [Channels, seq_len].
        """
        gas_window = self.gas_data[:, idx : idx + self.seq_len]
        env_window = self.env_data[:, idx : idx + self.seq_len]
        return gas_window, env_window


# --- Test Block ---
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    # Create dummy CSV for testing
    df = pd.DataFrame(
        np.random.randn(1000, 15),
        columns=["co2", "temp", "humidity", "pressure", "label", "soft_door_open"]
        + [f"gas_{i}" for i in range(10)],
    )
    df.to_csv("test_dummy.csv", index=False)

    ds = FridgeDataset("test_dummy.csv", seq_len=64)
    g, e = ds[0]
    logger.info("Gas Shape: %s (Expected: [10, 64])", g.shape)
    logger.info("Env Shape: %s (Expected: [3, 64])", e.shape)
