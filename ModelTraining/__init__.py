"""E-Nose Model Training Module.

This module contains the neural network architectures and training scripts
for the E-Nose anomaly detection system using MAE and DINOv2.
"""

from .dataset import FridgeDataset
from .model import FridgeMoCA
from .model_advanced import FridgeMoCA_Pro

__all__ = ["FridgeMoCA", "FridgeMoCA_Pro", "FridgeDataset"]
