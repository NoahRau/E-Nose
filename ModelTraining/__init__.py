"""E-Nose Model Training Module.

This module contains the neural network architectures and training scripts
for the E-Nose anomaly detection system using MAE and DINOv2.
"""

from ModelTraining.dataset import FridgeDataset
from ModelTraining.model import FridgeMoCA
from ModelTraining.model_advanced import FridgeMoCA_Pro

__all__ = ["FridgeMoCA", "FridgeMoCA_Pro", "FridgeDataset"]
