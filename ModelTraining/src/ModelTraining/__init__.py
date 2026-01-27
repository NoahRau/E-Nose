"""E-Nose Model Training Module.

This module contains the neural network architectures and training scripts
for the E-Nose anomaly detection system using MAE and DINOv2.
"""

from ModelTraining.dataset import FridgeDataset
from ModelTraining.model import FridgeMoCA
# Change 'FridgeMoCA_Pro' to 'FridgeMoCA_V3' below:
from ModelTraining.model_advanced import FridgeMoCA_V3

__all__ = ["FridgeDataset", "FridgeMoCA", "FridgeMoCA_V3"]