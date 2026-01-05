"""E-Nose Data Acquisition Module.

This module handles sensor data collection from SCD30 (CO2/Temp/Humidity)
and BME688 (Gas/Temp/Pressure) sensors on Raspberry Pi.
"""

from .door_detector import AdaptiveDoorDetector
from .sensors import SensorManager

__all__ = ["SensorManager", "AdaptiveDoorDetector"]
