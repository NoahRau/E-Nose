"""E-Nose Data Acquisition Module.

This module handles sensor data collection from SCD30 (CO2/Temp/Humidity)
and BME688 (Gas/Temp/Pressure) sensors on Raspberry Pi.
"""

from DataAcquisition.door_detector import AdaptiveDoorDetector
from DataAcquisition.sensors import SensorManager

__all__ = ["SensorManager", "AdaptiveDoorDetector"]
