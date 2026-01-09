# door_detector.py
import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class AdaptiveDoorDetector:
    """Detects door/lid opening events based on relative changes (Z-score).

    Uses adaptive thresholding instead of fixed values. Learns the sensor's
    noise characteristics from recent measurements.

    Attributes:
        SENSITIVITY: Number of standard deviations for detection threshold.
            3.0 = Statistically significant (99.7%)
            4.0 = Very confident event detection (avoids false positives)
    """

    def __init__(self, window_size: int = 60, sensitivity: float = 4.0):
        """Initialize the door detector.

        Args:
            window_size: Number of samples to use for baseline learning.
                         60 samples at 2s interval = ~2 minutes of history.
            sensitivity: Z-score threshold for detection (default: 4.0 sigma).
        """
        self.co2_deltas = deque(maxlen=window_size)
        self.temp_deltas = deque(maxlen=window_size)
        self.SENSITIVITY = sensitivity
        self.last_co2 = None
        self.last_temp = None

        logger.debug(
            "AdaptiveDoorDetector initialized (window=%d, sensitivity=%.1f)",
            window_size,
            sensitivity,
        )

    def update(self, current_co2: float, current_temp: float) -> tuple[int, float, float]:
        """Analyze current values and detect door opening events.

        Args:
            current_co2: Current CO2 reading in ppm.
            current_temp: Current temperature reading in Celsius.

        Returns:
            Tuple of (is_door, sigma_co2, sigma_temp):
            - is_door: 1 if door is detected open, 0 otherwise
            - sigma_co2: Z-score of CO2 change (for debugging)
            - sigma_temp: Z-score of temperature change (for debugging)
        """
        # First call initialization
        if self.last_co2 is None:
            self.last_co2 = current_co2
            self.last_temp = current_temp
            logger.debug("Door detector baseline initialized: CO2=%.0f, Temp=%.1f", current_co2, current_temp)
            return 0, 0.0, 0.0

        # Calculate deltas (change from last step)
        delta_co2 = current_co2 - self.last_co2
        delta_temp = current_temp - self.last_temp

        # Store values for next iteration
        self.last_co2 = current_co2
        self.last_temp = current_temp

        is_door = 0
        sigma_co2 = 0.0
        sigma_temp = 0.0

        # Need at least 10 samples before making judgments
        if len(self.co2_deltas) > 10:
            # Z-Score analysis for CO2
            co2_mean = np.mean(self.co2_deltas)
            co2_std = np.std(self.co2_deltas) + 0.1  # Prevent division by zero
            sigma_co2 = (delta_co2 - co2_mean) / co2_std

            # Z-Score analysis for temperature
            temp_mean = np.mean(self.temp_deltas)
            temp_std = np.std(self.temp_deltas) + 0.01
            sigma_temp = (delta_temp - temp_mean) / temp_std

            # Detection logic:
            # - Significant CO2 drop (negative sigma beyond threshold)
            # - OR rapid temperature rise (positive sigma beyond threshold)
            if sigma_co2 < -self.SENSITIVITY:
                is_door = 1
                logger.debug(
                    "Door detected via CO2 drop: delta=%.1f, sigma=%.2f",
                    delta_co2,
                    sigma_co2,
                )
            elif sigma_temp > self.SENSITIVITY:
                is_door = 1
                logger.debug(
                    "Door detected via temp rise: delta=%.2f, sigma=%.2f",
                    delta_temp,
                    sigma_temp,
                )

        # Update memory buffer
        # IMPORTANT: Don't store extreme deltas from door events in our
        # "normality memory" - otherwise the system would learn that
        # door openings are "normal"
        if is_door == 0:
            self.co2_deltas.append(delta_co2)
            self.temp_deltas.append(delta_temp)

        return is_door, sigma_co2, sigma_temp
