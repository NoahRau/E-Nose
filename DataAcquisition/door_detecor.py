# door_detector.py
import numpy as np
from collections import deque

class AdaptiveDoorDetector:
    """
    Erkennt Tür-Öffnungen basierend auf relativen Änderungen (Z-Score),
    statt auf festen Schwellenwerten. Lernt das Rauschen des Sensors.
    """
    def __init__(self, window_size=60, sensitivity=4.0):
        # window_size=60 bedeutet: Wir lernen aus den letzten ~2 Minuten (bei 2s Takt)
        self.co2_deltas = deque(maxlen=window_size)
        self.temp_deltas = deque(maxlen=window_size)

        # Empfindlichkeit (Sigma):
        # 3.0 = Statistisch signifikant (99.7%)
        # 4.0 = Sehr sicher ein Event (vermeidet Fehlalarme)
        self.SENSITIVITY = sensitivity

        self.last_co2 = None
        self.last_temp = None

    def update(self, current_co2, current_temp):
        """
        Analysiert die aktuellen Werte und gibt zurück, ob eine Anomalie vorliegt.
        
        Returns:
            is_door (int): 1 wenn Tür offen, 0 wenn zu
            sigma_co2 (float): Wie stark weicht CO2 ab? (für Debugging)
            sigma_temp (float): Wie stark weicht Temp ab?
        """
        # 1. Initialisierung beim allerersten Start
        if self.last_co2 is None:
            self.last_co2 = current_co2
            self.last_temp = current_temp
            return 0, 0.0, 0.0

        # 2. Delta berechnen (Veränderung zum letzten Schritt)
        delta_co2 = current_co2 - self.last_co2
        delta_temp = current_temp - self.last_temp

        # Werte für nächsten Schritt merken
        self.last_co2 = current_co2
        self.last_temp = current_temp

        is_door = 0
        sigma_co2 = 0.0
        sigma_temp = 0.0

        # Wir brauchen erst ein paar Daten (z.B. 10 Puffer-Einträge), bevor wir urteilen
        if len(self.co2_deltas) > 10:
            # --- Z-Score Analyse ---

            # CO2
            co2_mean = np.mean(self.co2_deltas)
            co2_std = np.std(self.co2_deltas) + 0.1 # +0.1 verhindert Division durch Null
            sigma_co2 = (delta_co2 - co2_mean) / co2_std

            # Temp
            temp_mean = np.mean(self.temp_deltas)
            temp_std = np.std(self.temp_deltas) + 0.01
            sigma_temp = (delta_temp - temp_mean) / temp_std

            # LOGIK: Ist die Abweichung negativ genug (CO2 Sturz) und stark genug?
            if sigma_co2 < -self.SENSITIVITY:
                is_door = 1
            # Oder rapider ANSTIEG bei Temp
            elif sigma_temp > self.SENSITIVITY:
                is_door = 1

        # 3. Gedächtnis updaten
        # WICHTIG: Wenn die Tür offen ist, speichern wir dieses extreme Delta NICHT 
        # in unser "Normalitäts-Gedächtnis". Sonst denkt der Sensor beim nächsten Mal,
        # dass Türöffnen "normal" ist.
        if is_door == 0:
            self.co2_deltas.append(delta_co2)
            self.temp_deltas.append(delta_temp)

        return is_door, sigma_co2, sigma_temp