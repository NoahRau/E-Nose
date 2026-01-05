"""
Specific tool to verify that all sensors (SCD30, BME688) are connected and delivering plausible data.
It logs sensor readings to the console (standard output) for verification.
"""

import logging
import time

from DataAcquisition.sensors import SensorManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_hardware_test():
    logger.info("--- System Start ---")
    try:
        manager = SensorManager()
        logger.info("Sensoren initialisiert. Warte auf stabile Werte (ca. 10s)...")
    except Exception as e:
        logger.error("Fehler bei der Initialisierung: %s", e)
        return

    time.sleep(2)

    # Header for the log output
    header = f"{'ZEIT':<10} | {'BME T':<8} {'BME H':<8} {'GAS':<8} || {'SCD CO2':<10} {'SCD T':<8} {'SCD H':<8}"
    logger.info("-" * 80)
    logger.info(header)
    logger.info("-" * 80)

    try:
        while True:
            data = manager.get_formatted_data()

            t_str = time.strftime("%H:%M:%S")

            # BME Formatierung
            if data["bme_t"] is not None:
                bme_part = (
                    f"{data['bme_t']}°C".ljust(9)
                    + f"{data['bme_h']}%".ljust(9)
                    + f"{data['bme_g'] or '-'}".ljust(9)
                )
            else:
                bme_part = "WARTE... ".ljust(27)

            # SCD Formatierung
            if data["scd_c"] is not None:
                # Hier haben wir jetzt echte Werte!
                scd_part = (
                    f"|| {data['scd_c']} ppm".ljust(13)
                    + f"{data['scd_t']}°C".ljust(9)
                    + f"{data['scd_h']}%".ljust(8)
                )
            else:
                # Wenn der Sensor noch spinnt oder kalibriert
                scd_part = "|| WARTE... (Kalibrierung)"

            logger.info(f"{t_str:<10} | {bme_part} {scd_part}")

            time.sleep(1.0)

    except KeyboardInterrupt:
        logger.info("Ende.")

if __name__ == "__main__":
    run_hardware_test()