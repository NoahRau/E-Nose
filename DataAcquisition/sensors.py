import logging

import adafruit_bme680
import adafruit_scd30
import board
import busio

logger = logging.getLogger(__name__)


class SensorManager:
    """Manages I2C communication with SCD30 and BME688 sensors."""

    def __init__(self):
        self.scd = None
        self.bme = None
        self.i2c = None

        try:
            self.i2c = busio.I2C(board.SCL, board.SDA, frequency=20000)
            logger.info("I2C bus initialized.")
        except Exception as e:
            logger.exception("Critical I2C error: %s", e)
            return

        # SCD30 CO2/Temp/Humidity sensor
        try:
            self.scd = adafruit_scd30.SCD30(self.i2c)
            self.scd.measurement_interval = 2
            logger.info("SCD30 connected.")
        except Exception as e:
            logger.warning("SCD30 not found: %s", e)
            self.scd = None

        # BME680/688 Gas/Temp/Humidity/Pressure sensor
        try:
            self.bme = adafruit_bme680.Adafruit_BME680_I2C(self.i2c, address=0x77)
            self.bme.sea_level_pressure = 1013.25
            logger.info("BME688 connected at 0x77.")
        except Exception:
            try:
                self.bme = adafruit_bme680.Adafruit_BME680_I2C(self.i2c, address=0x76)
                self.bme.sea_level_pressure = 1013.25
                logger.info("BME688 connected at 0x76.")
            except Exception as e:
                logger.warning("BME688 not found: %s", e)
                self.bme = None

    def get_formatted_data(self) -> dict:
        """Read and return formatted sensor data.

        Returns:
            Dictionary with sensor readings:
            - scd_c: CO2 in ppm
            - scd_t: Temperature in C (from SCD30)
            - scd_h: Relative humidity % (from SCD30)
            - bme_t: Temperature in C (from BME688)
            - bme_h: Relative humidity % (from BME688)
            - bme_p: Pressure in hPa
            - bme_g: Gas resistance in Ohms
        """
        result = {
            "scd_c": None,
            "scd_t": None,
            "scd_h": None,
            "bme_t": None,
            "bme_h": None,
            "bme_p": None,
            "bme_g": None,
        }

        # BME688 readings
        if self.bme:
            try:
                result["bme_t"] = round(self.bme.temperature, 2)
                result["bme_h"] = round(self.bme.relative_humidity, 2)
                result["bme_p"] = round(self.bme.pressure, 2)

                # Gas property name can vary between library versions
                try:
                    result["bme_g"] = int(self.bme.gas)
                except KeyboardInterrupt:
                    raise
                except Exception:
                    try:
                        result["bme_g"] = int(self.bme.gas_resistance)
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        pass

            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.debug("BME688 read error: %s", e)

        # SCD30 readings
        if self.scd:
            try:
                if self.scd.data_available:
                    result["scd_c"] = int(self.scd.CO2)
                    result["scd_t"] = round(self.scd.temperature, 2)
                    result["scd_h"] = round(self.scd.relative_humidity, 2)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.debug("SCD30 read error: %s", e)

        return result
