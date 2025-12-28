import time
import board
import bitbangio

# Diese Frequenzen testen wir nacheinander
# SCD30 mag oft Bereiche zwischen 10000 (10kHz) und 50000 (50kHz)
FREQUENZEN = [1000, 5000, 10000, 20000, 40000, 50000, 100000]

print("--- Starte I2C Baudraten-Scanner ---")
print("Voraussetzung: dtoverlay=i2c-gpio muss in config.txt deaktiviert sein!")

target_scd = 0x61
target_bme = 0x77

found_best_freq = False

for freq in FREQUENZEN:
    print(f"\n[TEST] Prüfe Frequenz: {freq} Hz ...")

    i2c = None
    try:
        # Wir erstellen einen temporären Bus mit der aktuellen Frequenz
        i2c = bitbangio.I2C(board.SCL, board.SDA, frequency=freq)

        # Versuchen den Bus zu sperren für den Scan
        while not i2c.try_lock():
            pass

        # Scan durchführen
        adressen = i2c.scan()
        i2c.unlock()

        print(f"   -> Gefundene Geräte: {[hex(x) for x in adressen]}")

        if target_scd in adressen and target_bme in adressen:
            print(f"   >>> TREFFER! Beide Sensoren bei {freq} Hz gefunden! <<<")
            found_best_freq = True
            # Optional: Hier könnte man weitermachen, aber wir haben ja gefunden was wir wollen
        elif target_scd in adressen:
            print(f"   -> Nur SCD30 gefunden.")
        elif target_bme in adressen:
            print(f"   -> Nur BME688 gefunden.")
        else:
            print("   -> Nichts gefunden.")

    except ValueError as e:
        print(f"   [FEHLER] Pins sind belegt! Hast du config.txt bereinigt? ({e})")
        break
    except Exception as e:
        print(f"   [FEHLER] {e}")
    finally:
        # Wichtig: Bus wieder freigeben für den nächsten Durchlauf
        if i2c:
            i2c.deinit()

print("\n--- Scan beendet ---")