"""
Simulates realistic sensor data (CO2, Temperature) with noise and drift to
verify and tune the AdaptiveDoorDetector algorithm without needing physical hardware.
Generates plots using matplotlib to visualize the simulation results.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np

from DataAcquisition.door_detector import AdaptiveDoorDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def generate_realistic_data(
    steps=600,
    dt=2.0,
    door_open_at=120,
    door_duration=25,
    seed=42,
):
    rng = np.random.default_rng(seed)

    # --- "Wahre" Umgebungszustände (latent) ---
    # Innenraum-Baseline
    co2_true = np.empty(steps)
    temp_true = np.empty(steps)

    # Startwerte (typisch: Innenraum 700-1200, kühlere Umgebung z.B. 4-10°C)
    co2_true[0] = 850.0
    temp_true[0] = 6.0

    # Außenluft / angrenzender Bereich (z.B. Flur/außen)
    co2_out = 430.0
    temp_out = 10.0  # z.B. wärmerer Flur oder Außenluft im Sommer; anpassen!

    # Langsame Drift / Quellen (Menschen, Pflanzen, etc.)
    # CO2-Quelle: ppm pro Sekunde (sehr klein, aber über Zeit sichtbar)
    co2_source = 0.02  # ~0.02 ppm/s => 2.4 ppm/min
    # Temp Drift Richtung "HVAC setpoint"
    temp_set = 6.5

    # Zusätzliche zufällige "Belegung" Events (CO2 bumps)
    bump_prob = 0.01  # pro Schritt
    bump_strength = (20, 80)  # ppm jump in "true" CO2, verteilt über mehrere Schritte

    # Tür-Mischungsparameter (wie stark Innenluft mit Außenluft gemischt wird pro Schritt)
    # 0.0 = keine Mischung, 1.0 = komplett ersetzt
    mix_when_open = 0.10  # pro Zeitschritt; realistisch eher 0.05-0.20 je nach Luftzug

    # Zeitkonstanten (wie schnell "true" Zustände zur Umgebung tendieren)
    # (Zusätzliche Relaxation auch ohne Tür, z.B. Leckage/Belüftung)
    leak = 0.003  # pro Schritt

    # --- Simulation "true" ---
    bump_remaining = 0
    bump_per_step = 0

    for t in range(1, steps):
        # zufällige CO2 bump starten (Person kommt rein / Aktivität)
        if bump_remaining <= 0 and rng.random() < bump_prob:
            bump_remaining = rng.integers(8, 25)  # 16-50s (bei dt=2)
            total = rng.uniform(*bump_strength)
            bump_per_step = total / bump_remaining

        co2_add = 0.0
        if bump_remaining > 0:
            co2_add = bump_per_step
            bump_remaining -= 1

        # Tür offen?
        door_open = door_open_at <= t < (door_open_at + door_duration)

        # Mischung Richtung Außenluft wenn offen
        mix = mix_when_open if door_open else 0.0

        # CO2 Dynamik:
        # - Quelle (Menschen) + bump
        # - leichte Leckage Richtung Außenluft
        # - starke Mischung bei offener Tür
        co2 = co2_true[t - 1]
        co2 += co2_source * dt  # ppm/s * s
        co2 += co2_add
        # Leckage / langsame Lüftung
        co2 += leak * (co2_out - co2)
        # Tür-Mischung
        co2 += mix * (co2_out - co2)

        # Temp Dynamik:
        temp = temp_true[t - 1]
        # leichte Relaxation Richtung setpoint
        temp += 0.01 * (temp_set - temp)
        # Leckage Richtung Außenluft
        temp += leak * (temp_out - temp)
        # Tür-Mischung (Temperatur reagiert oft schneller als CO2)
        temp += (mix * 1.5) * (temp_out - temp)

        co2_true[t] = co2
        temp_true[t] = temp

    # --- Sensor-Modell (was dein Gerät wirklich misst) ---
    # CO2 NDIR: typ. Rauschen ~ +/- (5-20 ppm) je nach Sensor/Filterung
    # Temp: typ. +/- 0.05..0.2°C Rauschen, oft quantisiert
    co2_noise_sigma = 8.0
    temp_noise_sigma = 0.07

    co2_meas = co2_true + rng.normal(0, co2_noise_sigma, size=steps)
    temp_meas = temp_true + rng.normal(0, temp_noise_sigma, size=steps)

    # Quantisierung (realistisch)
    # viele CO2 Sensoren liefern in 1 ppm steps (oder 5 ppm)
    co2_meas = np.round(co2_meas / 1.0) * 1.0
    # Temp z.B. 0.01 oder 0.1 °C steps
    temp_meas = np.round(temp_meas / 0.01) * 0.01

    # Optional: gelegentliche Mess-Ausreißer
    outlier_prob = 0.002
    outliers = rng.random(steps) < outlier_prob
    co2_meas[outliers] += rng.normal(0, 60, size=outliers.sum())

    return co2_meas, temp_meas, (door_open_at, door_open_at + door_duration)


def run_test():
    logger.info("--- Starte realistische Simulation ---")

    co2_data, temp_data, (t0, t1) = generate_realistic_data(
        steps=600, dt=2.0, door_open_at=140, door_duration=30, seed=7
    )

    detector = AdaptiveDoorDetector(window_size=30, sensitivity=4.0)

    results_door = []
    results_sigma_co2 = []
    results_sigma_temp = []

    for i in range(len(co2_data)):
        is_door, sigma_co2, sigma_temp = detector.update(co2_data[i], temp_data[i])
        results_door.append(is_door)
        results_sigma_co2.append(sigma_co2)
        results_sigma_temp.append(sigma_temp)

    # Plot
    _fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

    ax1.plot(co2_data, label="CO2 (ppm)")
    ax1.axvspan(t0, t1, alpha=0.15, label="Tür offen (Ground Truth)")
    ax1.set_ylabel("CO2 ppm")
    ax1.set_title("1) Realistische CO2 Messwerte")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(temp_data, label="Temp (°C)")
    ax2.axvspan(t0, t1, alpha=0.15)
    ax2.set_ylabel("°C")
    ax2.set_title("2) Realistische Temperatur Messwerte")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.plot(results_sigma_co2, label="Sigma CO2")
    ax3.axhline(y=-4.0, linestyle="--", label="Schwelle (-4.0)")
    ax3.axvspan(t0, t1, alpha=0.15)
    ax3.set_ylabel("Z-Score")
    ax3.set_title("3) Sigma/Abweichung (CO2)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4.plot(results_door, label="Erkannte Tür", linewidth=2)
    ax4.axvspan(t0, t1, alpha=0.15)
    ax4.set_ylim(-0.1, 1.1)
    ax4.set_ylabel("1=Offen")
    ax4.set_title("4) Entscheidung des Detektors")
    ax4.grid(True, alpha=0.3)
    ax4.fill_between(range(len(results_door)), results_door, alpha=0.2)
    ax4.legend()

    plt.xlabel(f"Zeitschritte (dt=2s)  |  Tür offen ca. t={t0}..{t1}")
    plt.tight_layout()
    logger.info("Simulation beendet. Zeige Plot...")
    plt.show()


if __name__ == "__main__":
    run_test()
