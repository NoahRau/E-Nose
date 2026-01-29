import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.dates as mdates

# --- KONFIGURATION ---
DATA_FILE = Path("Data/all_predictions_combined.csv")
OUTPUT_IMAGE = Path("Data/evaluation_results.png")

def main():
    if not DATA_FILE.exists():
        print(f"FEHLER: Datei {DATA_FILE} nicht gefunden.")
        print("Bitte führe erst das 'run_inference.py' Skript aus, um die Vorhersagen zu generieren.")
        return

    print(f"Lade Daten aus {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)

    # Versuche, die Zeitspalte zu lesen
    time_col = None
    if 'datetime' in df.columns:
        # Datetime parsen
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        time_col = 'datetime'
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        time_col = 'timestamp'
    else:
        # Fallback auf Index
        df['index'] = df.index
        time_col = 'index'

    # Welche Dateien haben wir?
    files = df['source_file'].unique()
    n_files = len(files)

    print(f"Gefundene Dateien: {n_files}")

    # Plot Setup (Ein Subplot pro Datei)
    fig, axes = plt.subplots(n_files, 1, figsize=(15, 5 * n_files), sharey=True)
    if n_files == 1: axes = [axes] # Falls nur 1 Datei, in Liste packen

    sns.set_theme(style="whitegrid")

    for i, file_name in enumerate(files):
        ax = axes[i]
        subset = df[df['source_file'] == file_name].copy()

        # Plotten
        sns.lineplot(data=subset, x=time_col, y="ai_mold_prob", ax=ax, linewidth=2, color="#3498db")

        # Schwellenwerte einzeichnen
        ax.axhline(0.8, color='red', linestyle='--', linewidth=1.5, label='Schimmel (>0.8)')
        ax.axhline(0.5, color='orange', linestyle='--', linewidth=1.5, label='Verdacht (>0.5)')

        # Fülle den Bereich unter der Kurve rot, wenn er hoch ist
        # (Trick: Wir füllen alles über 0.5 leicht orange und über 0.8 rot)
        if time_col == 'datetime':
            # Fill_between funktioniert am besten mit numerischen x-Werten oder sortierten Datetimes
            subset = subset.sort_values(by=time_col)

        ax.set_title(f"Datei: {file_name}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Schimmel-Wahrscheinlichkeit (0-1)")
        ax.set_ylim(-0.05, 1.1)
        ax.legend(loc='upper right')

        # X-Achse formatieren (falls Zeit)
        if time_col == 'datetime':
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    plt.tight_layout()

    print(f"Speichere Grafik unter: {OUTPUT_IMAGE}")
    plt.savefig(OUTPUT_IMAGE)
    print("✅ Fertig! Grafik erstellt.")

if __name__ == "__main__":
    main()