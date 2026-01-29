import sys
import pandas as pd
import numpy as np
import torch
import joblib
import torch.nn.functional as F
from pathlib import Path

# --- KONFIGURATION ---
DATA_DIR = Path("Data")
CHECKPOINT_DIR = Path("checkpoints")
FOUNDATION_PATH = CHECKPOINT_DIR / "fridge_moca_v3_foundation.pth"
HEAD_PATH = CHECKPOINT_DIR / "mold_classifier_head.pth"
SCALER_PATH = CHECKPOINT_DIR / "scaler.pkl"

TARGET_FILES = [
    "data_LeerSchrankv2_20260113_202227.csv",
    "data_Mandarinenimkühlschrank_20260117_221035.csv",
    "data_Mandarineschlecht_20260122_173422.csv"
]

SEQ_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import Setup
sys.path.append(str(Path("ModelTraining/src").absolute()))

try:
    from ModelTraining.model_advanced import FridgeMoCA_V3, MoldClassifierHead
except ImportError:
    print("Fehler: Konnte 'ModelTraining' nicht importieren.")
    sys.exit(1)

def load_models():
    print(f"--- Lade Modelle ---")
    scalers = joblib.load(SCALER_PATH)
    scaler_gas = scalers["gas"]
    scaler_env = scalers["env"]

    gas_chans = scaler_gas.mean_.shape[0]
    env_chans = scaler_env.mean_.shape[0]

    foundation = FridgeMoCA_V3(gas_chans=gas_chans, env_chans=env_chans, seq_len=SEQ_LEN).to(DEVICE)
    foundation.load_state_dict(torch.load(FOUNDATION_PATH, map_location=DEVICE))
    foundation.eval()

    classifier = MoldClassifierHead(embed_dim=384, num_classes=2).to(DEVICE)
    classifier.load_state_dict(torch.load(HEAD_PATH, map_location=DEVICE))
    classifier.eval()
    return scaler_gas, scaler_env, foundation, classifier

def drop_bad_rows(df, filename):
    """Löscht Zeilen mit fehlerhaften Sensordaten"""
    initial_count = len(df)
    print(f"  🧹 Bereinige Daten für {filename} (Start: {initial_count} Zeilen)...")

    # 1. Spalten finden und umbenennen
    col_map = {
        "scd_co2": ["scd_co2", "scd_c"], "scd_temp": ["scd_temp", "scd_t"],
        "scd_hum": ["scd_hum", "scd_h"], "bme_temp": ["bme_temp", "bme_t"],
        "bme_hum": ["bme_hum", "bme_h"], "bme_pres": ["bme_pres", "bme_p"],
        "bme_gas": ["bme_gas", "bme_g"]
    }

    for target, options in col_map.items():
        found = False
        for opt in options:
            if opt in df.columns:
                df.rename(columns={opt: target}, inplace=True)
                found = True
                break
        if not found:
            # Wenn eine Spalte ganz fehlt, füllen wir sie mit NaN (damit wir sie gleich droppen können)
            df[target] = np.nan

    relevant_cols = ["scd_co2", "scd_temp", "scd_hum", "bme_temp", "bme_hum", "bme_pres", "bme_gas"]

    # 2. Drop NaN (Leere Felder)
    df.dropna(subset=relevant_cols, inplace=True)

    # 3. Drop Unmögliche Werte (Filter basierend auf deiner Log-Datei)
    # CO2 muss > 100 sein (0.0 ist Fehler)
    # Luftfeuchte muss <= 100 sein (489.0 ist Fehler)
    mask_valid = (
            (df["scd_co2"] > 100) &
            (df["scd_hum"] >= 0) &
            (df["scd_hum"] <= 100)
    )
    df = df[mask_valid]

    final_count = len(df)
    dropped = initial_count - final_count

    if dropped > 0:
        print(f"    🗑️ {dropped} fehlerhafte Zeilen gelöscht ({(dropped/initial_count)*100:.1f}%).")
    else:
        print("    ✨ Keine fehlerhaften Zeilen gefunden.")

    return df

def process_file(filename, scaler_gas, scaler_env, foundation, classifier):
    file_path = DATA_DIR / filename
    if not file_path.exists():
        return None

    df = pd.read_csv(file_path)

    # BEREINIGUNG (RAUSKICKEN)
    df = drop_bad_rows(df, filename)

    if len(df) < SEQ_LEN:
        print(f"    ❌ Zu wenig Daten übrig nach Bereinigung ({len(df)} < {SEQ_LEN}). Überspringe Datei.")
        return None

    # Features
    env_cols = ["scd_co2", "scd_temp", "scd_hum", "bme_temp", "bme_hum", "bme_pres"]
    gas_cols = ["bme_gas"]

    gas_raw = df[gas_cols].values.astype(np.float32)
    env_raw = df[env_cols].values.astype(np.float32)

    # Skalieren
    gas_norm = scaler_gas.transform(gas_raw)
    env_norm = scaler_env.transform(env_raw)

    probs, preds = [], []

    print(f"  🧠 Starte Inference ({len(df)} Zeilen)...")

    with torch.no_grad():
        for i in range(len(df)):
            start = max(0, i - SEQ_LEN + 1)
            g_win = gas_norm[start : i+1]
            e_win = env_norm[start : i+1]

            # Padding am Anfang
            if len(g_win) < SEQ_LEN:
                pad = SEQ_LEN - len(g_win)
                g_win = np.pad(g_win, ((pad, 0), (0, 0)), mode='edge')
                e_win = np.pad(e_win, ((pad, 0), (0, 0)), mode='edge')

            g_t = torch.tensor(g_win).transpose(0, 1).unsqueeze(0).to(DEVICE).float()
            e_t = torch.tensor(e_win).transpose(0, 1).unsqueeze(0).to(DEVICE).float()

            features = foundation.forward_features(g_t, e_t)
            logits = classifier(features[:, 0, :])
            mold_prob = F.softmax(logits, dim=1)[0, 1].item()

            probs.append(mold_prob)

            if mold_prob > 0.8: preds.append("SCHIMMEL")
            elif mold_prob > 0.5: preds.append("VERDACHT")
            else: preds.append("OK")

            if i % 5000 == 0 and i > 0: print(f"\r    {i}/{len(df)}", end="")

    print("\r    Fertig.           ")

    df["ai_mold_prob"] = probs
    df["ai_prediction"] = preds
    df["source_file"] = filename

    return df

def main():
    scaler_gas, scaler_env, foundation, classifier = load_models()
    all_dfs = []

    for fname in TARGET_FILES:
        df = process_file(fname, scaler_gas, scaler_env, foundation, classifier)
        if df is not None:
            all_dfs.append(df)

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        out_path = DATA_DIR / "all_predictions_combined.csv"
        full_df.to_csv(out_path, index=False)
        print(f"\n✅ Datei gespeichert: {out_path}")
        print("   (Du kannst jetzt wieder das Plot-Skript ausführen)")

if __name__ == "__main__":
    main()