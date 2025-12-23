import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# Eigene Module importieren
from model import FridgeMoCA     # Deine Architektur (aus der Datei model.py)
from dataset import FridgeDataset # Dein DataLoader (aus der Datei dataset.py)

# --- KONFIGURATION ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 20
SEQ_LEN = 512    # Fenstergröße (bei 2s Takt = 17 Min Kontext)
CSV_PATH = "data/deine_daten.csv" # <--- HIER DEINE DATEI ANGEBEN
CHECKPOINT_DIR = "checkpoints"

def main():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training auf: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 2. Daten laden
    # Falls noch keine echte CSV da ist, meckern
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Datei {CSV_PATH} nicht gefunden! Bitte Daten in den Ordner kopieren.")
        return

    dataset = FridgeDataset(CSV_PATH, seq_len=SEQ_LEN, mode='train')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # workers=0 für Windows/Mac safe

    # 3. Modell instanzieren
    model = FridgeMoCA(
        seq_len=SEQ_LEN,
        patch_size=16,
        gas_chans=10,
        env_chans=3,
        embed_dim=128,
        depth=4
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("Modell initialisiert. Starte Training...")

    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch_idx, (gas, env) in enumerate(dataloader):
            gas, env = gas.to(device), env.to(device)

            # Forward Pass (Modell berechnet Loss selbst, siehe model.py)
            optimizer.zero_grad()
            loss, _, _ = model(gas, env)

            # Backward Pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Batch {batch_idx}: Loss = {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"=== Epoch {epoch+1} beendet. Avg Loss: {avg_loss:.4f} ===")

        # 5. Speichern (Checkpointing)
        # Wir speichern nur, wenn es besser wird (oder immer am Ende)
        save_path = os.path.join(CHECKPOINT_DIR, f"fridge_moca_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, save_path)
        print(f"Modell gespeichert: {save_path}")

    print("Training abgeschlossen! 🚀")

if __name__ == "__main__":
    main()