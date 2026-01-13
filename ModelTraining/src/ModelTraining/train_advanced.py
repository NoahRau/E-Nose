import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# WICHTIG: Stelle sicher, dass Python das Modul findet
# export PYTHONPATH=$PYTHONPATH:.  (im Terminal ausführen)
from ModelTraining.dataset import FridgeDataset
from ModelTraining.losses import DINOLoss, KoLeoLoss
from ModelTraining.model_advanced import FridgeMoCA_Pro

logger = logging.getLogger(__name__)

# --- Configuration ---
EPOCHS = 100
LR = 0.0005
MOMENTUM_TEACHER = 0.996
LAMBDA_DINO = 1.0
LAMBDA_IBOT = 1.0
LAMBDA_KOLEO = 0.1
BATCH_SIZE = 32
SEQ_LEN = 512
LOG_INTERVAL = 100      # Zeige alle 100 Batches einen Status an

# Pfad zum Daten-Ordner (nicht einzelne Datei)
CSV_DIR = Path("Data")
CHECKPOINT_DIR = Path("checkpoints")

def setup_logging(log_file=None, level=logging.INFO):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)

def update_teacher(student, teacher, momentum):
    """EMA Update: Teacher = momentum * Teacher + (1-momentum) * Student"""
    with torch.no_grad():
        for param_q, param_k in zip(student.parameters(), teacher.parameters()):
            param_k.data.mul_(momentum).add_((1 - momentum) * param_q.data)

def main():
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(log_file=f"train_dino_{timestamp_str}.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dataset initialisieren
    if not CSV_DIR.exists():
        logger.error(f"Datenordner nicht gefunden: {CSV_DIR}")
        return

    logger.info(f"Initialisiere Dataset aus Ordner: {CSV_DIR}")
    dataset = FridgeDataset(CSV_DIR, seq_len=SEQ_LEN, mode="train", scaler_path=CHECKPOINT_DIR / "scaler.pkl")

    # Kanäle dynamisch auslesen
    num_gas_channels = dataset.gas_data.shape[0]
    num_env_channels = dataset.env_data.shape[0]
    logger.info(f"Erkannte Kanäle -> Gas: {num_gas_channels}, Env: {num_env_channels}")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
    logger.info(f"Dataset Größe: {len(dataset)} Samples | Batches pro Epoche: {len(dataloader)}")

    # 2. Modell Setup mit korrekten Kanälen
    logger.info("Initialisiere Modelle...")

    student = FridgeMoCA_Pro(
        gas_chans=num_gas_channels,
        env_chans=num_env_channels,
        seq_len=SEQ_LEN
    ).to(device)

    teacher = FridgeMoCA_Pro(
        gas_chans=num_gas_channels,
        env_chans=num_env_channels,
        seq_len=SEQ_LEN
    ).to(device)

    # Teacher startet als Kopie des Student
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    # 3. Optimizer & Loss
    dino_loss_fn = DINOLoss(out_dim=4096, nepochs=EPOCHS).to(device)
    koleo_loss_fn = KoLeoLoss().to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.04)

    logger.info("Starte Training...")

    for epoch in range(EPOCHS):
        student.train()
        total_loss = 0
        total_dino = 0
        total_ibot = 0
        total_koleo = 0

        for batch_idx, (gas, env) in enumerate(dataloader):
            gas, env = gas.to(device), env.to(device)

            # Teacher Forward (Global View)
            with torch.no_grad():
                t_dino, t_ibot, _ = teacher(gas, env)

            # Student Forward (Masked/Local View)
            s_dino, s_ibot, s_cls_feat = student(gas, env)

            # Loss Berechnung
            l_dino = dino_loss_fn(s_dino, t_dino, epoch, is_ibot=False)

            # Reshape für iBOT Loss: [Batch, Patches, Dim] -> [Batch*Patches, Dim]
            l_ibot = dino_loss_fn(
                s_ibot.reshape(-1, 4096),
                t_ibot.reshape(-1, 4096),
                epoch,
                is_ibot=True
            )
            l_koleo = koleo_loss_fn(s_cls_feat)

            loss = (LAMBDA_DINO * l_dino) + (LAMBDA_IBOT * l_ibot) + (LAMBDA_KOLEO * l_koleo)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Teacher EMA Update
            update_teacher(student, teacher, MOMENTUM_TEACHER)

            # Logging Stats
            total_loss += loss.item()
            total_dino += l_dino.item()
            total_ibot += l_ibot.item()
            total_koleo += l_koleo.item()

            # --- LOGGER HIER EINGEFÜGT ---
            if batch_idx % LOG_INTERVAL == 0:
                progress = (batch_idx / len(dataloader)) * 100
                logger.info(
                    f"Epoch {epoch+1}/{EPOCHS} [{batch_idx}/{len(dataloader)}] ({progress:.1f}%) "
                    f"| Loss: {loss.item():.4f} "
                    f"| DINO: {l_dino.item():.4f} iBOT: {l_ibot.item():.4f} KoLeo: {l_koleo.item():.4f}"
                )
            # -----------------------------

        # Epoch Summary
        avg_loss = total_loss / len(dataloader)
        avg_dino = total_dino / len(dataloader)
        avg_ibot = total_ibot / len(dataloader)

        logger.info(f"==> Epoch {epoch+1}/{EPOCHS} Summary | Avg Loss: {avg_loss:.4f} (DINO: {avg_dino:.4f}, iBOT: {avg_ibot:.4f})")

        # Checkpoints speichern
        if (epoch + 1) % 10 == 0:
            save_path = CHECKPOINT_DIR / f"checkpoint_ep{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            logger.info(f"Checkpoint gespeichert: {save_path}")

    # Finales Modell speichern
    final_path = CHECKPOINT_DIR / "fridge_moca_pro_final.pth"
    torch.save(student.state_dict(), final_path)
    logger.info(f"Training fertig! Modell gespeichert unter: {final_path}")

if __name__ == "__main__":
    main()