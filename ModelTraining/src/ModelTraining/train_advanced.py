import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

# Stelle sicher, dass Python die Module findet
from ModelTraining.dataset import FridgeDataset
# HIER: GramLoss muss in losses.py existieren!
from ModelTraining.losses import DINOLoss, KoLeoLoss, GramLoss
from ModelTraining.model_advanced import FridgeMoCA_V3

logger = logging.getLogger(__name__)

# --- Configuration ---
EPOCHS = 100
LR = 0.0005
MOMENTUM_TEACHER = 0.996
BATCH_SIZE = 32
SEQ_LEN = 512

# Loss Gewichtungen (Balancing ist Key für V3)
LAMBDA_DINO = 1.0
LAMBDA_IBOT = 1.0
LAMBDA_KOLEO = 0.1
LAMBDA_GRAM = 0.5       # Gram Anchoring Stärke

LOG_INTERVAL = 20
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
    """Exponential Moving Average (EMA) Update für den Teacher"""
    with torch.no_grad():
        for param_q, param_k in zip(student.parameters(), teacher.parameters()):
            param_k.data.mul_(momentum).add_((1 - momentum) * param_q.data)

def main():
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(log_file=f"train_foundation_v3_{timestamp_str}.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Start Training auf Device: {device}")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dataset
    if not CSV_DIR.exists():
        logger.error(f"Datenordner nicht gefunden: {CSV_DIR}")
        return

    # Scaler wird gespeichert, damit wir ihn für Phase 2 wieder laden können!
    dataset = FridgeDataset(CSV_DIR, seq_len=SEQ_LEN, mode="train", scaler_path=CHECKPOINT_DIR / "scaler.pkl")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)

    gas_chans = dataset.gas_data.shape[0]
    env_chans = dataset.env_data.shape[0]

    # 2. Modelle (V3)
    logger.info(f"Initialisiere FridgeMoCA_V3 (Gas: {gas_chans}, Env: {env_chans})...")

    student = FridgeMoCA_V3(gas_chans=gas_chans, env_chans=env_chans, seq_len=SEQ_LEN).to(device)
    teacher = FridgeMoCA_V3(gas_chans=gas_chans, env_chans=env_chans, seq_len=SEQ_LEN).to(device)

    # Teacher mit Student Weights initialisieren & einfrieren
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    # 3. Losses & Optimizer
    dino_loss_fn = DINOLoss(out_dim=4096, nepochs=EPOCHS).to(device)
    koleo_loss_fn = KoLeoLoss().to(device)
    gram_loss_fn = GramLoss().to(device) # Gram Anchoring

    optimizer = optim.AdamW(student.parameters(), lr=LR, weight_decay=0.04)

    logger.info("Starte Foundation Model Training (Phase 1)...")

    for epoch in range(EPOCHS):
        student.train()
        metrics = {"loss": 0.0, "dino": 0.0, "ibot": 0.0, "koleo": 0.0, "gram": 0.0}

        for batch_idx, (gas, env) in enumerate(dataloader):
            gas, env = gas.to(device), env.to(device)

            # Teacher Forward (ohne Gradient)
            with torch.no_grad():
                t_dino, t_ibot, _, t_patch_feat = teacher(gas, env)

            # Student Forward
            s_dino, s_ibot, s_cls_feat, s_patch_feat = student(gas, env)

            # --- LOSS BERECHNUNG ---
            # 1. DINO Loss (Global View auf CLS Token)
            l_dino = dino_loss_fn(s_dino, t_dino, epoch, is_ibot=False)

            # 2. iBOT Loss (Local View auf Patches)
            l_ibot = dino_loss_fn(s_ibot.reshape(-1, 4096), t_ibot.reshape(-1, 4096), epoch, is_ibot=True)

            # 3. KoLeo Loss (Uniformity im Feature Space)
            s_cls_norm = F.normalize(s_cls_feat, dim=-1, p=2)
            l_koleo = koleo_loss_fn(s_cls_norm)

            # 4. Gram Anchoring (Struktur/Kovarianz Erhaltung)
            l_gram = gram_loss_fn(s_patch_feat, t_patch_feat)

            # Total Loss
            loss = (LAMBDA_DINO * l_dino) + \
                   (LAMBDA_IBOT * l_ibot) + \
                   (LAMBDA_KOLEO * l_koleo) + \
                   (LAMBDA_GRAM * l_gram)

            # Backprop
            optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping (Sicherheit)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0)

            optimizer.step()

            # EMA Update Teacher
            update_teacher(student, teacher, MOMENTUM_TEACHER)

            # Metrics
            metrics["loss"] += loss.item()
            metrics["dino"] += l_dino.item()
            metrics["ibot"] += l_ibot.item()
            metrics["koleo"] += l_koleo.item()
            metrics["gram"] += l_gram.item()

            if batch_idx % LOG_INTERVAL == 0:
                logger.info(
                    f"Ep {epoch+1} [{batch_idx}/{len(dataloader)}] "
                    f"L: {loss.item():.4f} | "
                    f"D: {l_dino.item():.3f} i: {l_ibot.item():.3f} K: {l_koleo.item():.3f} G: {l_gram.item():.3f}"
                )

        # Epoch Log
        avg_metrics = {k: v / len(dataloader) for k, v in metrics.items()}
        logger.info(f"==> End Ep {epoch+1} | Avg Loss: {avg_metrics['loss']:.4f}")

        # Checkpoints
        if (epoch + 1) % 10 == 0:
            save_path = CHECKPOINT_DIR / f"checkpoint_v3_ep{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'student': student.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, save_path)
            logger.info(f"Checkpoint gesichert: {save_path}")

    # Final Save (Das ist dein Foundation Model für Phase 2!)
    final_path = CHECKPOINT_DIR / "fridge_moca_v3_foundation.pth"
    torch.save(student.state_dict(), final_path)
    logger.info(f"Training fertig! Foundation Model gespeichert: {final_path}")

if __name__ == "__main__":
    main()