import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

# Ensure Python finds your modules
from ModelTraining.dataset import FridgeDataset
# IMPORTANT: GramLoss must exist in losses.py
from ModelTraining.losses import DINOLoss, KoLeoLoss, GramLoss
from ModelTraining.model_advanced import FridgeMoCA_V3

logger = logging.getLogger(__name__)

# --- Configuration (Defaults) ---
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
LR = 0.0005
MOMENTUM_TEACHER = 0.996
SEQ_LEN = 512

# Loss Weighting
LAMBDA_DINO = 1.0
LAMBDA_IBOT = 1.0
LAMBDA_KOLEO = 0.1
LAMBDA_GRAM = 0.5

LOG_INTERVAL = 20
CSV_DIR = Path("Data")
CHECKPOINT_DIR = Path("checkpoints")


def parse_args():
    parser = argparse.ArgumentParser(description="E-Nose Advanced Training (FridgeMoCA V3)")
    parser.add_argument("--logfile", type=str, help="Path to log file.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"Number of epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--no-console", action="store_true", help="Disable console logging")
    return parser.parse_args()


def setup_logging(log_file=None, console=True, level=logging.INFO):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)


def update_teacher(student, teacher, momentum):
    """Exponential Moving Average (EMA) Update for the Teacher"""
    with torch.no_grad():
        for param_q, param_k in zip(student.parameters(), teacher.parameters()):
            param_k.data.mul_(momentum).add_((1 - momentum) * param_q.data)


def main():
    args = parse_args()

    # Logging Setup
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = args.logfile if args.logfile else f"train_foundation_v3_{timestamp_str}.log"
    setup_logging(log_file=log_filename, console=not args.no_console)

    logger.info(f"Arguments: {args}")

    # Params
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Start Training on Device: {device}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dataset
    if not CSV_DIR.exists():
        logger.error(f"Data folder not found: {CSV_DIR}")
        return

    logger.info(f"Initializing Dataset from: {CSV_DIR}")
    dataset = FridgeDataset(CSV_DIR, seq_len=SEQ_LEN, mode="train", scaler_path=CHECKPOINT_DIR / "scaler.pkl")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)

    gas_chans = dataset.gas_data.shape[0]
    env_chans = dataset.env_data.shape[0]
    logger.info(f"Detected Channels -> Gas: {gas_chans}, Env: {env_chans}")

    # 2. Models (V3)
    logger.info("Initializing FridgeMoCA_V3...")
    student = FridgeMoCA_V3(gas_chans=gas_chans, env_chans=env_chans, seq_len=SEQ_LEN).to(device)
    teacher = FridgeMoCA_V3(gas_chans=gas_chans, env_chans=env_chans, seq_len=SEQ_LEN).to(device)

    # Initialize Teacher with Student Weights & Freeze
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    # 3. Losses & Optimizer
    dino_loss_fn = DINOLoss(out_dim=4096, nepochs=EPOCHS).to(device)
    koleo_loss_fn = KoLeoLoss().to(device)
    gram_loss_fn = GramLoss().to(device)

    optimizer = optim.AdamW(student.parameters(), lr=LR, weight_decay=0.04)

    logger.info(f"Starting Foundation Model Training (Phase 1) for {EPOCHS} Epochs...")

    for epoch in range(EPOCHS):
        student.train()
        metrics = {"loss": 0.0, "dino": 0.0, "ibot": 0.0, "koleo": 0.0, "gram": 0.0}

        for batch_idx, (gas, env) in enumerate(dataloader):
            gas, env = gas.to(device), env.to(device)

            # Teacher: Sees EVERYTHING (mask_ratio=0.0)
            with torch.no_grad():
                t_dino, t_ibot, _, t_patch_feat = teacher(gas, env, mask_ratio=0.0)

            # Student: Sees ONLY 50% (mask_ratio=0.5)
            s_dino, s_ibot, s_cls_feat, s_patch_feat = student(gas, env, mask_ratio=0.5)

            # --- LOSS CALCULATION ---
            # 1. DINO Loss (Global View on CLS Token)
            l_dino = dino_loss_fn(s_dino, t_dino, epoch, is_ibot=False)

            # 2. iBOT Loss (Local View on Patches)
            l_ibot = dino_loss_fn(s_ibot.reshape(-1, 4096), t_ibot.reshape(-1, 4096), epoch, is_ibot=True)

            # 3. KoLeo Loss (Uniformity in Feature Space)
            s_cls_norm = F.normalize(s_cls_feat, dim=-1, p=2)
            l_koleo = koleo_loss_fn(s_cls_norm)

            # 4. Gram Anchoring (Structure/Covariance Preservation)
            # IMPORTANT: Gram matrix helps the student reconstruct the correlations of the MISSING parts
            # Since s_patch_feat is shorter (masked), and t_patch_feat is full,
            # we can't compare them 1:1. Gram compares the *covariance matrix* (size Dim x Dim).
            # This works regardless of sequence length!
            l_gram = gram_loss_fn(s_patch_feat, t_patch_feat)

            # Total Loss
            loss = (LAMBDA_DINO * l_dino) + \
                   (LAMBDA_IBOT * l_ibot) + \
                   (LAMBDA_KOLEO * l_koleo) + \
                   (LAMBDA_GRAM * l_gram)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
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
                progress = (batch_idx / len(dataloader)) * 100
                logger.info(
                    f"Ep {epoch+1}/{EPOCHS} [{batch_idx}/{len(dataloader)}] ({progress:.1f}%) "
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
            logger.info(f"Checkpoint saved: {save_path}")

    # Final Save
    final_path = CHECKPOINT_DIR / "fridge_moca_v3_foundation.pth"
    torch.save(student.state_dict(), final_path)
    logger.info(f"Training complete! Foundation Model saved to: {final_path}")

if __name__ == "__main__":
    main()