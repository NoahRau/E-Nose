import logging
import os
import sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from ModelTraining.dataset import FridgeDataset
from ModelTraining.losses import DINOLoss, KoLeoLoss
from ModelTraining.model_advanced import FridgeMoCA_Pro

logger = logging.getLogger(__name__)

# --- Configuration ---
EPOCHS = 50
LR = 0.0005
MOMENTUM_TEACHER = 0.996  # Teacher learns slowly from Student via EMA
LAMBDA_DINO = 1.0
LAMBDA_IBOT = 1.0
LAMBDA_KOLEO = 0.1
BATCH_SIZE = 32
SEQ_LEN = 512
CSV_PATH = "data/deine_daten.csv"
CHECKPOINT_DIR = "checkpoints"


def setup_logging(log_file: str | None = None, level: int = logging.INFO) -> None:
    """Configure logging for the training module."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)


def update_teacher(student: torch.nn.Module, teacher: torch.nn.Module, momentum: float) -> None:
    """EMA Update: Teacher = momentum * Teacher + (1-momentum) * Student"""
    with torch.no_grad():
        for param_q, param_k in zip(student.parameters(), teacher.parameters()):
            param_k.data.mul_(momentum).add_((1 - momentum) * param_q.data)


def main() -> None:
    """Main entry point for FridgeMoCA_Pro DINOv2 + iBOT training."""
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"train_dino_{timestamp_str}.log"
    setup_logging(log_file=log_filename, level=logging.DEBUG)

    logger.info("=" * 50)
    logger.info("FridgeMoCA Pro - DINOv2 + iBOT Training")
    logger.info("=" * 50)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("GPU Memory: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load data
    if not os.path.exists(CSV_PATH):
        logger.error("Data file not found: %s", CSV_PATH)
        return

    logger.info("Loading dataset from: %s", CSV_PATH)
    dataset = FridgeDataset(CSV_PATH, seq_len=SEQ_LEN, mode="train")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    logger.info("Dataset size: %d samples", len(dataset))
    logger.info("Batches per epoch: %d", len(dataloader))

    # Initialize Student & Teacher models
    logger.info("Initializing Student and Teacher models...")
    student = FridgeMoCA_Pro().to(device)
    teacher = FridgeMoCA_Pro().to(device)

    # Teacher starts identical to Student, no gradients needed
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    num_params = sum(p.numel() for p in student.parameters())
    logger.info("Model parameters: %d (%.2f MB)", num_params, num_params * 4 / 1024 / 1024)

    # Loss functions & Optimizer
    dino_loss_fn = DINOLoss(out_dim=4096, nepochs=EPOCHS).to(device)
    koleo_loss_fn = KoLeoLoss().to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR)

    logger.info("Loss weights: DINO=%.1f, iBOT=%.1f, KoLeo=%.2f", LAMBDA_DINO, LAMBDA_IBOT, LAMBDA_KOLEO)
    logger.info("Teacher momentum: %.4f", MOMENTUM_TEACHER)
    logger.info("Optimizer: AdamW (lr=%.4f)", LR)

    logger.info("-" * 50)
    logger.info("Starting DINOv2 + iBOT training for %d epochs...", EPOCHS)
    logger.info("-" * 50)

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        student.train()
        total_loss = 0.0
        total_dino = 0.0
        total_ibot = 0.0
        total_koleo = 0.0

        for batch_idx, (gas, env) in enumerate(dataloader):
            gas, env = gas.to(device), env.to(device)

            # Teacher forward (Global View, Unmasked)
            with torch.no_grad():
                t_dino, t_ibot, _ = teacher(gas, env)

            # Student forward (Local/Masked View)
            s_dino, s_ibot, s_cls_feat = student(gas, env)

            # Loss computation
            l_dino = dino_loss_fn(s_dino, t_dino, epoch, is_ibot=False)
            l_ibot = dino_loss_fn(
                s_ibot.reshape(-1, 4096), t_ibot.reshape(-1, 4096), epoch, is_ibot=True
            )
            l_koleo = koleo_loss_fn(s_cls_feat)

            loss = (LAMBDA_DINO * l_dino) + (LAMBDA_IBOT * l_ibot) + (LAMBDA_KOLEO * l_koleo)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Teacher EMA update
            update_teacher(student, teacher, MOMENTUM_TEACHER)

            total_loss += loss.item()
            total_dino += l_dino.item()
            total_ibot += l_ibot.item()
            total_koleo += l_koleo.item()

            if batch_idx % 10 == 0:
                logger.debug(
                    "Epoch [%d/%d] Batch %d/%d: Loss=%.4f (DINO=%.4f, iBOT=%.4f, KoLeo=%.4f)",
                    epoch + 1, EPOCHS, batch_idx, len(dataloader),
                    loss.item(), l_dino.item(), l_ibot.item(), l_koleo.item()
                )

        # Epoch summary
        n_batches = len(dataloader)
        avg_loss = total_loss / n_batches
        avg_dino = total_dino / n_batches
        avg_ibot = total_ibot / n_batches
        avg_koleo = total_koleo / n_batches

        logger.info(
            "Epoch %d/%d | Loss: %.4f (DINO: %.4f, iBOT: %.4f, KoLeo: %.4f)",
            epoch + 1, EPOCHS, avg_loss, avg_dino, avg_ibot, avg_koleo
        )

        # Save checkpoint
        save_path = os.path.join(CHECKPOINT_DIR, f"fridge_moca_pro_epoch_{epoch + 1}.pth")
        torch.save(
            {
                "epoch": epoch,
                "student_state_dict": student.state_dict(),
                "teacher_state_dict": teacher.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            },
            save_path,
        )
        logger.debug("Checkpoint saved: %s", save_path)

        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(CHECKPOINT_DIR, "fridge_moca_pro_best.pth")
            torch.save(student.state_dict(), best_path)
            logger.info("New best model saved (loss: %.4f)", best_loss)

    logger.info("=" * 50)
    logger.info("Training completed!")
    logger.info("Best loss: %.4f", best_loss)
    logger.info("Log saved to: %s", log_filename)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
