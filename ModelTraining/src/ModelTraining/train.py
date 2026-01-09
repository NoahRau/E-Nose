import logging
import os
import sys
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from ModelTraining.dataset import FridgeDataset
from ModelTraining.model import FridgeMoCA

logger = logging.getLogger(__name__)

# --- Configuration ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 20
SEQ_LEN = 512  # Window size (at 2s interval = ~17 min context)
CSV_PATH = "data/deine_daten.csv"
CHECKPOINT_DIR = "checkpoints"


def setup_logging(log_file: str | None = None, level: int = logging.INFO) -> None:
    """Configure logging for the training module.

    Args:
        log_file: Optional path to log file. If None, logs only to console.
        level: Logging level (default: INFO).
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)


def main() -> None:
    """Main entry point for FridgeMoCA MAE training."""
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"train_mae_{timestamp_str}.log"
    setup_logging(log_file=log_filename, level=logging.DEBUG)

    logger.info("=" * 50)
    logger.info("FridgeMoCA MAE Training")
    logger.info("=" * 50)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    logger.info("Checkpoint directory: %s", CHECKPOINT_DIR)

    # Load data
    if not os.path.exists(CSV_PATH):
        logger.error("Data file not found: %s", CSV_PATH)
        return

    logger.info("Loading dataset from: %s", CSV_PATH)
    dataset = FridgeDataset(CSV_PATH, seq_len=SEQ_LEN, mode="train")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    logger.info("Dataset size: %d samples", len(dataset))
    logger.info("Batches per epoch: %d", len(dataloader))

    # Initialize model
    model = FridgeMoCA(
        seq_len=SEQ_LEN,
        patch_size=16,
        gas_chans=10,
        env_chans=3,
        embed_dim=128,
        depth=4,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model parameters: %d (%.2f MB)", num_params, num_params * 4 / 1024 / 1024
    )

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    logger.info("Optimizer: AdamW (lr=%.0e)", LEARNING_RATE)

    logger.info("-" * 50)
    logger.info("Starting training for %d epochs...", EPOCHS)
    logger.info("-" * 50)

    best_loss = float("inf")

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_idx, (gas, env) in enumerate(dataloader):
            gas, env = gas.to(device), env.to(device)

            optimizer.zero_grad()
            loss, _, _ = model(gas, env)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.debug(
                    "Epoch [%d/%d] Batch %d/%d: Loss = %.4f",
                    epoch + 1,
                    EPOCHS,
                    batch_idx,
                    len(dataloader),
                    loss.item(),
                )

        avg_loss = total_loss / len(dataloader)
        logger.info(
            "Epoch %d/%d completed | Avg Loss: %.4f", epoch + 1, EPOCHS, avg_loss
        )

        # Save checkpoint
        save_path = os.path.join(CHECKPOINT_DIR, f"fridge_moca_epoch_{epoch + 1}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            },
            save_path,
        )
        logger.debug("Checkpoint saved: %s", save_path)

        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(CHECKPOINT_DIR, "fridge_moca_best.pth")
            torch.save(model.state_dict(), best_path)
            logger.info("New best model saved (loss: %.4f)", best_loss)

    logger.info("=" * 50)
    logger.info("Training completed!")
    logger.info("Best loss: %.4f", best_loss)
    logger.info("Log saved to: %s", log_filename)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
