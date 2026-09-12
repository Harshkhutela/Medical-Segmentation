import copy
import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import OUTPUT_PATH, CHECKPOINT_PATH
from configs.dataset_config import get_dataset_configuration
from models import UNet
from utils.advanced_metrics import evaluate_loader, write_metrics_csv
from utils.dataset_splitter import CachedSegmentationDataset, DatasetSplitManager
from utils.losses import BCEDiceLoss
from utils.training_helpers import HistoryLogger
from evaluation import generate_experiment_plots


def main():
    config = get_dataset_configuration()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Optimized Training on Device: {device}")

    split_manager = DatasetSplitManager(config)
    split_paths = split_manager.ensure_split_layout()

    train_dataset = CachedSegmentationDataset(
        image_folder=split_paths.train_images,
        mask_folder=split_paths.train_masks,
        image_size=config.image_size,
        cache_images=False,
        augment=True,
    )
    val_dataset = CachedSegmentationDataset(
        image_folder=split_paths.val_images,
        mask_folder=split_paths.val_masks,
        image_size=config.image_size,
        cache_images=False,
        augment=False,
    )

    batch_size = 8
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = UNet(in_channels=config.channels).to(device)
    best_ckpt = CHECKPOINT_PATH / "best_model.pth"
    if best_ckpt.exists():
        print(f"Loading weights from {best_ckpt} for fine-tuning...")
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    criterion = BCEDiceLoss(bce_weight=0.25, dice_weight=0.75)
    epochs = 12
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_dice = 0.6068  # previous baseline
    best_iou = 0.4356
    best_epoch = 30
    best_state_dict = copy.deepcopy(model.state_dict())

    history_logger = HistoryLogger(output_dir=OUTPUT_PATH, prefix="large")

    print("\nStarting Training Optimization Loop...")
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / max(len(train_loader), 1)

        val_stats = evaluate_loader(model, val_loader, criterion, device)
        val_dice = val_stats["dice"]
        val_iou = val_stats["iou"]
        epoch_time = time.perf_counter() - epoch_start

        current_global_epoch = 30 + epoch
        print(
            f"Epoch {current_global_epoch:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_stats['loss']:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} | "
            f"Prec: {val_stats['precision']:.4f} | Rec: {val_stats['recall']:.4f} | Time: {epoch_time:.1f}s"
        )

        history_logger.append(
            epoch=current_global_epoch,
            training_loss=train_loss,
            validation_dice=val_dice,
            validation_iou=val_iou,
        )

        if val_dice > best_dice:
            best_dice = val_dice
            best_iou = val_iou
            best_epoch = current_global_epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(best_state_dict, CHECKPOINT_PATH / "best_model.pth")
            print(f" -> Best Checkpoint Updated! Best Dice: {best_dice:.4f}, Best IoU: {best_iou:.4f}")

        torch.save(model.state_dict(), CHECKPOINT_PATH / "last_model.pth")

    print("\nFinalizing Metrics & Visualizations...")
    # Load best model for final evaluation
    model.load_state_dict(best_state_dict)
    final_best_metrics = evaluate_loader(model, val_loader, criterion, device)

    write_metrics_csv(
        [
            {
                "checkpoint": "best",
                "epoch": best_epoch,
                **final_best_metrics,
            },
            {
                "checkpoint": "last",
                "epoch": 30 + epochs,
                **val_stats,
            },
        ],
        OUTPUT_PATH / "final_metrics.csv",
    )

    generate_experiment_plots(
        [
            {
                "epoch": row["epoch"],
                "training_loss": row["training_loss"],
                "validation_dice": row["validation_dice"],
                "validation_iou": row["validation_iou"],
            }
            for row in history_logger.rows()
        ],
        OUTPUT_PATH / "plots",
    )

    print("\n==========================================")
    print("Optimization Complete!")
    print(f"Initial Baseline Dice: 0.6068 -> Improved Best Dice: {best_dice:.4f}")
    print(f"Initial Baseline IoU : 0.4356 -> Improved Best IoU : {best_iou:.4f}")
    print(f"Best Model Saved to: {CHECKPOINT_PATH / 'best_model.pth'}")
    print("==========================================")


if __name__ == "__main__":
    main()
