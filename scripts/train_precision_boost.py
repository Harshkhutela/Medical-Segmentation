"""Precision-Boosted Training to achieve ~0.80+ Dice & IoU on Medical MRI Segmentation."""

import copy
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from configs.config import CHECKPOINT_PATH, OUTPUT_PATH
from configs.dataset_config import get_dataset_configuration
from evaluation import generate_experiment_plots
from models import UNet
from utils.advanced_metrics import evaluate_loader, write_metrics_csv
from utils.dataset import MedicalSegmentationDataset
from utils.dataset_splitter import DatasetSplitManager
from utils.losses import BCEDiceLoss
from utils.training_helpers import HistoryLogger


class BalancedMRIDataset(Dataset):
    """Dataset with tumor-positive mining and clean validation tensors."""

    def __init__(self, image_folder, mask_folder, image_size=256, is_train=True):
        self.image_size = image_size
        self.is_train = is_train

        base_ds = MedicalSegmentationDataset(
            image_folder=image_folder,
            mask_folder=mask_folder,
            image_size=image_size,
        )

        pos_samples = []
        neg_samples = []

        print(f"Loading {len(base_ds)} samples (is_train={is_train})...", flush=True)
        for img_p, mask_p in base_ds.samples:
            img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)

            msk = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            if msk is None:
                continue
            msk = cv2.resize(msk, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
            msk = (msk > 0).astype(np.float32)

            if msk.sum() > 0:
                pos_samples.append((img, msk))
            else:
                neg_samples.append((img, msk))

        if is_train:
            # Keep all positive samples + 30% healthy for balanced background suppression
            np.random.seed(42)
            num_neg = int(len(pos_samples) * 0.30)
            sel_neg_idx = np.random.choice(len(neg_samples), size=min(num_neg, len(neg_samples)), replace=False)
            self.samples = pos_samples + [neg_samples[i] for i in sel_neg_idx]
            print(f"Train samples: {len(pos_samples)} Tumor + {num_neg} Healthy = {len(self.samples)} Total", flush=True)
        else:
            self.samples = pos_samples + neg_samples
            print(f"Val samples: {len(self.samples)} Total ({len(pos_samples)} Tumor, {len(neg_samples)} Healthy)", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img, msk = self.samples[index]
        img = img.copy()
        msk = msk.copy()

        if self.is_train:
            # 1. Flip
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=2).copy()
                msk = np.fliplr(msk).copy()
            # 2. Contrast
            if np.random.rand() > 0.5:
                c = np.random.uniform(0.92, 1.08)
                img = np.clip(img * c, 0.0, 1.0)

        return torch.from_numpy(img), torch.from_numpy(msk)


def run_precision_boost():
    torch.set_num_threads(6)
    config = get_dataset_configuration()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Precision-Boosted Fine-Tuning for ~0.80+ Dice & IoU ({device}) ===", flush=True)

    split_manager = DatasetSplitManager(config)
    split_paths = split_manager.ensure_split_layout()

    train_ds = BalancedMRIDataset(split_paths.train_images, split_paths.train_masks, is_train=True)
    val_ds = BalancedMRIDataset(split_paths.val_images, split_paths.val_masks, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = UNet(in_channels=config.channels).to(device)
    best_ckpt = CHECKPOINT_PATH / "best_model.pth"
    if best_ckpt.exists():
        print(f"Loading weights from {best_ckpt}...", flush=True)
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # Balanced Loss: BCE 0.55 + Dice 0.45 suppresses false positives and sharpens Dice
    criterion = BCEDiceLoss(bce_weight=0.55, dice_weight=0.45, smooth=1e-5)
    epochs = 4
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    best_dice = 0.6653
    best_iou = 0.4985
    best_epoch = 31

    history_logger = HistoryLogger(output_dir=OUTPUT_PATH, prefix="large")

    print("\nStarting Fast Epochs...", flush=True)
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader, start=1):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            print(f"\r  Epoch {epoch}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}", end="", flush=True)

        train_loss = total_loss / len(train_loader)
        val_stats = evaluate_loader(model, val_loader, criterion, device, threshold=0.5)
        val_dice = val_stats["dice"]
        val_iou = val_stats["iou"]
        epoch_time = time.perf_counter() - epoch_start
        cur_epoch = 31 + epoch

        print(
            f"\nEpoch {cur_epoch:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_stats['loss']:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} | "
            f"Prec: {val_stats['precision']:.4f} | Rec: {val_stats['recall']:.4f} | Time: {epoch_time:.1f}s",
            flush=True,
        )

        history_logger.append(
            epoch=cur_epoch,
            training_loss=train_loss,
            validation_dice=val_dice,
            validation_iou=val_iou,
        )

        if val_dice > best_dice:
            best_dice = val_dice
            best_iou = val_iou
            best_epoch = cur_epoch
            torch.save(model.state_dict(), CHECKPOINT_PATH / "best_model.pth")
            print(f"  >>> Best Checkpoint Updated! Best Dice: {best_dice:.4f}, Best IoU: {best_iou:.4f}", flush=True)

    # Re-evaluate best model
    model.load_state_dict(torch.load(CHECKPOINT_PATH / "best_model.pth", map_location=device))
    final_metrics = evaluate_loader(model, val_loader, criterion, device, threshold=0.5)

    write_metrics_csv(
        [
            {
                "checkpoint": "best",
                "epoch": best_epoch,
                **final_metrics,
            }
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

    print("\n=======================================================")
    print(f"SUCCESS: Best Dice = {best_dice:.4f} | Best IoU = {best_iou:.4f}")
    print(f"Best Model Saved to: {CHECKPOINT_PATH / 'best_model.pth'}")
    print("=======================================================")


if __name__ == "__main__":
    run_precision_boost()
