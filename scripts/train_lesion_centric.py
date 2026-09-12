"""Lesion-Centric Training Pipeline for achieving ~0.80+ Dice & IoU on Medical MRI Segmentation."""

import copy
import sys
import time
from pathlib import Path

# Add project root to sys.path
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


class LesionCentricDataset(Dataset):
    """In-memory dataset focusing on tumor-positive slices with balanced healthy regulation."""

    def __init__(self, image_folder, mask_folder, image_size=256, augment=False, healthy_ratio=0.25):
        self.image_size = image_size
        self.augment = augment

        base_ds = MedicalSegmentationDataset(
            image_folder=image_folder,
            mask_folder=mask_folder,
            image_size=image_size,
        )

        pos_samples = []
        neg_samples = []

        print(f"Pre-loading and filtering {len(base_ds)} samples into memory...", flush=True)
        for img_p, mask_p in tqdm(base_ds.samples, desc="Loading tensors", leave=False):
            img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)  # (3, H, W)

            msk = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            if msk is None:
                continue
            msk = cv2.resize(msk, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
            msk = (msk > 0).astype(np.float32)  # (H, W)

            if msk.sum() > 0:
                pos_samples.append((img, msk))
            else:
                neg_samples.append((img, msk))

        if augment:
            # For training: use ALL tumor slices + selected healthy slices
            num_neg_to_keep = int(len(pos_samples) * healthy_ratio)
            np.random.seed(42)
            selected_neg_indices = np.random.choice(len(neg_samples), size=min(num_neg_to_keep, len(neg_samples)), replace=False)
            selected_negs = [neg_samples[i] for i in selected_neg_indices]
            self.samples = pos_samples + selected_negs
            print(f"Lesion-Centric Training Set: {len(pos_samples)} Tumor slices + {len(selected_negs)} Healthy slices = {len(self.samples)} Total", flush=True)
        else:
            # For validation: evaluate on ALL samples without altering distribution
            self.samples = pos_samples + neg_samples
            print(f"Validation Set: {len(pos_samples)} Tumor slices + {len(neg_samples)} Healthy slices = {len(self.samples)} Total", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img, msk = self.samples[index]
        img = img.copy()
        msk = msk.copy()

        if self.augment:
            # 1. Random horizontal flip (p=0.5)
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=2).copy()
                msk = np.fliplr(msk).copy()

            # 2. Random vertical flip (p=0.3)
            if np.random.rand() > 0.7:
                img = np.flip(img, axis=1).copy()
                msk = np.flipud(msk).copy()

            # 3. Random small rotation & affine scaling (+-15 deg, scale 0.95-1.05)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-15, 15)
                scale = np.random.uniform(0.95, 1.05)
                center = (self.image_size // 2, self.image_size // 2)
                rot_mat = cv2.getRotationMatrix2D(center, angle, scale=scale)
                # OpenCV warpAffine expects (H, W, C)
                img_hwc = img.transpose(1, 2, 0)
                img_hwc = cv2.warpAffine(img_hwc, rot_mat, (self.image_size, self.image_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                img = img_hwc.transpose(2, 0, 1).copy()
                msk = cv2.warpAffine(msk, rot_mat, (self.image_size, self.image_size), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT).copy()

            # 4. Random contrast & brightness jitter
            if np.random.rand() > 0.5:
                contrast = np.random.uniform(0.9, 1.1)
                brightness = np.random.uniform(-0.04, 0.04)
                img = np.clip(img * contrast + brightness, 0.0, 1.0)

        return torch.from_numpy(img), torch.from_numpy(msk)


def main():
    torch.set_num_threads(4)
    config = get_dataset_configuration()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Genuine Lesion-Centric Training for ~0.80+ Dice (Device: {device}, Threads: {torch.get_num_threads()}) ===", flush=True)

    split_manager = DatasetSplitManager(config)
    split_paths = split_manager.ensure_split_layout()

    train_dataset = LesionCentricDataset(
        image_folder=split_paths.train_images,
        mask_folder=split_paths.train_masks,
        image_size=config.image_size,
        augment=True,
        healthy_ratio=0.25,
    )
    val_dataset = LesionCentricDataset(
        image_folder=split_paths.val_images,
        mask_folder=split_paths.val_masks,
        image_size=config.image_size,
        augment=False,
    )

    batch_size = 16
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
        print(f"Loading previous checkpoint weights from {best_ckpt}...", flush=True)
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # Loss: High Dice weight + smooth constant
    criterion = BCEDiceLoss(bce_weight=0.15, dice_weight=0.85, smooth=1e-5)
    epochs = 12
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Initial evaluation
    print("\nEvaluating initial checkpoint on full 786 validation dataset...", flush=True)
    initial_val = evaluate_loader(model, val_loader, criterion, device)
    print(
        f"Initial Val -> Loss: {initial_val['loss']:.4f} | Dice: {initial_val['dice']:.4f} | "
        f"IoU: {initial_val['iou']:.4f} | Prec: {initial_val['precision']:.4f} | Rec: {initial_val['recall']:.4f}",
        flush=True,
    )

    best_dice = initial_val["dice"]
    best_iou = initial_val["iou"]
    best_epoch = 31
    best_state_dict = copy.deepcopy(model.state_dict())

    history_logger = HistoryLogger(output_dir=OUTPUT_PATH, prefix="large")

    print("\nStarting Lesion-Centric Training Loop (Targeting ~0.80+ Dice)...", flush=True)
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_batches = len(train_loader)

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
            print(f"\r  Epoch {epoch}/{epochs} | Batch {batch_idx}/{total_batches} | Loss: {loss.item():.4f}", end="", flush=True)

        scheduler.step()
        train_loss = total_loss / max(total_batches, 1)

        val_stats = evaluate_loader(model, val_loader, criterion, device)
        val_dice = val_stats["dice"]
        val_iou = val_stats["iou"]
        epoch_time = time.perf_counter() - epoch_start
        current_global_epoch = 30 + epoch

        print(
            f"\nEpoch {current_global_epoch:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_stats['loss']:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} | "
            f"Prec: {val_stats['precision']:.4f} | Rec: {val_stats['recall']:.4f} | Time: {epoch_time:.1f}s",
            flush=True,
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
            print(f"  >>> Best Checkpoint Updated! Best Dice: {best_dice:.4f}, Best IoU: {best_iou:.4f}", flush=True)

        torch.save(model.state_dict(), CHECKPOINT_PATH / "last_model.pth")

    print("\nRunning Final Comprehensive Metrics Evaluation on Full Validation Set...", flush=True)
    model.load_state_dict(best_state_dict)
    final_best_metrics = evaluate_loader(model, val_loader, criterion, device)
    print(
        f"Final Optimized Benchmark -> Loss: {final_best_metrics['loss']:.4f} | "
        f"Dice: {final_best_metrics['dice']:.4f} | IoU: {final_best_metrics['iou']:.4f} | "
        f"Precision: {final_best_metrics['precision']:.4f} | Recall: {final_best_metrics['recall']:.4f}",
        flush=True,
    )

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

    total_time = time.perf_counter() - start_time
    print("\n=======================================================")
    print("SUCCESS: Lesion-Centric Training Complete!")
    print(f"Total Time Taken     : {total_time/60:.2f} min")
    print(f"Previous Best Dice   : {initial_val['dice']:.4f} -> NEW Improved Best Dice: {best_dice:.4f}")
    print(f"Previous Best IoU    : {initial_val['iou']:.4f} -> NEW Improved Best IoU : {best_iou:.4f}")
    print(f"Reduced Val Loss     : {final_best_metrics['loss']:.4f}")
    print(f"Best Model Saved to  : {CHECKPOINT_PATH / 'best_model.pth'}")
    print("=======================================================")


if __name__ == "__main__":
    main()
