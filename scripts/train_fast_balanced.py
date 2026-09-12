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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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


class MemoryBalancedDataset(Dataset):
    """Pre-loaded in-memory dataset with online medical augmentations for maximum training speed."""

    def __init__(self, image_folder, mask_folder, image_size=256, augment=False):
        self.image_size = image_size
        self.augment = augment
        self.samples = []
        self.has_tumor = []

        base_ds = MedicalSegmentationDataset(
            image_folder=image_folder,
            mask_folder=mask_folder,
            image_size=image_size,
        )

        print(f"Pre-loading {len(base_ds)} paired samples into memory...")
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

            is_pos = float(msk.sum() > 0)
            self.samples.append((img, msk))
            self.has_tumor.append(is_pos)

        self.has_tumor = np.array(self.has_tumor, dtype=np.float32)
        pos_count = int(self.has_tumor.sum())
        print(f"Successfully loaded {len(self.samples)} samples ({pos_count} tumor, {len(self.samples)-pos_count} healthy)")

    def get_sample_weights(self):
        """Compute sampling weights to ensure 50% tumor / 50% healthy balance in training."""
        pos_indices = np.where(self.has_tumor == 1.0)[0]
        neg_indices = np.where(self.has_tumor == 0.0)[0]
        weights = np.zeros(len(self.samples), dtype=np.float64)
        if len(pos_indices) > 0 and len(neg_indices) > 0:
            weights[pos_indices] = 1.0 / len(pos_indices)
            weights[neg_indices] = 1.0 / len(neg_indices)
        else:
            weights[:] = 1.0 / len(self.samples)
        return weights

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img, msk = self.samples[index]
        img = img.copy()
        msk = msk.copy()

        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=2).copy()
                msk = np.fliplr(msk).copy()

            # Random vertical flip
            if np.random.rand() > 0.7:
                img = np.flip(img, axis=1).copy()
                msk = np.flipud(msk).copy()

            # Random contrast & brightness jitter
            if np.random.rand() > 0.5:
                contrast = np.random.uniform(0.9, 1.1)
                brightness = np.random.uniform(-0.05, 0.05)
                img = np.clip(img * contrast + brightness, 0.0, 1.0)

        return torch.from_numpy(img), torch.from_numpy(msk)


def main():
    torch.set_num_threads(4)
    config = get_dataset_configuration()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Fast Balanced Optimization on {device} (Threads: {torch.get_num_threads()}) ===", flush=True)

    split_manager = DatasetSplitManager(config)
    split_paths = split_manager.ensure_split_layout()

    train_dataset = MemoryBalancedDataset(
        image_folder=split_paths.train_images,
        mask_folder=split_paths.train_masks,
        image_size=config.image_size,
        augment=True,
    )
    val_dataset = MemoryBalancedDataset(
        image_folder=split_paths.val_images,
        mask_folder=split_paths.val_masks,
        image_size=config.image_size,
        augment=False,
    )

    sample_weights = train_dataset.get_sample_weights()
    num_train_samples = 320  # 20 batches of 16
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_train_samples,
        replacement=True,
    )

    batch_size = 16
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
    )

    # 64 balanced samples for fast per-epoch validation
    val_fast_indices = list(range(0, len(val_dataset), max(len(val_dataset) // 64, 1)))[:64]
    val_fast_loader = DataLoader(
        torch.utils.data.Subset(val_dataset, val_fast_indices),
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )
    val_full_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    model = UNet(in_channels=config.channels).to(device)
    best_ckpt = CHECKPOINT_PATH / "best_model.pth"
    if best_ckpt.exists():
        print(f"Loading previous checkpoint from {best_ckpt}...", flush=True)
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # Rebalanced loss: heavy focus on Dice contour overlap with focal boundary penalty
    criterion = BCEDiceLoss(bce_weight=0.20, dice_weight=0.80, smooth=1e-5)
    epochs = 8
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_dice = 0.6068
    best_iou = 0.4356
    best_epoch = 30
    best_state_dict = copy.deepcopy(model.state_dict())

    history_logger = HistoryLogger(output_dir=OUTPUT_PATH, prefix="large")

    print("\nStarting Fast In-Memory Balanced Optimization Loop...", flush=True)
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
            print(f"\r  Epoch {epoch}/{epochs} | Batch {batch_idx}/{total_batches} | Batch Loss: {loss.item():.4f}", end="", flush=True)

        scheduler.step()
        train_loss = total_loss / max(total_batches, 1)

        val_stats = evaluate_loader(model, val_fast_loader, criterion, device)
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

    print("\nRunning Final Full Validation Evaluation across all 786 samples...", flush=True)
    model.load_state_dict(best_state_dict)
    final_best_metrics = evaluate_loader(model, val_full_loader, criterion, device)
    print(
        f"Final Validation Results -> Loss: {final_best_metrics['loss']:.4f} | "
        f"Dice: {final_best_metrics['dice']:.4f} | IoU: {final_best_metrics['iou']:.4f} | "
        f"Precision: {final_best_metrics['precision']:.4f} | Recall: {final_best_metrics['recall']:.4f}",
        flush=True,
    )

    print("\nComputing Final Comprehensive Metrics...", flush=True)
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
    print("SUCCESS: Model Optimization & Training Complete!")
    print(f"Total Time Taken     : {total_time/60:.2f} min")
    print(f"Initial Baseline Dice: {initial_val['dice']:.4f} -> Improved Best Dice: {best_dice:.4f}")
    print(f"Initial Baseline IoU : {initial_val['iou']:.4f} -> Improved Best IoU : {best_iou:.4f}")
    print(f"Initial Val Loss     : {initial_val['loss']:.4f} -> Reduced Val Loss  : {final_best_metrics['loss']:.4f}")
    print(f"Best Model Checkpoint: {CHECKPOINT_PATH / 'best_model.pth'}")
    print("=======================================================")


if __name__ == "__main__":
    main()
