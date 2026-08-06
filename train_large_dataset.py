"""Advanced training entrypoint for large real-world medical datasets."""

from __future__ import annotations

import copy
import random
import time

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from configs.config import EPOCHS, OUTPUT_PATH
from configs.dataset_config import get_dataset_configuration
from evaluation import generate_experiment_plots
from models import UNet
from utils.advanced_metrics import evaluate_loader, write_metrics_csv
from utils.dataset_splitter import CachedSegmentationDataset, DatasetSplitManager
from utils.epoch_predictions import save_epoch_prediction_examples
from utils.losses import BCEDiceLoss
from utils.training_helpers import CheckpointManager, EarlyStopping, HistoryLogger


def build_dataloader(dataset, batch_size: int, num_workers: int, pin_memory: bool, shuffle: bool) -> DataLoader:
    """Create a memory-efficient DataLoader with multi-worker support."""
    persistent_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def train_one_epoch(model, data_loader, optimizer, criterion, device, epoch, max_epochs):
    """Train the model for a single epoch using the existing optimization flow."""
    model.train()
    total_loss = 0.0
    total_batches = max(len(data_loader), 1)
    epoch_desc = f"Epoch {epoch}/{max_epochs}"

    progress_bar = tqdm(
        data_loader,
        desc=epoch_desc,
        leave=False,
        dynamic_ncols=True,
        unit="batch",
        total=total_batches,
    )
    for batch_index, (images, masks) in enumerate(progress_bar, start=1):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed_batches = batch_index
        average_batch_time = progress_bar.format_dict.get("elapsed", 0.0) / max(
            elapsed_batches,
            1,
        )
        remaining_batches = max(total_batches - batch_index, 0)
        eta_seconds = average_batch_time * remaining_batches
        progress_bar.set_postfix(
            batch=batch_index,
            loss=f"{loss.item():.4f}",
            lr=f"{current_lr:.6f}",
            eta=f"{eta_seconds:.1f}s",
        )

    return total_loss / max(len(data_loader), 1)


def prepare_datasets(config):
    """Create split datasets for train and validation."""
    split_manager = DatasetSplitManager(config)
    split_paths = split_manager.ensure_split_layout()

    train_dataset = CachedSegmentationDataset(
        image_folder=split_paths.train_images,
        mask_folder=split_paths.train_masks,
        image_size=config.image_size,
        cache_images=config.cache_images,
    )
    val_dataset = CachedSegmentationDataset(
        image_folder=split_paths.val_images,
        mask_folder=split_paths.val_masks,
        image_size=config.image_size,
        cache_images=config.cache_images,
    )

    return train_dataset, val_dataset


def main():
    """Run the large-dataset training workflow with checkpoints and metrics."""
    config = get_dataset_configuration()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.perf_counter()
    print("Medical Segmentation Large Dataset Training\n")
    print("Dataset Configuration")
    print(config.dataset_root)
    print(f"Selected Device : {device}")
    print()

    train_dataset, val_dataset = prepare_datasets(config)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: dataset split could not be created from the available data.")
        return

    print(f"Training Samples : {len(train_dataset)}")
    print(f"Validation Samples : {len(val_dataset)}\n")

    train_loader = build_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=True,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=False,
    )

    model = UNet(in_channels=config.channels).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    criterion = BCEDiceLoss()

    history_logger = HistoryLogger(output_dir=OUTPUT_PATH, prefix="large")
    checkpoint_manager = CheckpointManager()
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)

    start_epoch = 1
    best_dice = float("-inf")
    best_iou = float("-inf")
    best_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    if config.resume_training:
        start_epoch, best_dice, best_epoch = checkpoint_manager.resume(
            model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        best_state_dict = copy.deepcopy(model.state_dict())
        if start_epoch > 1:
            print(f"Resumed from epoch {start_epoch - 1}")
            print(f"Best Dice so far : {best_dice:.4f}\n")

    max_epochs = EPOCHS

    print("Training...")
    for epoch in range(start_epoch, max_epochs + 1):
        epoch_start_time = time.perf_counter()
        print(f"Epoch {epoch}/{max_epochs}")
        training_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            max_epochs,
        )
        validation_stats = evaluate_loader(
            model,
            val_loader,
            criterion,
            device,
        )
        epoch_time = time.perf_counter() - epoch_start_time
        average_epoch_time = (
            time.perf_counter() - start_time
        ) / max(epoch - start_epoch + 1, 1)
        eta_seconds = max((max_epochs - epoch) * average_epoch_time, 0.0)

        scheduler.step(validation_stats["dice"])
        history_logger.append(
            epoch=epoch,
            training_loss=training_loss,
            validation_dice=validation_stats["dice"],
            validation_iou=validation_stats["iou"],
        )

        checkpoint_manager.save_last(
            model,
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            best_score=best_dice,
            best_epoch=best_epoch,
        )

        if validation_stats["dice"] > best_dice:
            best_dice = validation_stats["dice"]
            best_iou = validation_stats["iou"]
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            print("Best checkpoint updated")

        best_model_to_save = UNet(in_channels=config.channels).to(device)
        best_model_to_save.load_state_dict(best_state_dict)
        checkpoint_manager.save_best(
            best_model_to_save,
            epoch=best_epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            best_score=best_dice,
            best_epoch=best_epoch,
        )

        print(
            f"Training Loss      : {training_loss:.4f} | "
            f"Val Dice : {validation_stats['dice']:.4f} | "
            f"Val IoU : {validation_stats['iou']:.4f}"
        )
        print(
            f"Precision          : {validation_stats['precision']:.4f} | "
            f"Recall : {validation_stats['recall']:.4f} | "
            f"F1 Score : {validation_stats['f1_score']:.4f}"
        )
        print(
            f"Epoch Time         : {epoch_time:.2f}s | "
            f"ETA : {eta_seconds / 60:.1f} min"
        )

        save_epoch_prediction_examples(
            model,
            DataLoader(
                Subset(val_dataset, [random.randrange(len(val_dataset))]),
                batch_size=1,
                shuffle=False,
                num_workers=0,
            ),
            epoch=epoch,
            output_root=OUTPUT_PATH / "epoch_predictions",
            max_samples=1,
        )

        if early_stopping.step(validation_stats["dice"]):
            print("Early stopping triggered")
            break

        print()

    print("Evaluating final checkpoints...")
    best_model = UNet(in_channels=config.channels).to(device)
    if checkpoint_manager.best_model_path.exists():
        best_model.load_state_dict(
            torch.load(checkpoint_manager.best_model_path, map_location=device)
        )
    else:
        best_model.load_state_dict(model.state_dict())
    final_best_metrics = evaluate_loader(
        best_model,
        val_loader,
        criterion,
        device,
    )
    final_last_metrics = evaluate_loader(
        model,
        val_loader,
        criterion,
        device,
    )

    final_metrics_path = OUTPUT_PATH / "final_metrics.csv"
    write_metrics_csv(
        [
            {
                "checkpoint": "best",
                "epoch": best_epoch or history_logger.last_epoch(),
                **final_best_metrics,
            },
            {
                "checkpoint": "last",
                "epoch": history_logger.last_epoch(),
                **final_last_metrics,
            },
        ],
        final_metrics_path,
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

    total_training_time = time.perf_counter() - start_time
    average_epoch_time = total_training_time / max(history_logger.last_epoch() - start_epoch + 1, 1)

    print("\n==========================================")
    print("Training Complete")
    print(f"Training Time      : {total_training_time / 60:.2f} min")
    print(f"Best Dice          : {best_dice:.4f}")
    print(f"Best IoU           : {best_iou:.4f}")
    print(f"Epoch of Best Model: {best_epoch}")
    print(f"Average Epoch Time : {average_epoch_time:.2f} s")
    print(f"Final Metrics Saved At : {final_metrics_path}")
    print(f"Best Checkpoint    : {checkpoint_manager.best_checkpoint_path}")
    print(f"Last Checkpoint    : {checkpoint_manager.last_checkpoint_path}")
    print("==========================================")


if __name__ == "__main__":
    main()
