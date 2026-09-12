"""Training script for the U-Net medical image segmentation model."""

import csv
import json

from torch import no_grad, save
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from configs.config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    OUTPUT_PATH,
)
from models import UNet
from utils.dataset import MedicalSegmentationDataset
from utils.losses import BCEDiceLoss
from utils.metrics import dice_score, iou_score


def create_data_loaders(dataset):
    """Split a dataset into 80% training and 20% validation loaders.

    Args:
        dataset: A populated PyTorch dataset.

    Returns:
        A tuple containing the training and validation data loaders.
    """
    dataset_size = len(dataset)
    training_size = int(dataset_size * 0.8)
    validation_size = dataset_size - training_size

    if training_size == 0 or validation_size == 0:
        raise ValueError(
            "At least two image-mask pairs are required for an 80/20 split."
        )

    training_dataset, validation_dataset = random_split(
        dataset,
        [training_size, validation_size],
    )
    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return training_loader, validation_loader


def train_one_epoch(model, data_loader, optimizer, criterion):
    """Train the model over one epoch and return the average training loss."""
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    for images, masks in progress_bar:
        # Transfer one batch to the configured CPU or GPU device.
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # Clear old gradients, calculate loss, and update model weights.
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(data_loader)


def validate(model, data_loader):
    """Evaluate the model and return average validation Dice and IoU scores."""
    model.eval()
    total_dice = 0.0
    total_iou = 0.0

    with no_grad():
        for images, masks in tqdm(data_loader, desc="Validation", leave=False):
            # Gradients are disabled because validation only evaluates the model.
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            logits = model(images)

            total_dice += dice_score(logits, masks)
            total_iou += iou_score(logits, masks)

    return total_dice / len(data_loader), total_iou / len(data_loader)


def initialize_history_files():
    """Create the training history files with the expected schema."""
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    history_csv = OUTPUT_PATH / "history.csv"
    history_json = OUTPUT_PATH / "history.json"

    if not history_csv.exists():
        with history_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "Epoch",
                    "Training Loss",
                    "Validation Dice",
                    "Validation IoU",
                ],
            )
            writer.writeheader()

    if not history_json.exists():
        history_json.write_text("[]", encoding="utf-8")

    return history_csv, history_json


def append_history_row(history_csv, history_json, row):
    """Append one epoch worth of metrics to CSV and JSON history files."""
    with history_csv.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Epoch",
                "Training Loss",
                "Validation Dice",
                "Validation IoU",
            ],
        )
        writer.writerow(
            {
                "Epoch": row["epoch"],
                "Training Loss": row["training_loss"],
                "Validation Dice": row["validation_dice"],
                "Validation IoU": row["validation_iou"],
            }
        )

    try:
        existing_history = json.loads(history_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        existing_history = []

    if not isinstance(existing_history, list):
        existing_history = []

    existing_history.append(
        {
            "epoch": row["epoch"],
            "training_loss": row["training_loss"],
            "validation_dice": row["validation_dice"],
            "validation_iou": row["validation_iou"],
        }
    )
    history_json.write_text(
        json.dumps(existing_history, indent=2),
        encoding="utf-8",
    )


def main():
    """Load data, train U-Net, and save the checkpoint with best Dice score."""
    print("Medical Segmentation Training\n")
    print("Loading Dataset...")
    dataset = MedicalSegmentationDataset()

    if len(dataset) == 0:
        print(
            "Error: no matching image-mask pairs were found. "
            "Check the dataset paths in configs/config.py."
        )
        return

    try:
        training_loader, validation_loader = create_data_loaders(dataset)
    except ValueError as error:
        print(f"Error: {error}")
        return

    print("Dataset Loaded\n")
    print(f"Training Samples : {len(training_loader.dataset)}")
    print(f"Validation Samples : {len(validation_loader.dataset)}\n")

    model = UNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = BCEDiceLoss(bce_weight=0.3, dice_weight=0.7)
    best_dice = float("-inf")
    checkpoint_file = CHECKPOINT_PATH / "best_model.pth"
    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
    history_csv, history_json = initialize_history_files()

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        training_loss = train_one_epoch(
            model,
            training_loader,
            optimizer,
            criterion,
        )
        validation_dice, validation_iou = validate(model, validation_loader)
        scheduler.step()

        print(f"Training Loss : {training_loss:.4f}")
        print(f"Validation Dice : {validation_dice:.4f}")
        print(f"Validation IoU : {validation_iou:.4f}")

        append_history_row(
            history_csv,
            history_json,
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_dice": validation_dice,
                "validation_iou": validation_iou,
            },
        )

        if validation_dice > best_dice:
            best_dice = validation_dice
            save(model.state_dict(), checkpoint_file)
            print("Model Saved")
        print()

    print("Training Complete")
    print(f"Best Dice Score : {best_dice:.4f}")
    print("Checkpoint Saved At :")
    print(checkpoint_file)


if __name__ == "__main__":
    main()
