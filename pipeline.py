"""End-to-end research pipeline for the medical segmentation project."""

from __future__ import annotations

import csv
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch
from torch.optim import Adam

from active_learning.query_strategy import select_top_k
from active_learning.sampler import ActiveSampler
from configs.config import (
    BASE_DIR,
    CHECKPOINT_PATH,
    DATASET_PATH,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    OUTPUT_PATH,
)
from evaluation import ExperimentEvaluator, generate_experiment_plots
from inference import load_model
from semi_supervised.pseudo_label import PseudoLabelGenerator
from semi_supervised.retrain import prepare_retraining_dataset
from train import (
    append_history_row,
    create_data_loaders,
    train_one_epoch,
    validate,
    main as train_main,
)
from utils.dataset import MedicalSegmentationDataset
from utils.losses import BCEDiceLoss


class Tee:
    """Mirror printed output to the console and a log file."""

    def __init__(self, *streams):
        """Store the output streams that should receive the same text."""
        self.streams = streams

    def write(self, data):
        """Write text to every stream."""
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        """Flush every stream."""
        for stream in self.streams:
            stream.flush()


def count_dataset_images(directory: Path) -> int:
    """Count supported image files in a directory if it exists."""
    if not directory.is_dir():
        return 0

    supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sum(
        1
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in supported_extensions
    )


def save_scores_csv(scores, csv_file):
    """Save uncertainty scores to CSV for reproducible research logs."""
    fieldnames = [
        "filename",
        "least_confidence",
        "entropy",
        "margin",
        "final_score",
    ]
    with csv_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scores)


def load_epoch_offset(history_file: Path) -> int:
    """Return how many history rows already exist before retraining starts."""
    if not history_file.exists():
        return 0

    with history_file.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return sum(1 for _ in reader)


def retrain_on_merged_dataset(checkpoint_file: Path) -> bool:
    """Fine-tune the model on the merged retraining dataset."""
    retraining_directory = DATASET_PATH / "retraining"
    retraining_dataset = MedicalSegmentationDataset(
        image_folder=retraining_directory / "images",
        mask_folder=retraining_directory / "masks",
    )

    if len(retraining_dataset) == 0:
        print("No retraining pairs were found. Skipping retraining stage.")
        return False

    try:
        training_loader, validation_loader = create_data_loaders(
            retraining_dataset
        )
    except ValueError as error:
        print(f"Error during retraining dataset split: {error}")
        return False

    history_file = OUTPUT_PATH / "history.csv"
    history_json_file = OUTPUT_PATH / "history.json"
    epoch_offset = load_epoch_offset(history_file)

    try:
        model = load_model(checkpoint_file)
    except (RuntimeError, OSError) as error:
        print(f"Error: unable to load checkpoint for retraining. {error}")
        return False

    model.train()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = BCEDiceLoss()
    best_dice = float("-inf")

    print("Retraining Model...")
    for epoch in range(1, EPOCHS + 1):
        logged_epoch = epoch_offset + epoch
        print(f"Retraining Epoch {epoch}/{EPOCHS}")
        training_loss = train_one_epoch(
            model,
            training_loader,
            optimizer,
            criterion,
        )
        validation_dice, validation_iou = validate(model, validation_loader)

        print(f"Training Loss : {training_loss:.4f}")
        print(f"Validation Dice : {validation_dice:.4f}")
        print(f"Validation IoU : {validation_iou:.4f}")

        append_history_row(
            history_file,
            history_json_file,
            {
                "epoch": logged_epoch,
                "training_loss": training_loss,
                "validation_dice": validation_dice,
                "validation_iou": validation_iou,
            },
        )

        if validation_dice > best_dice:
            best_dice = validation_dice
            torch.save(model.state_dict(), checkpoint_file)
            print("Model Saved")
        print()

    print("Retraining Complete")
    return True


def main():
    """Run the full research workflow from training through evaluation."""
    logs_dir = BASE_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "pipeline_log.txt"

    with log_file.open("a", encoding="utf-8") as log_handle:
        with redirect_stdout(Tee(sys.stdout, log_handle)):
            print("==========================================")
            print("Research Pipeline")
            print("==========================================\n")

            labeled_count = count_dataset_images(DATASET_PATH / "images")
            unlabeled_count = count_dataset_images(DATASET_PATH / "unlabeled")
            print("Step 1: Verifying dataset folders...")
            print(f"Labeled Images : {labeled_count}")
            print(f"Unlabeled Images : {unlabeled_count}\n")

            print("Step 2: Training the model...")
            train_main()
            print("Step 2 Completed\n")

            checkpoint_file = CHECKPOINT_PATH / "best_model.pth"
            if not checkpoint_file.is_file():
                print("Error: checkpoint not found after training.")
                return

            print("Step 3: Running inference on unlabeled images...")
            sampler = ActiveSampler(checkpoint_file=checkpoint_file)
            try:
                sampler.load_model()
                scores = sampler.score_unlabeled_images()
            except (RuntimeError, OSError) as error:
                print(f"Error during unlabeled inference: {error}")
                return

            print(f"Inference Completed : {len(scores)} images\n")

            print("Step 4: Computing uncertainty scores...")
            print(f"Uncertainty Scores Computed : {len(scores)}\n")

            print("Step 5: Selecting Top-K uncertain images...")
            top_k = select_top_k(scores, k=5)
            print(f"Selected Images : {len(top_k)}\n")

            OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
            active_learning_csv = OUTPUT_PATH / "active_learning_scores.csv"
            save_scores_csv(scores, active_learning_csv)

            print("Step 6: Generating pseudo labels...")
            generator = PseudoLabelGenerator(checkpoint_file=checkpoint_file)
            try:
                generator.load_model()
                pseudo_stats = generator.generate_pseudo_labels()
            except (RuntimeError, OSError) as error:
                print(f"Error during pseudo-label generation: {error}")
                return

            print(f"Pseudo Labels : {pseudo_stats['generated']}\n")

            print("Step 7: Preparing retraining dataset...")
            retraining_stats = prepare_retraining_dataset()
            print(
                "Retraining Dataset Size : "
                f"{retraining_stats['final_count']}\n"
            )

            print("Step 8: Retraining model...")
            retraining_success = retrain_on_merged_dataset(checkpoint_file)
            if not retraining_success:
                print("Retraining stage did not complete successfully.\n")
            else:
                print("Step 8 Completed\n")

            print("Step 9: Running evaluation...")
            evaluator = ExperimentEvaluator()
            evaluator.ensure_output_folders()
            history = evaluator.load_training_history()
            active_learning_results = evaluator.load_active_learning_results()
            semi_supervised_report = evaluator.load_semi_supervised_report()
            evaluator.export_training_csv(history)
            generate_experiment_plots(history, evaluator.plots_dir)

            summary = evaluator.compute_summary(
                history=history,
                active_learning_results=active_learning_results,
                semi_supervised_report=semi_supervised_report,
            )
            print("Step 9 Completed\n")

            print("==========================================")
            print("Research Pipeline Completed")
            print()
            print(f"Labeled Images : {labeled_count}")
            print(f"Unlabeled Images : {unlabeled_count}")
            print(f"Selected Images : {len(top_k)}")
            print(f"Pseudo Labels : {pseudo_stats['generated']}")
            print(f"Best Dice : {summary['best_dice']:.4f}")
            print(f"Best IoU : {summary['best_iou']:.4f}")
            print("Outputs Saved")
            print("==========================================")

            print("\nLog Saved")
            print(log_file.relative_to(BASE_DIR))


if __name__ == "__main__":
    main()
