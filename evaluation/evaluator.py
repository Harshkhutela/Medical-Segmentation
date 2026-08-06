"""Utilities for reading experiment logs and summarizing results."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from configs.config import OUTPUT_PATH


class ExperimentEvaluator:
    """Collect experiment artifacts and turn them into a compact summary."""

    def __init__(self, output_dir: Path | None = None):
        """Store the output directory used by the project."""
        self.output_dir = output_dir or OUTPUT_PATH
        self.plots_dir = self.output_dir / "plots"
        self.experiment_csv = self.output_dir / "experiment_results.csv"
        self.active_learning_csv = self.output_dir / "active_learning_scores.csv"
        self.semi_supervised_report = (
            self.output_dir / "semi_supervised_report.txt"
        )

    def ensure_output_folders(self) -> None:
        """Create the output folders used by the evaluation pipeline."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def load_training_history(self) -> list[dict[str, float | int]]:
        """Load epoch-level training history from available experiment logs."""
        candidates = [
            self.output_dir / "large_history.csv",
            self.output_dir / "large_history.json",
            self.output_dir / "history.csv",
            self.output_dir / "history.json",
            self.output_dir / "final_metrics.csv",
            self.experiment_csv,
            self.output_dir / "training_history.csv",
            self.output_dir / "train_history.csv",
            self.output_dir / "training_log.txt",
            self.output_dir / "train_log.txt",
            self.output_dir / "training_output.txt",
        ]

        for candidate in candidates:
            if not candidate.exists():
                continue

            if candidate.suffix.lower() == ".csv":
                history = self._load_history_from_csv(candidate)
            elif candidate.suffix.lower() in {".txt", ".log"}:
                history = self._load_history_from_text(candidate)
            elif candidate.suffix.lower() == ".json":
                history = self._load_history_from_json(candidate)
            else:
                history = []

            if history:
                return history

        return []

    def load_active_learning_results(self) -> list[dict[str, Any]]:
        """Load uncertainty scores if active-learning results are available."""
        if not self.active_learning_csv.exists():
            return []

        with self.active_learning_csv.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            return [row for row in reader]

    def load_semi_supervised_report(self) -> dict[str, Any]:
        """Load the semi-supervised summary report if it exists."""
        if not self.semi_supervised_report.exists():
            return {}

        report_data: dict[str, Any] = {}
        for line in self.semi_supervised_report.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()
            report_data[key] = self._coerce_value(value)

        return report_data

    def export_training_csv(
        self,
        history: list[dict[str, float | int]],
    ) -> Path:
        """Export the normalized experiment history to the expected CSV."""
        self.ensure_output_folders()

        fieldnames = [
            "Epoch",
            "Training Loss",
            "Validation Dice",
            "Validation IoU",
        ]

        with self.experiment_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in history:
                writer.writerow(
                    {
                        "Epoch": row.get("epoch", ""),
                        "Training Loss": row.get("training_loss", ""),
                        "Validation Dice": row.get("validation_dice", ""),
                        "Validation IoU": row.get("validation_iou", ""),
                    }
                )

        return self.experiment_csv

    def compute_summary(
        self,
        history: list[dict[str, float | int]],
        active_learning_results: list[dict[str, Any]] | None = None,
        semi_supervised_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute a concise summary from the available experiment artifacts."""
        active_learning_results = active_learning_results or []
        semi_supervised_report = semi_supervised_report or {}

        best_dice = self._best_metric(history, "validation_dice")
        best_iou = self._best_metric(history, "validation_iou")
        best_epoch = self._best_epoch(history, "validation_dice")
        average_training_loss = self._average_metric(history, "training_loss")
        final_epoch = int(history[-1]["epoch"]) if history else 0
        total_epochs = len(history)

        pseudo_labels_generated = semi_supervised_report.get(
            "generated_pseudo_labels",
            semi_supervised_report.get("generated", 0),
        )
        most_uncertain_images = len(active_learning_results)

        return {
            "best_dice": best_dice,
            "best_iou": best_iou,
            "best_epoch": best_epoch,
            "average_training_loss": average_training_loss,
            "final_epoch": final_epoch,
            "total_epochs": total_epochs,
            "pseudo_labels_generated": pseudo_labels_generated,
            "most_uncertain_images_evaluated": most_uncertain_images,
        }

    def _load_history_from_csv(self, path: Path) -> list[dict[str, float | int]]:
        """Read a CSV history file and normalize its column names."""
        history: list[dict[str, float | int]] = []
        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                parsed = self._normalize_history_row(row)
                if parsed:
                    history.append(parsed)
        return history

    def _load_history_from_text(self, path: Path) -> list[dict[str, float | int]]:
        """Parse a plain-text training log created from console output."""
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        history: list[dict[str, float | int]] = []
        current: dict[str, float | int] = {}

        epoch_pattern = re.compile(r"Epoch\s+(\d+)(?:/\d+)?")
        loss_pattern = re.compile(r"Training Loss\s*:\s*([0-9eE+\-.]+)")
        dice_pattern = re.compile(r"Validation Dice\s*:\s*([0-9eE+\-.]+)")
        iou_pattern = re.compile(r"Validation IoU\s*:\s*([0-9eE+\-.]+)")

        for line in lines:
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                if current:
                    parsed = self._finalize_history_row(current)
                    if parsed:
                        history.append(parsed)
                    current = {}
                current["epoch"] = int(epoch_match.group(1))

            loss_match = loss_pattern.search(line)
            if loss_match:
                current["training_loss"] = float(loss_match.group(1))

            dice_match = dice_pattern.search(line)
            if dice_match:
                current["validation_dice"] = float(dice_match.group(1))

            iou_match = iou_pattern.search(line)
            if iou_match:
                current["validation_iou"] = float(iou_match.group(1))

        if current:
            parsed = self._finalize_history_row(current)
            if parsed:
                history.append(parsed)

        return history

    def _load_history_from_json(self, path: Path) -> list[dict[str, float | int]]:
        """Read a JSON history file if one is available."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

        if isinstance(data, dict):
            data = data.get("history", [])

        history: list[dict[str, float | int]] = []
        if isinstance(data, list):
            for index, item in enumerate(data, start=1):
                if isinstance(item, dict):
                    normalized = self._normalize_history_row(item)
                    if normalized:
                        history.append(normalized)
                    continue

                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    history.append(
                        {
                            "epoch": index,
                            "training_loss": float(item[0]),
                            "validation_dice": float(item[1]),
                            "validation_iou": float(item[2]),
                        }
                    )

        return history

    def _normalize_history_row(
        self,
        row: dict[str, Any],
    ) -> dict[str, float | int]:
        """Map a loose dictionary of values to the expected history schema."""
        key_map = {
            "epoch": ["epoch", "Epoch"],
            "training_loss": [
                "training_loss",
                "Training Loss",
                "train_loss",
                "loss",
            ],
            "validation_dice": [
                "validation_dice",
                "Validation Dice",
                "val_dice",
                "dice",
            ],
            "validation_iou": [
                "validation_iou",
                "Validation IoU",
                "val_iou",
                "iou",
            ],
        }

        normalized: dict[str, float | int] = {}
        for target_key, aliases in key_map.items():
            for alias in aliases:
                if alias in row and row[alias] not in {"", None}:
                    value = row[alias]
                    if target_key == "epoch":
                        normalized[target_key] = int(float(value))
                    else:
                        normalized[target_key] = float(value)
                    break

        if "epoch" not in normalized:
            normalized["epoch"] = 0

        if "training_loss" not in normalized:
            normalized["training_loss"] = 0.0

        if "validation_dice" not in normalized:
            normalized["validation_dice"] = 0.0

        if "validation_iou" not in normalized:
            normalized["validation_iou"] = 0.0

        if not any(key in row for key in ("epoch", "Epoch")):
            return self._finalize_history_row(normalized)

        return normalized

    def _finalize_history_row(
        self,
        row: dict[str, float | int],
    ) -> dict[str, float | int] | None:
        """Ensure a row contains only numeric values and an epoch index."""
        if not row:
            return None

        epoch = int(row.get("epoch", 0))
        if epoch <= 0:
            epoch = len(row)

        return {
            "epoch": epoch,
            "training_loss": float(row.get("training_loss", 0.0)),
            "validation_dice": float(row.get("validation_dice", 0.0)),
            "validation_iou": float(row.get("validation_iou", 0.0)),
        }

    def _coerce_value(self, value: str) -> Any:
        """Convert a text value from a report into a numeric type when possible."""
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    def _best_metric(self, history: list[dict[str, float | int]], key: str) -> float:
        """Return the maximum value for a metric or zero if no history exists."""
        values = [float(row[key]) for row in history if key in row]
        if not values:
            return 0.0
        return max(values)

    def _best_epoch(self, history: list[dict[str, float | int]], key: str) -> int:
        """Return the epoch index of the best value for a metric."""
        if not history:
            return 0

        best_row = max(history, key=lambda row: float(row.get(key, 0.0)))
        return int(best_row.get("epoch", 0))

    def _average_metric(
        self,
        history: list[dict[str, float | int]],
        key: str,
    ) -> float:
        """Return the average of a metric across epochs for convenience."""
        values = [float(row[key]) for row in history if key in row]
        if not values:
            return 0.0
        return mean(values)
