"""Training helpers for large medical segmentation datasets."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from configs.config import CHECKPOINT_PATH, OUTPUT_PATH


@dataclass
class HistoryPaths:
    """Store paths used for large-dataset training history."""

    csv_path: Path
    json_path: Path


class HistoryLogger:
    """Append epoch-level metrics to CSV and JSON logs."""

    def __init__(self, output_dir: Path | None = None, prefix: str = "large"):
        """Initialize the logger with a dedicated filename prefix."""
        self.output_dir = output_dir or OUTPUT_PATH
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths = HistoryPaths(
            csv_path=self.output_dir / f"{prefix}_history.csv",
            json_path=self.output_dir / f"{prefix}_history.json",
        )
        self._ensure_files()

    def _ensure_files(self) -> None:
        """Create the history files if they do not already exist."""
        if not self.paths.csv_path.exists():
            with self.paths.csv_path.open("w", newline="", encoding="utf-8") as file:
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

        if not self.paths.json_path.exists():
            self.paths.json_path.write_text("[]", encoding="utf-8")

    def append(self, epoch: int, training_loss: float, validation_dice: float, validation_iou: float) -> None:
        """Append one epoch of training metrics."""
        with self.paths.csv_path.open("a", newline="", encoding="utf-8") as file:
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
                    "Epoch": epoch,
                    "Training Loss": training_loss,
                    "Validation Dice": validation_dice,
                    "Validation IoU": validation_iou,
                }
            )

        try:
            history = json.loads(self.paths.json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = []

        if not isinstance(history, list):
            history = []

        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_dice": validation_dice,
                "validation_iou": validation_iou,
            }
        )
        self.paths.json_path.write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

    def last_epoch(self) -> int:
        """Return the last logged epoch, or zero when no history exists."""
        try:
            rows = json.loads(self.paths.json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return 0

        if not rows:
            return 0
        last_row = rows[-1]
        return int(last_row.get("epoch", 0))

    def rows(self) -> list[dict[str, Any]]:
        """Return all history rows from JSON."""
        try:
            data = json.loads(self.paths.json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []


class EarlyStopping:
    """Monitor a validation metric and stop when it stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """Initialize the early stopping state."""
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float("-inf")
        self.counter = 0

    def step(self, score: float) -> bool:
        """Return ``True`` when training should stop early."""
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


class CheckpointManager:
    """Save, load, and resume large-dataset training checkpoints."""

    def __init__(self, checkpoint_dir: Path | None = None):
        """Store the checkpoint paths used for the advanced workflow."""
        self.checkpoint_dir = checkpoint_dir or CHECKPOINT_PATH
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.checkpoint_dir / "best_model.pth"
        self.best_checkpoint_path = self.checkpoint_dir / "best_checkpoint.pth"
        self.last_model_path = self.checkpoint_dir / "last_model.pth"
        self.last_checkpoint_path = self.checkpoint_dir / "last_checkpoint.pth"

    def _pack_state(
        self,
        model: torch.nn.Module,
        epoch: int,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        best_score: float | None = None,
        best_epoch: int | None = None,
    ) -> dict[str, Any]:
        """Create a full checkpoint dictionary."""
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        }
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            state["scheduler_state_dict"] = scheduler.state_dict()
        if best_score is not None:
            state["best_score"] = best_score
        if best_epoch is not None:
            state["best_epoch"] = best_epoch
        return state

    def save_best(
        self,
        model: torch.nn.Module,
        epoch: int,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        best_score: float | None = None,
        best_epoch: int | None = None,
    ) -> None:
        """Save the best model in compatibility and resumable formats."""
        torch.save(model.state_dict(), self.best_model_path)
        torch.save(
            self._pack_state(
                model,
                epoch,
                optimizer,
                scheduler,
                best_score,
                best_epoch,
            ),
            self.best_checkpoint_path,
        )

    def save_last(
        self,
        model: torch.nn.Module,
        epoch: int,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        best_score: float | None = None,
        best_epoch: int | None = None,
    ) -> None:
        """Save the latest training state."""
        torch.save(model.state_dict(), self.last_model_path)
        torch.save(
            self._pack_state(
                model,
                epoch,
                optimizer,
                scheduler,
                best_score,
                best_epoch,
            ),
            self.last_checkpoint_path,
        )

    def resume(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
    ) -> tuple[int, float, int]:
        """Restore the most recent checkpoint if available."""
        checkpoint_path = (
            self.last_checkpoint_path
            if self.last_checkpoint_path.exists()
            else self.best_checkpoint_path
        )
        if not checkpoint_path.exists():
            return 1, float("-inf"), 0

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        next_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_score = float(checkpoint.get("best_score", float("-inf")))
        best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0)))
        return max(next_epoch, 1), best_score, best_epoch

    def list_available_checkpoints(self) -> list[Path]:
        """Return all known checkpoint files that currently exist."""
        return [
            path
            for path in [
                self.best_model_path,
                self.best_checkpoint_path,
                self.last_model_path,
                self.last_checkpoint_path,
            ]
            if path.exists()
        ]
