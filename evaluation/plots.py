"""Plotting helpers for segmentation experiment curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_experiment_plots(
    history: list[dict[str, float | int]],
    plots_dir: Path,
) -> dict[str, Path]:
    """Create one plot per metric and save them to the plots directory."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs = [int(row.get("epoch", index + 1)) for index, row in enumerate(history)]
    losses = [float(row.get("training_loss", 0.0)) for row in history]
    dices = [float(row.get("validation_dice", 0.0)) for row in history]
    ious = [float(row.get("validation_iou", 0.0)) for row in history]

    saved_paths = {
        "loss": plots_dir / "loss_curve.png",
        "dice": plots_dir / "dice_curve.png",
        "iou": plots_dir / "iou_curve.png",
    }

    _save_single_curve(
        epochs,
        losses,
        saved_paths["loss"],
        title="Training Loss vs Epoch",
        x_label="Epoch",
        y_label="Training Loss",
    )
    _save_single_curve(
        epochs,
        dices,
        saved_paths["dice"],
        title="Validation Dice vs Epoch",
        x_label="Epoch",
        y_label="Validation Dice",
    )
    _save_single_curve(
        epochs,
        ious,
        saved_paths["iou"],
        title="Validation IoU vs Epoch",
        x_label="Epoch",
        y_label="Validation IoU",
    )

    return saved_paths


def _save_single_curve(
    epochs: list[int],
    values: list[float],
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """Save a single matplotlib figure for one metric curve."""
    figure = plt.figure(figsize=(8, 5), dpi=300)
    axis = figure.add_subplot(1, 1, 1)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, linestyle="--", alpha=0.3)

    if epochs and values:
        axis.plot(epochs, values, linewidth=2)
    else:
        axis.text(
            0.5,
            0.5,
            "No history available",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
