"""Advanced segmentation metrics for large-dataset evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class SegmentationMetrics:
    """Store the standard binary segmentation evaluation metrics."""

    true_positive: float
    true_negative: float
    false_positive: float
    false_negative: float
    precision: float
    recall: float
    f1_score: float
    dice: float
    iou: float


def _prepare_binary_tensor(tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert logits or probabilities to a binary tensor."""
    if tensor.dtype.is_floating_point:
        min_value = float(tensor.min().item())
        max_value = float(tensor.max().item())
        if min_value < 0.0 or max_value > 1.0:
            tensor = torch.sigmoid(tensor)

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)

    return (tensor >= threshold).float()


def compute_confusion_counts(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return confusion matrix counts for binary segmentation."""
    predicted = _prepare_binary_tensor(predictions, threshold)
    target = targets.float()
    if target.ndim == 3:
        target = target.unsqueeze(1)
    target = (target > 0).float()

    true_positive = float((predicted * target).sum().item())
    true_negative = float(((1 - predicted) * (1 - target)).sum().item())
    false_positive = float((predicted * (1 - target)).sum().item())
    false_negative = float(((1 - predicted) * target).sum().item())

    return {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def compute_segmentation_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-8,
) -> SegmentationMetrics:
    """Compute precision, recall, F1, Dice, IoU, and confusion counts."""
    counts = compute_confusion_counts(predictions, targets, threshold)
    tp = counts["true_positive"]
    tn = counts["true_negative"]
    fp = counts["false_positive"]
    fn = counts["false_negative"]

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_score = 2 * precision * recall / (precision + recall + epsilon)
    dice = 2 * tp / (2 * tp + fp + fn + epsilon)
    iou = tp / (tp + fp + fn + epsilon)

    return SegmentationMetrics(
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        dice=dice,
        iou=iou,
    )


def evaluate_loader(
    model: torch.nn.Module,
    data_loader,
    criterion,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a model on a data loader and aggregate segmentation metrics."""
    model.eval()
    total_loss = 0.0
    running_tp = 0.0
    running_tn = 0.0
    running_fp = 0.0
    running_fn = 0.0
    total_batches = 0

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            metrics = compute_segmentation_metrics(logits, masks, threshold)

            total_loss += float(loss.item())
            running_tp += metrics.true_positive
            running_tn += metrics.true_negative
            running_fp += metrics.false_positive
            running_fn += metrics.false_negative
            total_batches += 1

    if total_batches == 0:
        return {
            "loss": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "dice": 0.0,
            "iou": 0.0,
            "true_positive": 0.0,
            "true_negative": 0.0,
            "false_positive": 0.0,
            "false_negative": 0.0,
        }

    precision = running_tp / (running_tp + running_fp + 1e-8)
    recall = running_tp / (running_tp + running_fn + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)
    dice = 2 * running_tp / (2 * running_tp + running_fp + running_fn + 1e-8)
    iou = running_tp / (running_tp + running_fp + running_fn + 1e-8)

    return {
        "loss": total_loss / total_batches,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "dice": dice,
        "iou": iou,
        "true_positive": running_tp,
        "true_negative": running_tn,
        "false_positive": running_fp,
        "false_negative": running_fn,
    }


def write_metrics_csv(rows: list[dict[str, Any]], output_path) -> None:
    """Save a metrics table to disk."""
    import csv
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "checkpoint",
        "epoch",
        "loss",
        "precision",
        "recall",
        "f1_score",
        "dice",
        "iou",
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

