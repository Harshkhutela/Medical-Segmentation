"""Evaluation metrics for binary medical image segmentation."""

import torch

from utils.losses import prepare_targets


def _prepare_predictions(predictions, threshold=0.5):
    """Convert raw logits into binary segmentation predictions."""
    return (torch.sigmoid(predictions) >= threshold).float()


def dice_score(predictions, targets, smooth=1.0):
    """Return the mean Dice score as a Python float.

    Args:
        predictions: Raw model logits with shape ``(N, 1, H, W)``.
        targets: Ground-truth mask tensor from the dataset.
        smooth: Small value preventing division by zero.
    """
    predicted_masks = _prepare_predictions(predictions)
    target_masks = prepare_targets(targets).to(dtype=predicted_masks.dtype)
    predicted_masks = predicted_masks.flatten(start_dim=1)
    target_masks = target_masks.flatten(start_dim=1)

    intersection = (predicted_masks * target_masks).sum(dim=1)
    denominator = predicted_masks.sum(dim=1) + target_masks.sum(dim=1)
    score = (2 * intersection + smooth) / (denominator + smooth)
    return score.mean().item()


def iou_score(predictions, targets, smooth=1.0):
    """Return the mean intersection-over-union score as a Python float.

    Args:
        predictions: Raw model logits with shape ``(N, 1, H, W)``.
        targets: Ground-truth mask tensor from the dataset.
        smooth: Small value preventing division by zero.
    """
    predicted_masks = _prepare_predictions(predictions)
    target_masks = prepare_targets(targets).to(dtype=predicted_masks.dtype)
    predicted_masks = predicted_masks.flatten(start_dim=1)
    target_masks = target_masks.flatten(start_dim=1)

    intersection = (predicted_masks * target_masks).sum(dim=1)
    union = predicted_masks.sum(dim=1) + target_masks.sum(dim=1) - intersection
    score = (intersection + smooth) / (union + smooth)
    return score.mean().item()


def pixel_accuracy(predictions, targets, threshold=0.5):
    """Return the fraction of correctly classified pixels as a Python float.

    Args:
        predictions: Raw model logits with shape ``(N, 1, H, W)``.
        targets: Ground-truth mask tensor from the dataset.
        threshold: Probability threshold used to form binary predictions.
    """
    predicted_masks = _prepare_predictions(predictions, threshold)
    target_masks = prepare_targets(targets).to(dtype=predicted_masks.dtype)
    return (predicted_masks == target_masks).float().mean().item()
