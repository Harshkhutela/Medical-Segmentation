"""Loss functions for binary medical image segmentation."""

import torch
from torch import nn


def prepare_targets(targets):
    """Convert masks to binary float tensors with a channel dimension.

    The existing dataset returns masks with shape ``(N, H, W)`` and can contain
    pixel values in the 0--255 range. This helper creates targets appropriate
    for binary segmentation losses, with shape ``(N, 1, H, W)``.
    """
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    return (targets > 0).float()


class FocalLoss(nn.Module):
    """Focal Loss to focus learning on hard tumor boundary examples."""

    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        targets = prepare_targets(targets).to(dtype=logits.dtype)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * ((1 - pt) ** self.gamma)
        return (focal_weight * bce_loss).mean()


class DiceLoss(nn.Module):
    """Measure the overlap error between sigmoid predictions and target masks."""

    def __init__(self, smooth=1e-5):
        """Initialize Dice smoothing to avoid division by zero.

        Args:
            smooth: Small value added to the Dice numerator and denominator.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """Return one minus the mean Dice overlap for a batch of raw logits."""
        probabilities = torch.sigmoid(logits)
        targets = prepare_targets(targets).to(dtype=probabilities.dtype)

        probabilities = probabilities.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)
        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combine pixel-wise BCE loss with region-overlap Dice loss."""

    def __init__(self, bce_weight=0.3, dice_weight=0.7, smooth=1e-5):
        """Initialize the component losses and their relative weights.

        Args:
            bce_weight: Contribution of binary cross-entropy loss.
            dice_weight: Contribution of Dice loss.
            smooth: Smoothing constant for Dice calculation.
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        """Return the weighted sum of BCEWithLogits and Dice losses."""
        targets = prepare_targets(targets).to(dtype=logits.dtype)
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return (self.bce_weight * bce) + (self.dice_weight * dice)

