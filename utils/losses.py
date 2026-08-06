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


class DiceLoss(nn.Module):
    """Measure the overlap error between sigmoid predictions and target masks."""

    def __init__(self, smooth=1.0):
        """Initialize Dice smoothing to avoid division by zero.

        Args:
            smooth: Small value added to the Dice numerator and denominator.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """Return one minus the mean Dice overlap for a batch of raw logits.

        Sigmoid converts raw model logits into probabilities before overlap is
        calculated. Targets are converted to binary masks when necessary.
        """
        probabilities = torch.sigmoid(logits)
        targets = prepare_targets(targets).to(dtype=probabilities.dtype)

        probabilities = probabilities.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)
        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combine pixel-wise BCE loss with region-overlap Dice loss."""

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        """Initialize the component losses and their relative weights.

        Args:
            bce_weight: Contribution of binary cross-entropy loss.
            dice_weight: Contribution of Dice loss.
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, logits, targets):
        """Return the weighted sum of BCEWithLogits and Dice losses."""
        targets = prepare_targets(targets).to(dtype=logits.dtype)
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return (self.bce_weight * bce) + (self.dice_weight * dice)
