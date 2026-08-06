"""Uncertainty measures for binary medical image segmentation predictions."""

import torch


def least_confidence(probabilities):
    """Return mean uncertainty based on the least confident class prediction.

    For binary segmentation, every pixel has foreground probability ``p`` and
    background probability ``1 - p``. A pixel is uncertain when neither class
    has a high probability, so its score is ``1 - max(p, 1 - p)``.
    """
    foreground_confidence = probabilities
    background_confidence = 1 - probabilities
    maximum_confidence = torch.maximum(
        foreground_confidence,
        background_confidence,
    )
    return (1 - maximum_confidence).mean().item()


def entropy(probabilities, epsilon=1e-8):
    """Return the average pixel-wise binary entropy as a Python float.

    Values are clamped only for the logarithm calculation, preventing
    numerical errors when a probability is exactly zero or one.
    """
    probabilities = probabilities.clamp(min=epsilon, max=1 - epsilon)
    pixel_entropy = -(
        probabilities * torch.log(probabilities)
        + (1 - probabilities) * torch.log(1 - probabilities)
    )
    return pixel_entropy.mean().item()


def margin_sampling(probabilities):
    """Return margin-based uncertainty, where higher values are more uncertain.

    The binary foreground/background confidence margin is ``abs(2p - 1)``.
    Since a smaller margin means greater uncertainty, this function returns
    ``1 - margin`` so its output can be sorted in descending uncertainty order.
    """
    confidence_margin = torch.abs((2 * probabilities) - 1)
    return (1 - confidence_margin).mean().item()
